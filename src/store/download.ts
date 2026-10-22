// Model download and caching.
import {
    CHECKPOINT_IDS,
    HF_BASE,
    MODEL_SPECS,
    checkpointKey,
    resolveSpec,
} from "../models/specs";
import { CHUNK_BYTES } from "./chunk";
import { ProgressPhase } from "../tensor/types";
import type {
    DeleteModelOptions,
    DownloadProgress,
    ModelSpec,
    ModelStore,
    PrepareModelOptions,
} from "../tensor/types";
const
    /** Most range requests in flight at once. */
    MAX_PARALLEL = 4,
    /** Attempts per chunk before the download gives up. */
    MAX_ATTEMPTS = 4,
    /** First retry delay in milliseconds. */
    RETRY_BASE_MS = 250,
    /** Estimated bytes for one text file not yet cached. */
    TEXT_ESTIMATE_BYTES = 256 * 1024,
    /** Base URL without a trailing slash. */
    weightsBase = (base?: string): string =>
        (base ?? HF_BASE).replace(/\/+$/, ""),
    /** URL of one file in a repository. */
    fileUrl = (options: {
        base: string;
        repo: string;
        file: string;
    }): string => {
        const { base, repo, file } = options;
        return `${base}/${repo}/resolve/main/${file}`;
    },
    /** Total bytes declared by a content range header. */
    rangeTotal = (header: string | null): number => {
        if (!header) {
            return 0;
        }
        const match = /\/(\d+)$/.exec(header);
        return match ? Number(match[1]) : 0;
    },
    /** Pause before one retry. */
    backoff = async (attempt: number): Promise<void> => {
        await new Promise((resolve) => {
            setTimeout(resolve, RETRY_BASE_MS * 2 ** attempt);
        });
    },
    /** Fetch one byte to learn whether ranges are supported. */
    probeRange = async (url: string): Promise<{
        supported: boolean;
        total: number;
        response: Response;
    }> => {
        const response = await fetch(url, {
            headers: { Range: "bytes=0-0" },
        });
        if (!response.ok || !response.body) {
            throw new Error(`fetch failed ${response.status}`);
        }
        if (response.status === 206) {
            const total = rangeTotal(response.headers.get("content-range"));
            await response.arrayBuffer();
            return { supported: true, total, response };
        }
        return {
            supported: false,
            total: Number(response.headers.get("content-length") ?? "0"),
            response,
        };
    },
    /** Download one byte range with retries. */
    fetchRange = async (options: {
        url: string;
        start: number;
        end: number;
    }): Promise<Uint8Array> => {
        const { url, start, end } = options;
        let last: Error = new Error("range fetch failed");
        for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt += 1) {
            try {
                const response = await fetch(url, {
                    headers: {
                        Range: `bytes=${start}-${end}`,
                    },
                });
                if (response.status !== 206 || !response.body) {
                    throw new Error(`range status ${response.status}`);
                }
                const
                    bytes = new Uint8Array(
                        await response.arrayBuffer(),
                    ),
                    expected = end - start + 1;
                if (bytes.length !== expected) {
                    throw new Error(`range size ${bytes.length}/${expected}`);
                }
                return bytes;
            } catch (reason) {
                last = reason instanceof Error
                    ? reason
                    : new Error(String(reason));
                if (attempt + 1 < MAX_ATTEMPTS) {
                    await backoff(attempt);
                }
            }
        }
        throw last;
    },
    /** Store a whole weight response as one stream. */
    writeWholeWeight = async (options: {
        store: ModelStore;
        key: string;
        response: Response;
        onBytes?: (bytes: number) => void;
    }): Promise<number> => {
        const
            { store, key, response, onBytes } = options;
        if (!response.body) {
            throw new Error("weight response body missing");
        }
        let received = 0;
        await store.writeStream({
            key,
            stream: response.body,
            onProgress: (size) => {
                received += size;
                onBytes?.(size);
            },
        });
        return received;
    },
    /** Store a weight file, resuming and splitting when possible. */
    writeWeight = async (options: {
        store: ModelStore;
        spec: ModelSpec;
        file: string;
        url: string;
        onBytes?: (bytes: number) => void;
    }): Promise<number> => {
        const
            { store, spec, file, url, onBytes } = options,
            key = checkpointKey({ checkpoint: spec.checkpoint, file }),
            parts = store.parts,
            expected = spec.weightBytes;
        let probe: Awaited<ReturnType<typeof probeRange>>;
        try {
            probe = await probeRange(url);
        } catch {
            return await writeWholeWeight({
                store,
                key,
                response: await fetch(url),
                onBytes,
            });
        }
        if (!parts || probe.total <= 0) {
            return await writeWholeWeight({
                store,
                key,
                response: await fetch(url),
                onBytes,
            });
        }
        if (!probe.supported) {
            return await writeWholeWeight({
                store,
                key,
                response: probe.response,
                onBytes,
            });
        }
        const
            total = expected > 0 ? expected : probe.total,
            count = Math.ceil(total / CHUNK_BYTES);
        let sizes = await parts.readSizes(key);
        if (!sizes || sizes.length !== count) {
            const aligned: number[] = [];
            for (let index = 0; index < count; index += 1) {
                const size = sizes?.[index] ?? 0;
                aligned.push(size);
            }
            sizes = aligned;
        }
        await parts.writeSizes({ key, sizes: sizes.slice() });
        const
            pending: number[] = [],
            local = sizes;
        for (let index = 0; index < count; index += 1) {
            const start = index * CHUNK_BYTES,
                expectedSize = Math.min(CHUNK_BYTES, total - start);
            if (local[index] !== expectedSize) {
                pending.push(index);
            }
        }
        let manifest: Promise<void> = Promise.resolve();
        let failure: Error | null = null;
        const save = (): Promise<void> => {
            manifest = manifest.then(async () => {
                await parts.writeSizes({ key, sizes: local.slice() });
            });
            return manifest;
        };
        const worker = async (): Promise<void> => {
            for (;;) {
                if (failure) {
                    return;
                }
                const index = pending.shift();
                if (index === undefined) {
                    return;
                }
                const
                    start = index * CHUNK_BYTES,
                    end = Math.min(total, start + CHUNK_BYTES) - 1;
                try {
                    const bytes = await fetchRange({ url, start, end });
                    await parts.write({ key, index, bytes });
                    local[index] = bytes.length;
                    await save();
                    onBytes?.(bytes.length);
                } catch (reason) {
                    failure = reason instanceof Error
                        ? reason
                        : new Error(String(reason));
                    return;
                }
            }
        };
        const workers: Promise<void>[] = [];
        for (
            let index = 0;
            index < Math.min(MAX_PARALLEL, pending.length);
            index += 1
        ) {
            workers.push(worker());
        }
        await Promise.all(workers);
        await manifest;
        if (failure) {
            // a lost range endpoint falls back to one stream
            return await writeWholeWeight({
                store,
                key,
                response: await fetch(url),
                onBytes,
            });
        }
        let loaded = 0;
        for (const size of local) {
            loaded += size;
        }
        return loaded;
    },
    /** Download one model file when absent. */
    downloadFile = async (options: {
        store: ModelStore;
        spec: ModelSpec;
        file: string;
        base: string;
        onBytes?: (bytes: number) => void;
    }): Promise<void> => {
        const
            { store, spec, file, base } = options,
            key = checkpointKey({ checkpoint: spec.checkpoint, file }),
            isWeight = file === spec.weightFile;
        if (isWeight && store.hasBinary) {
            if (await store.hasBinary(key)) {
                return;
            }
        } else if (await store.exists(key)) {
            return;
        }
        const
            repo = isWeight ? spec.repo : spec.textRepo ?? spec.repo,
            url = fileUrl({ base, repo, file });
        if (!isWeight) {
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`fetch failed ${response.status} ${file}`);
            }
            const text = await response.text();
            await store.writeText({ key, text });
            options.onBytes?.(new TextEncoder().encode(text).length);
            return;
        }
        const received = await writeWeight({
            store,
            spec,
            file,
            url,
            onBytes: options.onBytes,
        });
        if (spec.weightBytes > 0 && received !== spec.weightBytes) {
            await store.remove(key);
            throw new Error(
                `size mismatch ${file} ${received} != ${spec.weightBytes}`,
            );
        }
    },
    /** Estimated total bytes of one checkpoint. */
    checkpointBytes = async (options: {
        store: ModelStore;
        spec: ModelSpec;
    }): Promise<number> => {
        const { store, spec } = options;
        let total = 0;
        for (const file of spec.files) {
            const
                isWeight = file === spec.weightFile,
                size = store.size
                    ? await store.size(
                        checkpointKey({
                            checkpoint: spec.checkpoint,
                            file,
                        }),
                    )
                    : 0;
            if (isWeight && spec.weightBytes > 0) {
                total += spec.weightBytes;
            } else if (size > 0) {
                total += size;
            } else if (!isWeight) {
                total += TEXT_ESTIMATE_BYTES;
            }
        }
        return total;
    },
    /** Bytes already stored for one checkpoint. */
    storedBytes = async (options: {
        store: ModelStore;
        spec: ModelSpec;
    }): Promise<number> => {
        const { store, spec } = options;
        if (!store.size) {
            return 0;
        }
        let total = 0;
        for (const file of spec.files) {
            total += await store.size(
                checkpointKey({ checkpoint: spec.checkpoint, file }),
            );
        }
        return total;
    },
    /** Download every file of a model into the store. */
    prepareModel = async (
        options: PrepareModelOptions & { store: ModelStore },
    ): Promise<ModelSpec> => {
        const
            spec = resolveSpec({
                model: options.model,
                language: options.language,
            }),
            base = weightsBase(options.weightsBaseUrl),
            total = await checkpointBytes({ store: options.store, spec });
        let loaded = await storedBytes({ store: options.store, spec });
        const report = (file: string, phase: ProgressPhase): void => {
            const progress: DownloadProgress = {
                model: spec.id,
                file,
                phase,
                loaded,
                total,
                ratio: total > 0 ? Math.min(1, loaded / total) : 0,
            };
            options.onProgress?.(progress);
        };
        for (const file of spec.files) {
            const key = checkpointKey({
                checkpoint: spec.checkpoint,
                file,
            });
            const present = file === spec.weightFile
                ? await options.store.hasBinary?.(key) ??
                    await options.store.exists(key)
                : await options.store.exists(key);
            if (present) {
                report(file, ProgressPhase.verify);
                continue;
            }
            report(file, ProgressPhase.download);
            await downloadFile({
                store: options.store,
                spec,
                file,
                base,
                onBytes: (bytes) => {
                    loaded += bytes;
                    report(file, ProgressPhase.download);
                },
            });
            if (file === spec.weightFile) {
                report(file, ProgressPhase.verify);
            }
        }
        const user = options.onProgress;
        if (user) {
            user({
                model: spec.id,
                file: spec.files[spec.files.length - 1] ?? "",
                phase: ProgressPhase.verify,
                loaded: Math.max(loaded, total),
                total: Math.max(loaded, total),
                ratio: 1,
            });
        }
        return spec;
    },
    /** Remove every file of a model from the store. */
    deleteModel = async (
        options: DeleteModelOptions & { store: ModelStore },
    ): Promise<void> => {
        for (const checkpoint of CHECKPOINT_IDS) {
            const spec = MODEL_SPECS[checkpoint];
            if (spec.id !== options.model) {
                continue;
            }
            for (const file of spec.files) {
                await options.store.remove(checkpointKey({ checkpoint, file }));
            }
        }
    },
    /** True when every file of one checkpoint is stored. */
    isCheckpointCached = async (options: {
        store: ModelStore;
        spec: ModelSpec;
    }): Promise<boolean> => {
        const { store, spec } = options;
        for (const file of spec.files) {
            const key = checkpointKey({ checkpoint: spec.checkpoint, file });
            if (file === spec.weightFile) {
                const binary = store.hasBinary
                    ? await store.hasBinary(key)
                    : await store.exists(key);
                if (!binary) {
                    return false;
                }
            } else if (!(await store.exists(key))) {
                return false;
            }
        }
        return true;
    };

export {
    prepareModel,
    deleteModel,
    isCheckpointCached,
};
