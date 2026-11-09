// Lazy voice download. Every voice stays cached.
import { getStore } from "../platform/store";
import { HF_BASE } from "../models/configs/specs";
import { ProgressPhase } from "../tensor/core/types";
import { getVoiceSpec, listVoices } from "./voices";
import type {
    LoadVoiceOptions,
    VoiceDownloadProgress,
    VoiceFile,
    VoiceId,
    VoiceSpec,
} from "./types";
const
    /** Attempts per voice file. */
    RETRY_ATTEMPTS = 3,
    /** First retry delay in milliseconds. */
    RETRY_DELAY_MS = 200,
    /** Store key of one voice file. */
    fileKey = (options: { voice: VoiceId; file: string }): string => {
        return `tts/${options.voice}/${options.file}`;
    },
    /** Host without a trailing slash. */
    host = (base?: string): string => (base ?? HF_BASE).replace(/\/+$/, ""),
    /** URL of one repository file. */
    fileUrl = (options: { base: string; file: VoiceFile }): string => {
        const { base, file } = options;
        return `${base}/${file.repo}/resolve/main/${file.file}`;
    },
    /** Pause before one retry. */
    pause = async (attempt: number): Promise<void> => {
        await new Promise((resolve) => {
            setTimeout(resolve, RETRY_DELAY_MS * 2 ** attempt);
        });
    },
    /** Drop a partial file after a failed stream. */
    discard = async (key: string): Promise<void> => {
        const store = await getStore();
        try {
            await store.remove(key);
        } catch {
            // Already absent.
        }
    },
    /** Ask the browser to keep the cache across eviction. */
    persistStorage = async (): Promise<void> => {
        try {
            if (typeof navigator === "undefined") {
                return;
            }
            await navigator.storage?.persist?.();
        } catch {
            // Persistence is best effort.
        }
    },
    /** Origin storage estimate, undefined when unknown. */
    storageEstimate = async (): Promise<{
        quota: number;
        usage: number;
    } | undefined> => {
        try {
            if (typeof navigator === "undefined") {
                return undefined;
            }
            const estimate = navigator.storage?.estimate;
            if (!estimate) {
                return undefined;
            }
            const { quota = 0, usage = 0 } = await estimate();
            return quota > 0 ? { quota, usage } : undefined;
        } catch {
            // Estimate is best effort.
            return undefined;
        }
    },
    /** Download one file into the store. */
    fetchFile = async (options: {
        voice: VoiceId;
        base: string;
        file: VoiceFile;
        signal: AbortSignal;
        onBytes?: (bytes: number) => void;
    }): Promise<void> => {
        const
            store = await getStore(),
            key = fileKey({ voice: options.voice, file: options.file.file });
        if (await store.exists(key)) {
            options.onBytes?.(options.file.bytes);
            return;
        }
        let last = new Error(`voice file failed`);
        for (let attempt = 0; attempt < RETRY_ATTEMPTS; attempt += 1) {
            try {
                const response = await fetch(
                    fileUrl({ base: options.base, file: options.file }),
                    { signal: options.signal },
                );
                if (!response.ok || !response.body) {
                    throw new Error(`voice file fetch failed ${response.status}`);
                }
                await store.writeStream({
                    key,
                    stream: response.body,
                    onProgress: options.onBytes,
                });
                return;
            } catch (reason) {
                last = reason instanceof Error
                    ? reason
                    : new Error(String(reason));
                await discard(key);
                if (options.signal.aborted || last.name === "AbortError") {
                    throw last;
                }
                if (attempt + 1 < RETRY_ATTEMPTS) {
                    await pause(attempt);
                }
            }
        }
        throw last;
    },
    /** Files of a voice that are not stored. */
    missingFiles = async (voice: VoiceId): Promise<VoiceFile[]> => {
        const
            store = await getStore(),
            missing: VoiceFile[] = [];
        for (const file of getVoiceSpec(voice).files) {
            const key = fileKey({ voice, file: file.file });
            if (!(await store.exists(key))) {
                missing.push(file);
            }
        }
        return missing;
    },
    /** Bytes of every file of a voice. */
    voiceBytes = (voice: VoiceId): number => {
        return getVoiceSpec(voice).files.reduce(
            (sum, file) => sum + file.bytes,
            0,
        );
    },
    /** True when every file of a voice is stored. */
    isVoiceCached = async (voice: VoiceId): Promise<boolean> =>
        (await missingFiles(voice)).length === 0,
    /** Remove stored files without touching a load. */
    clearVoice = async (voice: VoiceId): Promise<void> => {
        const store = await getStore();
        for (const file of getVoiceSpec(voice).files) {
            await store.remove(fileKey({ voice, file: file.file }));
        }
    },
    /** Drop other voices until space fits. */
    clearOtherVoices = async (options: {
        voice: VoiceId;
        needed: number;
    }): Promise<void> => {
        const estimate = await storageEstimate();
        if (!estimate) {
            return;
        }
        let free = estimate.quota - estimate.usage;
        if (free >= options.needed) {
            return;
        }
        const others = listVoices()
            .filter((spec) => spec.id !== options.voice)
            .sort((left, right) => voiceBytes(right.id) - voiceBytes(left.id));
        for (const spec of others) {
            if (free >= options.needed) {
                break;
            }
            if (!(await isVoiceCached(spec.id))) {
                continue;
            }
            await clearVoice(spec.id);
            free += voiceBytes(spec.id);
        }
    },
    /** Remove every stored file of a voice. */
    removeVoice = async (voice: VoiceId): Promise<void> => {
        if (activeVoice === voice) {
            activeToken += 1;
            activeAbort?.abort();
            activeLoad = undefined;
            activeVoice = undefined;
            activeAbort = undefined;
        }
        await clearVoice(voice);
    };

let
    /** Voice that owns the active load. */
    activeVoice: VoiceId | undefined,
    /** Single flight promise of the active load. */
    activeLoad: Promise<VoiceSpec> | undefined,
    /** Abort handle of the active load. */
    activeAbort: AbortController | undefined,
    /** Token that invalidates an older load. */
    activeToken = 0;

/** Download a voice, keeping other cached voices. */
const loadVoice = async (options: LoadVoiceOptions): Promise<VoiceSpec> => {
    if (activeLoad && activeVoice === options.voice) {
        return await activeLoad;
    }
    const
        token = ++activeToken,
        controller = new AbortController();
    activeAbort?.abort();
    activeAbort = controller;
    activeVoice = options.voice;
    const
        spec = getVoiceSpec(options.voice),
        base = host(options.weightsBaseUrl),
        total = spec.files.reduce((sum, file) => sum + file.bytes, 0);
    let loaded = 0;
    const
        /** Report one progress sample. */
        report = (file: string, ratio: number, phase: ProgressPhase): void => {
            const progress: VoiceDownloadProgress = {
                voice: options.voice,
                file,
                loaded,
                total,
                ratio: Math.max(0, Math.min(1, ratio)),
                phase,
            };
            options.onProgress?.(progress);
        },
        /** One cancellable load of the selected voice. */
        task = (async (): Promise<VoiceSpec> => {
            if (token !== activeToken) {
                throw new Error(`voice load cancelled`);
            }
            const missing = await missingFiles(options.voice);
            if (missing.length === 0) {
                return spec;
            }
            await persistStorage();
            await clearOtherVoices({
                voice: options.voice,
                needed: missing.reduce((sum, file) => sum + file.bytes, 0),
            });
            report("", 0, ProgressPhase.download);
            for (const file of spec.files) {
                if (token !== activeToken) {
                    throw new Error(`voice load cancelled`);
                }
                const before = loaded;
                await fetchFile({
                    voice: options.voice,
                    base,
                    file,
                    signal: controller.signal,
                    onBytes: (bytes) => {
                        loaded = before + bytes;
                        report(
                            file.file,
                            total > 0 ? loaded / total : 0,
                            ProgressPhase.download,
                        );
                    },
                });
                loaded = before + file.bytes;
                report(
                    file.file,
                    total > 0 ? loaded / total : 0,
                    ProgressPhase.verify,
                );
            }
            report("", 1, ProgressPhase.verify);
            return spec;
        })();
    activeLoad = task;
    try {
        return await task;
    } finally {
        if (activeToken === token) {
            activeLoad = undefined;
            activeVoice = undefined;
            activeAbort = undefined;
        }
    }
};

export { loadVoice, isVoiceCached, removeVoice };
