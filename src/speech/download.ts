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
    /** Browser quota error fragments. */
    QUOTA_PARTS = [
        "quota",
        "no space",
        "disk full",
        "ns_error_dom_quota_reached",
    ],
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
    /** Whether the origin reached its storage quota. */
    isOverQuota = async (): Promise<boolean> => {
        try {
            if (typeof navigator === "undefined") {
                return false;
            }
            const estimate = navigator.storage?.estimate;
            if (!estimate) {
                return false;
            }
            const { quota = 0, usage = 0 } = await estimate();
            return quota > 0 && usage >= quota;
        } catch {
            return false;
        }
    },
    /** Whether a failed write ran out of storage. */
    isQuotaError = (error: Error): boolean => {
        const text = `${error.name} ${error.message}`.toLowerCase();
        return QUOTA_PARTS.some((part) => text.includes(part));
    },
    /** Download one file into the store. */
    fetchFile = async (options: {
        voice: VoiceId;
        base: string;
        file: VoiceFile;
        signal: AbortSignal;
        onBytes?: (bytes: number) => void;
    }): Promise<void> => {
        if (options.file.bundled) {
            options.onBytes?.(options.file.bytes);
            return;
        }
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
    /** True when every file of a voice is stored. */
    isVoiceCached = async (voice: VoiceId): Promise<boolean> => {
        const store = await getStore();
        for (const file of getVoiceSpec(voice).files) {
            if (file.bundled) {
                continue;
            }
            const key = fileKey({ voice, file: file.file });
            if (!(await store.exists(key))) {
                return false;
            }
        }
        return true;
    },
    /** Remove stored files without touching a load. */
    clearVoice = async (voice: VoiceId): Promise<void> => {
        const store = await getStore();
        for (const file of getVoiceSpec(voice).files) {
            await store.remove(fileKey({ voice, file: file.file }));
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
    },
    /** Remove every stored voice file. */
    clearEveryVoice = async (): Promise<void> => {
        for (const voice of listVoices()) {
            await clearVoice(voice.id);
        }
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
            await persistStorage();
            if (await isOverQuota()) {
                await clearEveryVoice();
            }
            if (token !== activeToken) {
                throw new Error(`voice load cancelled`);
            }
            report("", 0, ProgressPhase.download);
            if (await isVoiceCached(options.voice)) {
                report("", 1, ProgressPhase.verify);
                return spec;
            }
            try {
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
            } catch (error) {
                if (isQuotaError(
                    error instanceof Error ? error : new Error(String(error)),
                )) {
                    await clearEveryVoice();
                }
                throw error;
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
