// Streaming transcription session.
import { decodeSource } from "../audio/file";
import { captureMicrophone } from "../audio/mic";
import { detectPlatform, hasMicrophone } from "./env";
import { DEFAULT_MODEL } from "../models/specs";
import { Language, TimestampMode } from "../tensor/types";
import { dedupeBoundary, joinText, planChunks } from "./segment";
import { kernelThreads } from "../tensor/wasm";
import type {
    Model,
    Segment,
    SessionOptions,
    TextSnapshot,
    Transcript,
    TranscribeOptions,
    TranscribeSession,
    Word,
} from "../tensor/types";
const
    /** Working sample rate of the engine. */
    RATE = 16000,
    /** Shortest pending audio that triggers an update. */
    MIN_UPDATE_SECONDS = 0.5,
    /** Longest pending audio before a forced commit. */
    MAX_WINDOW_SECONDS = 30,
    /** Silence length that flushes the pending audio. */
    SILENCE_FLUSH_SECONDS = 5,
    /** Default update cadence for microphone sessions. */
    DEFAULT_INTERVAL_MS = 1000,
    /** Coerce a thrown value into an Error. */
    toError = (reason: unknown): Error => {
        return reason instanceof Error ? reason : new Error(String(reason));
    },
    /** Shift segments and words by an offset in seconds. */
    shiftSegments = (options: {
        segments: Segment[];
        offset: number;
    }): Segment[] => {
        const { segments, offset } = options;
        return segments.map((segment) => ({
            start: segment.start + offset,
            end: segment.end + offset,
            text: segment.text,
            words: segment.words.map((word): Word => ({
                word: word.word,
                start: word.start + offset,
                end: word.end + offset,
            })),
        }));
    },
    /** Create one streaming session. */
    createSession = (input: SessionOptions): TranscribeSession => {
        const
            options: TranscribeOptions = input.options,
            voice = input.voice,
            interval = Math.max(100, options.updateIntervalMs ??
                DEFAULT_INTERVAL_MS);
        let
            decodeMs = 0,
            pending = new Float32Array(0),
            committed = "",
            partial = "",
            committedSegments: Segment[] = [],
            language: Language = options.language ?? Language.auto,
            totalSamples = 0,
            processedSamples = 0,
            busy = false,
            finished = false,
            stopping = false,
            sourceEnded = false,
            failure: Error | null = null,
            model: Model | null = null,
            modelPromise: Promise<Model> | null = null,
            timer: ReturnType<typeof setInterval> | null = null,
            sourceStop: (() => Promise<void>) | null = null,
            resolveDone = (_transcript: Transcript): void => undefined,
            rejectDone = (_error: Error): void => undefined,
            lastModel: string = options.model ?? DEFAULT_MODEL;
        const done = new Promise<Transcript>((resolve, reject) => {
            resolveDone = resolve;
            rejectDone = reject;
        });
        void done.catch(() => undefined);
        const
            /** Join the committed and partial text. */
            compose = (): string => {
                if (!committed) {
                    return partial.trim();
                }
                if (!partial) {
                    return committed;
                }
                const merged = dedupeBoundary({
                    previous: committed,
                    next: partial,
                });
                return merged
                    ? joinText({ previous: committed, next: merged })
                    : committed;
            },
            /** Full text with decode elapsed. */
            fullText = (): TextSnapshot => {
                return {
                    text: compose(),
                    msElapsed: decodeMs,
                };
            },
            /** Clear the running update timer. */
            clearTimers = (): void => {
                if (timer) {
                    clearInterval(timer);
                    timer = null;
                }
            },
            /** Build the final transcript. */
            finish = (): void => {
                if (finished) {
                    return;
                }
                finished = true;
                clearTimers();
                if (sourceStop) {
                    const release = sourceStop;
                    sourceStop = null;
                    void release();
                }
                const
                    duration = totalSamples / RATE,
                    segments = committedSegments.length > 0
                        ? committedSegments
                        : [{
                            start: 0,
                            end: duration,
                            text: committed,
                            words: [],
                        }];
                resolveDone({
                    text: committed,
                    language,
                    duration,
                    segments,
                    meta: {
                        model: model?.name ?? lastModel,
                        rtf: duration > 0
                            ? decodeMs / 1000 / duration
                            : 0,
                        threads: kernelThreads(),
                    },
                });
            },
            /** Reject the session and stop the source. */
            fail = (reason: Error): void => {
                if (finished) {
                    return;
                }
                failure = reason;
                finished = true;
                clearTimers();
                if (sourceStop) {
                    const release = sourceStop;
                    sourceStop = null;
                    void release();
                }
                rejectDone(reason);
            },
            /** Move the pending audio into the committed transcript. */
            commit = (values: { text: string; segments: Segment[] }): void => {
                const merged = committed
                    ? dedupeBoundary({
                        previous: committed,
                        next: values.text,
                    })
                    : values.text;
                if (committed && merged) {
                    committed = joinText({ previous: committed, next: merged });
                } else if (merged) {
                    committed = merged;
                }
                partial = "";
                committedSegments = committedSegments.concat(
                    shiftSegments({
                        segments: values.segments,
                        offset: processedSamples / RATE,
                    }),
                );
                processedSamples += pending.length;
                pending = new Float32Array(0);
            },
            /** True when a silence boundary allows a commit. */
            shouldCut = (): boolean => {
                const chunks = planChunks({ pcm: pending, samplingRate: RATE });
                if (chunks.length === 0) {
                    return pending.length >= SILENCE_FLUSH_SECONDS * RATE;
                }
                const boundary = chunks[chunks.length - 1].end * RATE;
                return pending.length - boundary >= 0.5 * RATE;
            },
            /** Transcribe the pending audio once. */
            pump = async (flush = false): Promise<void> => {
                if (finished || busy || !model) {
                    return;
                }
                if (pending.length === 0) {
                    if (sourceEnded) {
                        finish();
                    }
                    return;
                }
                const minSamples = MIN_UPDATE_SECONDS * RATE;
                if (!flush && !sourceEnded && pending.length < minSamples) {
                    return;
                }
                busy = true;
                let transcript: Transcript;
                const decodeStarted = performance.now();
                try {
                    transcript = voice.transcribe({
                        model,
                        pcm: pending,
                        language: options.language,
                        timestamps: options.timestamps ?? TimestampMode.none,
                        lowLatency: options.lowLatency ?? true,
                        itn: options.itn,
                    });
                    decodeMs += performance.now() - decodeStarted;
                } catch (reason) {
                    busy = false;
                    fail(toError(reason));
                    return;
                }
                busy = false;
                if (finished) {
                    return;
                }
                partial = transcript.text;
                language = transcript.language;
                lastModel = transcript.meta.model;
                const shouldCommit = flush || sourceEnded ||
                    pending.length >= MAX_WINDOW_SECONDS * RATE ||
                    shouldCut();
                if (shouldCommit) {
                    commit({
                        text: transcript.text,
                        segments: transcript.segments,
                    });
                }
                if (sourceEnded && pending.length === 0) {
                    finish();
                }
            },
            /** Append samples to the pending buffer. */
            append = (samples: Float32Array): void => {
                if (finished || stopping || samples.length === 0) {
                    return;
                }
                const merged = new Float32Array(
                    pending.length + samples.length,
                );
                merged.set(pending, 0);
                merged.set(samples, pending.length);
                pending = merged;
                totalSamples += samples.length;
            },
            /** Drain a finite buffer in one pass. */
            feedFinite = async (pcm: Float32Array): Promise<void> => {
                modelPromise = input.voice.ensureModel({
                    model: options.model,
                    language: options.language,
                    allowedDiskSpaceMB: options.allowedDiskSpaceMB,
                    weightsBaseUrl: options.weightsBaseUrl,
                    weightCacheMB: options.weightCacheMB,
                    threads: options.threads,
                    onProgress: options.onProgress,
                });
                model = await modelPromise;
                if (stopping || finished) {
                    return;
                }
                append(pcm);
                sourceEnded = true;
                await pump();
                if (!finished && pending.length === 0) {
                    finish();
                }
            },
            /** Stop the session, flush the tail and resolve. */
            stop = async (): Promise<Transcript> => {
                if (finished) {
                    return await done;
                }
                stopping = true;
                clearTimers();
                if (sourceStop) {
                    const release = sourceStop;
                    sourceStop = null;
                    await release();
                }
                if (!model && modelPromise) {
                    try {
                        model = await modelPromise;
                    } catch {
                        // failure already reported
                    }
                }
                if (model && pending.length > 0) {
                    await pump(true);
                }
                finish();
                return await done;
            };
        const session: TranscribeSession = {
            fullText,
            stop,
            get done(): Promise<Transcript> {
                return done;
            },
            get error(): Error | null {
                return failure;
            },
        };
        if (options.pcm) {
            void feedFinite(options.pcm).catch((reason: unknown) => {
                fail(toError(reason));
            });
        } else if (options.audioFilePath) {
            const source = options.audioFilePath;
            void (async () => {
                const audio = await decodeSource({
                    audioFilePath: source,
                    platform: detectPlatform(),
                });
                await feedFinite(audio.pcm);
            })().catch((reason: unknown) => {
                fail(toError(reason));
            });
        } else if (hasMicrophone()) {
            timer = setInterval(() => {
                void pump();
            }, interval);
            modelPromise = input.voice.ensureModel({
                model: options.model,
                language: options.language,
                allowedDiskSpaceMB: options.allowedDiskSpaceMB,
                weightsBaseUrl: options.weightsBaseUrl,
                weightCacheMB: options.weightCacheMB,
                threads: options.threads,
                onProgress: options.onProgress,
            });
            void modelPromise.then(
                (value) => {
                    model = value;
                    void pump();
                },
                (reason: unknown) => {
                    fail(toError(reason));
                },
            );
            sourceStop = captureMicrophone({
                onChunk: append,
                onError: fail,
            });
        } else {
            void Promise.resolve().then(() => {
                fail(new Error("no audio source available"));
            });
        }
        return session;
    };

export { createSession };
