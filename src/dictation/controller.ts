// Headless dictation controller.
import { captureMicrophone } from "../audio/mic";
import { hasMicrophone } from "../speech/env";
import { Language, ModelId, TimestampMode } from "../tensor/types";
import {
    DictationErrorSource,
    DictationMessageType,
    DictationPermission,
    DictationRequestType,
    DictationState,
} from "./types";
import type {
    DictationController,
    DictationDetail,
    DictationMessage,
    DictationOptions,
    DictationRequest,
} from "./types";
import type { MicrophoneCaptureHandle } from "../tensor/types";
const
    /** Default live update cadence. */
    DEFAULT_UPDATE_INTERVAL_MS = 1000,
    /** Default partial text poll cadence. */
    DEFAULT_PARTIAL_INTERVAL_MS = 200,
    /** Samples that make one audio message. */
    AUDIO_BATCH_SAMPLES = 2000;

let
    /** Shared worker, one per page. */
    worker: Worker | undefined,
    /** Worker asset URL. */
    workerUrl = ``,
    /** Worker factory override. */
    workerFactory: (() => Worker) | undefined,
    /** Rebuild a blocked worker. */
    workerFallback = true,
    /** One rebuild attempt. */
    workerRetried = false,
    /** Request replayed after a rebuild. */
    workerRequest: DictationRequest | undefined,
    /** Session the worker currently serves. */
    workerSessionId = 0,
    /** Model readiness of the current session. */
    workerModelReady = false,
    /** Active session callbacks. */
    workerLoaded: (() => void) | undefined,
    workerPartial: ((text: string) => void) | undefined,
    workerText: ((text: string) => void) | undefined,
    workerProgress: ((ratio: number) => void) | undefined,
    workerError: ((modelFailed: boolean, message: string) => void) | undefined,
    /** Controller that owns the worker session. */
    activeController: DictationController | undefined,
    /** Session id counter. */
    sessionCounter = 0;

const
    /** Forget worker callbacks and drop the worker. */
    workerStop = (): void => {
        workerLoaded = workerPartial = workerText
            = workerProgress = workerError = undefined;
        workerRequest = undefined;
        workerRetried = false;
        worker?.terminate();
        worker = undefined;
        workerSessionId = 0;
        workerModelReady = false;
    },
    /** Report one worker failure. */
    workerFailed = (modelFailed: boolean, message: string): void => {
        const handler = workerError;
        workerStop();
        handler?.(modelFailed, message);
    },
    /** Rebuild the worker from source when its script is blocked. */
    workerRetry = async (): Promise<void> => {
        const request = workerRequest;
        console.log(`Dictation worker blocked, rebuilding from source`);
        worker?.terminate();
        worker = undefined;
        if (!workerUrl) {
            workerFailed(false, `Dictation worker script failed`);
            return;
        }
        try {
            const response = await fetch(workerUrl);
            if (!response.ok) {
                throw new Error(`Worker source ${response.status}`);
            }
            const source = await response.text();
            worker = workerBind(new Worker(URL.createObjectURL(new Blob(
                [source],
                { type: `text/javascript` },
            ))));
        } catch (error) {
            console.log(`Dictation worker fallback failed`, error);
            workerFailed(false, `${error}`);
            return;
        }
        if (workerRequest !== request) {
            worker?.terminate();
            worker = undefined;
            return;
        }
        if (!request) {
            return;
        }
        try {
            worker?.postMessage(request);
        } catch (error) {
            console.log(`Dictation worker unavailable`, error);
            workerFailed(false, `${error}`);
        }
    },
    /** Attach the session callbacks to a fresh worker. */
    workerBind = (instance: Worker): Worker => {
        instance.onmessage = (event: MessageEvent<DictationMessage>) => {
            const message = event.data;
            if (message.type === DictationMessageType.progress) {
                workerProgress?.(message.ratio ?? 0);
                return;
            }
            if (message.type === DictationMessageType.loaded) {
                const handler = workerLoaded;
                workerLoaded = undefined;
                workerModelReady = true;
                handler?.();
                return;
            }
            if (message.type === DictationMessageType.partial) {
                if (message.id !== workerSessionId) {
                    return;
                }
                workerModelReady = true;
                workerPartial?.(message.text ?? ``);
                return;
            }
            if (message.type === DictationMessageType.done) {
                if (message.id !== workerSessionId) {
                    return;
                }
                const handler = workerText;
                workerLoaded = workerPartial = workerText
                    = workerProgress = workerError = undefined;
                workerSessionId = 0;
                workerModelReady = true;
                handler?.(message.text ?? ``);
                return;
            }
            if (
                message.id !== undefined
                && message.id !== workerSessionId
            ) {
                return;
            }
            console.log(`Dictation worker error`, message.message);
            workerFailed(!workerModelReady, message.message ?? `Decode failed`);
        };
        instance.onerror = (event) => {
            // script load failure
            if (!event.message && workerFallback && !workerRetried) {
                workerRetried = true;
                void workerRetry();
                return;
            }
            console.log(
                event.message
                    ? `Dictation worker failed: ${event.message}`
                    : `Dictation worker script failed to load`,
                event.filename,
                event.lineno,
            );
            workerFailed(false, event.message || `Worker failed`);
        };
        return instance;
    },
    /** Worker bound to the model, created on first use. */
    workerGet = (): Worker => {
        if (worker) {
            return worker;
        }
        if (workerFactory) {
            worker = workerBind(workerFactory());
            return worker;
        }
        if (!workerUrl) {
            throw new Error(`Dictation worker URL missing`);
        }
        worker = workerBind(new Worker(workerUrl));
        return worker;
    },
    /** Send a request, retrying from source when blocked. */
    workerPost = (request: DictationRequest, retry = true): void => {
        if (retry) {
            workerRequest = request;
        }
        try {
            workerGet().postMessage(request);
        } catch (error) {
            console.log(`Dictation worker unavailable`, error);
            workerRetried = true;
            void workerRetry();
        }
    },
    /** Forward one captured audio chunk. */
    workerAudio = (id: number, pcm: Float32Array): void => {
        if (!worker) {
            return;
        }
        try {
            worker.postMessage({
                type: DictationRequestType.audio,
                id,
                pcm,
            }, [pcm.buffer]);
        } catch (error) {
            console.log(`Dictation worker unavailable`, error);
            workerFailed(false, `${error}`);
        }
    },
    /** True when the microphone API is reachable. */
    isDictationSupported = (): boolean => hasMicrophone(),
    /** Wire one UI to dictation. */
    createDictation = (options: DictationOptions): DictationController => {
        let
            state = DictationState.idle,
            modelReady = false,
            modelRatio = 0,
            permission = DictationPermission.unknown,
            sessionId = 0,
            paused = false,
            destroyed = false,
            pending: Float32Array[] = [],
            pendingSamples = 0,
            captureHandle: MicrophoneCaptureHandle | undefined,
            resolveFinish: (() => void) | undefined;

        const
            /** Current detail for the UI. */
            detail = (): DictationDetail => ({
                ratio: modelRatio,
                modelReady,
                permission,
            }),
            /** Report the current state. */
            emit = (): void => {
                options.onState?.(state, detail());
            },
            /** Merge and send buffered capture frames. */
            flushAudio = (): void => {
                if (!sessionId || paused || !pendingSamples) {
                    return;
                }
                const merged = new Float32Array(pendingSamples);
                let offset = 0;
                for (const part of pending) {
                    merged.set(part, offset);
                    offset += part.length;
                }
                pending = [];
                pendingSamples = 0;
                workerAudio(sessionId, merged);
            },
            /** Append one capture frame. */
            feedAudio = (samples: Float32Array): void => {
                if (!sessionId || paused || !samples.length) {
                    return;
                }
                pending.push(samples);
                pendingSamples += samples.length;
                if (pendingSamples >= AUDIO_BATCH_SAMPLES) {
                    flushAudio();
                }
            },
            /** Stop capture, optionally flushing the tail. */
            stopCapture = (flush: boolean): void => {
                if (flush) {
                    flushAudio();
                }
                const handle = captureHandle;
                captureHandle = undefined;
                if (handle) {
                    void handle.stop().catch(() => undefined);
                }
                pending = [];
                pendingSamples = 0;
            },
            /** Clear the active worker callbacks. */
            clearCallbacks = (): void => {
                workerLoaded = workerPartial = workerText
                    = workerProgress = workerError = undefined;
            },
            /** Drop the current dictation. */
            cancel = (): void => {
                const
                    id = sessionId,
                    wasActive = state !== DictationState.idle;
                state = DictationState.idle;
                paused = false;
                stopCapture(false);
                if (id) {
                    if (worker) {
                        workerPost({
                            type: DictationRequestType.cancel,
                            id,
                        }, false);
                    }
                    workerSessionId = 0;
                    sessionId = 0;
                }
                clearCallbacks();
                if (activeController === controller) {
                    activeController = undefined;
                }
                resolveFinish?.();
                resolveFinish = undefined;
                emit();
                if (wasActive) {
                    options.onText?.(``, true);
                }
            },
            /** Resolve the running finish promise. */
            settleFinish = (): void => {
                resolveFinish?.();
                resolveFinish = undefined;
            },
            /** Ask for permission and start recording. */
            start = async (): Promise<void> => {
                if (destroyed || state !== DictationState.idle) {
                    return;
                }
                activeController?.cancel();
                activeController = controller;
                if (!worker) {
                    workerUrl = options.workerUrl ?? workerUrl;
                    workerFactory = options.workerFactory ?? workerFactory;
                    workerFallback = options.workerFallback
                        ?? !options.workerFactory;
                }
                state = DictationState.requesting;
                modelReady = false;
                modelRatio = 0;
                permission = DictationPermission.unknown;
                paused = false;
                workerModelReady = false;
                sessionId = ++sessionCounter;
                workerSessionId = sessionId;
                emit();
                try {
                    const status = await navigator.permissions?.query({
                        name: `microphone` as PermissionName,
                    });
                    permission = status?.state === `granted`
                        ? DictationPermission.granted
                        : status?.state === `denied`
                            ? DictationPermission.denied
                            : status?.state === `prompt`
                                ? DictationPermission.prompt
                                : DictationPermission.unknown;
                } catch {
                    permission = DictationPermission.unknown;
                }
                if (state !== DictationState.requesting) {
                    return;
                }
                emit();
                workerLoaded = () => {
                    if (sessionId !== workerSessionId) {
                        return;
                    }
                    modelReady = true;
                    emit();
                };
                workerPartial = (text) => {
                    if (
                        state !== DictationState.recording
                        && state !== DictationState.paused
                    ) {
                        return;
                    }
                    if (!modelReady) {
                        modelReady = true;
                        emit();
                    }
                    options.onText?.(text, false);
                };
                workerProgress = (ratio) => {
                    if (modelReady) {
                        return;
                    }
                    modelRatio = ratio;
                    emit();
                };
                workerText = (text) => {
                    modelReady = true;
                    state = DictationState.idle;
                    sessionId = 0;
                    workerSessionId = 0;
                    paused = false;
                    if (activeController === controller) {
                        activeController = undefined;
                    }
                    clearCallbacks();
                    emit();
                    options.onText?.(text.trim(), true);
                    settleFinish();
                };
                workerError = (modelFailed, message) => {
                    sessionId = 0;
                    workerSessionId = 0;
                    cancel();
                    options.onError?.({
                        source: modelFailed
                            ? DictationErrorSource.model
                            : DictationErrorSource.worker,
                        message,
                    });
                };
                workerPost({
                    type: DictationRequestType.start,
                    id: sessionId,
                    model: options.model ?? ModelId.WhisperBase,
                    language: options.language ?? Language.en,
                    timestamps: options.timestamps ?? TimestampMode.none,
                    updateIntervalMs: options.updateIntervalMs
                        ?? DEFAULT_UPDATE_INTERVAL_MS,
                    partialIntervalMs: options.partialIntervalMs
                        ?? DEFAULT_PARTIAL_INTERVAL_MS,
                });
                if (state !== DictationState.requesting) {
                    return;
                }
                captureHandle = (options.capture ?? captureMicrophone)({
                    onChunk: feedAudio,
                    onError: (error) => {
                        if (state === DictationState.idle) {
                            return;
                        }
                        cancel();
                        options.onError?.({
                            source: DictationErrorSource.microphone,
                            message: error.message,
                        });
                    },
                    onFrequency: (levels) => {
                        if (state === DictationState.recording) {
                            options.onFrequency?.(levels);
                        }
                    },
                });
                try {
                    await captureHandle.ready;
                } catch {
                    return;
                }
                if (state !== DictationState.requesting) {
                    return;
                }
                state = DictationState.recording;
                emit();
            },
            /** Pause an active recording. */
            pause = (): void => {
                if (state !== DictationState.recording) {
                    return;
                }
                paused = true;
                state = DictationState.paused;
                if (captureHandle) {
                    void captureHandle.pause().catch(() => undefined);
                }
                emit();
            },
            /** Resume a paused recording. */
            resume = (): void => {
                if (state !== DictationState.paused) {
                    return;
                }
                paused = false;
                state = DictationState.recording;
                if (captureHandle) {
                    void captureHandle.resume().catch(() => undefined);
                }
                emit();
            },
            /** Pause or resume, whichever applies. */
            togglePause = (): void => {
                if (state === DictationState.paused) {
                    resume();
                    return;
                }
                if (state === DictationState.recording) {
                    pause();
                }
            },
            /** Flush the tail and wait for the final text. */
            finish = async (): Promise<void> => {
                if (
                    state !== DictationState.recording
                    && state !== DictationState.paused
                ) {
                    return;
                }
                if (!sessionId) {
                    cancel();
                    return;
                }
                const id = sessionId;
                state = modelReady
                    ? DictationState.working
                    : DictationState.loading;
                paused = false;
                stopCapture(true);
                emit();
                const completed = new Promise<void>((resolve) => {
                    resolveFinish = resolve;
                });
                workerPost({
                    type: DictationRequestType.stop,
                    id,
                });
                await completed;
            },
            /** Detach this controller. */
            destroy = (): void => {
                if (destroyed) {
                    return;
                }
                cancel();
                destroyed = true;
            };

        const controller: DictationController = {
            start,
            pause,
            resume,
            togglePause,
            finish,
            cancel,
            reset: cancel,
            active: () => state !== DictationState.idle,
            state: () => state,
            destroy,
        };

        return controller;
    };

export { createDictation, isDictationSupported };
