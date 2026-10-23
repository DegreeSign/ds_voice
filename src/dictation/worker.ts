// Dictation worker session loop.
import { preloadVoice, transcribe } from "../index";
import {
    DictationMessageType,
    DictationRequestType,
} from "./types";
import type {
    DictationMessage,
    DictationRequest,
    DictationStartRequest,
    DictationWorkerScope,
} from "./types";
import type { DownloadProgress, TranscribeSession } from "../tensor/types";

declare const self: DictationWorkerScope;

const
    /** Reply to the page. */
    send = (message: DictationMessage): void => self.postMessage(message),
    /** Report one download step. */
    sendProgress = (progress: DownloadProgress): void => send({
        type: DictationMessageType.progress,
        ratio: progress.ratio ?? 0,
    });

let
    session: TranscribeSession | undefined,
    activeId = 0,
    poll: ReturnType<typeof setInterval> | undefined,
    lastText = ``,
    stopping = false,
    failed = false;

const
    /** Clear the live text poll. */
    stopPoll = (): void => {
        if (!poll) {
            return;
        }
        clearInterval(poll);
        poll = undefined;
    },
    /** Report one session failure. */
    failSession = (id: number, error: Error): void => {
        if (failed || id !== activeId) {
            return;
        }
        failed = true;
        stopPoll();
        send({
            type: DictationMessageType.error,
            id,
            message: error.message,
        });
    },
    /** Start a manual feed session. */
    startSession = (request: DictationStartRequest): void => {
        stopPoll();
        session?.cancel();
        session = undefined;
        activeId = request.id;
        stopping = false;
        failed = false;
        lastText = ``;
        // keep the model cache across storage pressure
        void navigator.storage?.persist?.();
        const active = transcribe({
            model: request.model,
            language: request.language,
            timestamps: request.timestamps,
            microphone: false,
            updateIntervalMs: request.updateIntervalMs,
            onProgress: sendProgress,
        });
        session = active;
        void active.done.catch((error: Error) => {
            if (stopping) {
                return;
            }
            failSession(request.id, error);
        });
        poll = setInterval(() => {
            if (request.id !== activeId) {
                return;
            }
            const text = active.fullText().text.trim();
            if (!text || text === lastText) {
                return;
            }
            lastText = text;
            send({
                type: DictationMessageType.partial,
                id: request.id,
                text,
            });
        }, request.partialIntervalMs);
        // an extra preload call resolves when the model is in memory
        void preloadVoice({
            model: request.model,
            language: request.language,
            onProgress: sendProgress,
        }).then(
            () => {
                if (request.id === activeId) {
                    send({ type: DictationMessageType.loaded });
                }
            },
            (error: Error) => failSession(request.id, error),
        );
    },
    /** Append captured samples. */
    pushAudio = (id: number, pcm: Float32Array): void => {
        if (stopping || id !== activeId || !session) {
            return;
        }
        session.push(pcm);
    },
    /** Flush the tail and report the final text. */
    stopSession = async (id: number): Promise<void> => {
        const active = session;
        if (!active || id !== activeId) {
            send({
                type: DictationMessageType.done,
                id,
                text: ``,
            });
            return;
        }
        stopping = true;
        stopPoll();
        try {
            const transcript = await active.stop();
            send({
                type: DictationMessageType.done,
                id,
                text: transcript?.text || ``,
            });
        } catch (error) {
            const reason = error instanceof Error
                ? error
                : new Error(String(error));
            failSession(id, reason);
        } finally {
            if (id === activeId) {
                session = undefined;
                activeId = 0;
            }
        }
    },
    /** Drop the session without a final decode. */
    cancelSession = (id: number): void => {
        if (id !== activeId) {
            return;
        }
        stopping = true;
        stopPoll();
        session?.cancel();
        session = undefined;
        activeId = 0;
    },
    /** Serve dictation requests in this worker. */
    serveDictation = (): void => {
        self.onmessage = (event) => {
            const data = event.data;
            if (data.type === DictationRequestType.start) {
                startSession(data);
            } else if (data.type === DictationRequestType.audio) {
                pushAudio(data.id, data.pcm);
            } else if (data.type === DictationRequestType.stop) {
                void stopSession(data.id);
            } else if (data.type === DictationRequestType.cancel) {
                cancelSession(data.id);
            }
        };
    };

export { serveDictation };
