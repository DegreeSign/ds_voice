import type { Language, MicrophoneCapture, MicrophoneCaptureHandle, ModelId, Quality, ThreadCount, TimestampMode } from "../tensor/types";
/** Dictation lifecycle. */
export declare enum DictationState {
    idle = "idle",
    requesting = "requesting",
    recording = "recording",
    paused = "paused",
    loading = "loading",
    working = "working"
}
/** Known microphone permission before capture. */
export declare enum DictationPermission {
    granted = "granted",
    prompt = "prompt",
    denied = "denied",
    unknown = "unknown"
}
/** Origin of a dictation failure. */
export declare enum DictationErrorSource {
    microphone = "microphone",
    model = "model",
    worker = "worker"
}
/** Message types the page posts to the dictation worker. */
export declare enum DictationRequestType {
    start = "start",
    audio = "audio",
    stop = "stop",
    cancel = "cancel"
}
/** Message types the dictation worker posts back. */
export declare enum DictationMessageType {
    progress = "progress",
    loaded = "loaded",
    partial = "partial",
    done = "done",
    error = "error"
}
/** Live state reported to the UI. */
export interface DictationDetail {
    /** Model download and load progress, 0 to 1. */
    ratio: number;
    /** True once the speech model is in memory. */
    modelReady: boolean;
    /** Microphone permission before capture. */
    permission: DictationPermission;
}
/** One dictation failure. */
export interface DictationError {
    /** Part of the pipeline that failed. */
    source: DictationErrorSource;
    /** Human readable reason. */
    message: string;
}
/** Capture factory used by the controller. */
export type DictationCapture = (options: MicrophoneCapture) => MicrophoneCaptureHandle;
/** Options for one dictation controller. */
export interface DictationOptions {
    /** Prebuilt worker asset, loaded same origin. */
    workerUrl?: string;
    /** Worker factory for bundlers and tests. */
    workerFactory?: () => Worker;
    /** Rebuild the worker from source when blocked. */
    workerFallback?: boolean;
    /** Speech model, default Whisper base. */
    model?: ModelId;
    /** Audio language, default English. */
    language?: Language;
    /** Weight precision, default `Quality.High`. */
    quality?: Quality;
    /** Worker threads, default `ThreadCount.Four`. */
    threads?: ThreadCount;
    /** Timestamp detail, default none. */
    timestamps?: TimestampMode;
    /** Live update cadence in milliseconds. */
    updateIntervalMs?: number;
    /** Partial text poll cadence in milliseconds. */
    partialIntervalMs?: number;
    /** Lifecycle and progress updates. */
    onState?: (state: DictationState, detail: DictationDetail) => void;
    /** Live text and the final text. */
    onText?: (text: string, final: boolean) => void;
    /** Reused frequency buffer for a waveform. */
    onFrequency?: (levels: Uint8Array) => void;
    /** First failure of the pipeline. */
    onError?: (error: DictationError) => void;
    /** Advanced capture override. */
    capture?: DictationCapture;
}
/** Handle over one dictation controller. */
export interface DictationController {
    /** Ask for permission and start recording. */
    start(): Promise<void>;
    /** Pause an active recording. */
    pause(): void;
    /** Resume a paused recording. */
    resume(): void;
    /** Pause or resume, whichever applies. */
    togglePause(): void;
    /** Flush the tail and wait for the final text. */
    finish(): Promise<void>;
    /** Drop the current dictation. */
    cancel(): void;
    /** Abort an active dictation. */
    reset(): void;
    /** True while dictation owns the UI. */
    active(): boolean;
    /** Current lifecycle state. */
    state(): DictationState;
    /** Detach this controller. */
    destroy(): void;
}
/** Start request posted to the dictation worker. */
export interface DictationStartRequest {
    type: DictationRequestType.start;
    id: number;
    model: ModelId;
    language: Language;
    quality: Quality;
    threads: ThreadCount;
    timestamps: TimestampMode;
    updateIntervalMs: number;
    partialIntervalMs: number;
}
/** Audio request posted to the dictation worker. */
export interface DictationAudioRequest {
    type: DictationRequestType.audio;
    id: number;
    pcm: Float32Array;
}
/** Stop request posted to the dictation worker. */
export interface DictationStopRequest {
    type: DictationRequestType.stop;
    id: number;
}
/** Cancel request posted to the dictation worker. */
export interface DictationCancelRequest {
    type: DictationRequestType.cancel;
    id: number;
}
/** Request posted to the dictation worker. */
export type DictationRequest = DictationStartRequest | DictationAudioRequest | DictationStopRequest | DictationCancelRequest;
/** Reply posted by the dictation worker. */
export interface DictationMessage {
    type: DictationMessageType;
    id?: number;
    ratio?: number;
    text?: string;
    message?: string;
}
/** Worker globals used by the dictation worker. */
export interface DictationWorkerScope {
    onmessage: ((event: MessageEvent<DictationRequest>) => void) | null;
    postMessage: (message: DictationMessage) => void;
}
