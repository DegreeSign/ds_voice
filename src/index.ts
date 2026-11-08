// Package entry, shared by Node and the browser.
import { getVoice } from "./speech/core/voice";
import { createApi } from "./speech/core/api";
import {
    Language,
    ModelId,
    ProgressPhase,
    Quality,
    ThreadCount,
    TimestampMode,
} from "./tensor/core/types";
const {
    transcribe,
    preloadVoice,
    loadModel,
    isModelCached,
} = createApi(getVoice());

export {
    transcribe,
    preloadVoice,
    loadModel,
    isModelCached,
    Language,
    ModelId,
    ProgressPhase,
    Quality,
    ThreadCount,
    TimestampMode,
};
export type {
    DiskBudgetOptions,
    DownloadProgress,
    ModelOptionsBase,
    PreloadVoiceOptions,
    ProgressHandler,
    Segment,
    TextSnapshot,
    TranscribeBase,
    TranscribeOptions,
    TranscribeSession,
    Transcript,
    TranscriptMeta,
    Word,
} from "./tensor/core/types";
