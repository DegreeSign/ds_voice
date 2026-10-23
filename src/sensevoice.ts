// SenseVoice only browser entry.
import { getVoice } from "./speech/senseVoice";
import { createApi } from "./speech/api";
import {
    Language,
    ModelId,
    ProgressPhase,
    Quality,
    ThreadCount,
    TimestampMode,
} from "./tensor/types";
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
} from "./tensor/types";
