// SenseVoice only browser entry.
import { getVoice } from "../bundle/senseVoice";
import { createApi } from "../core/api";
import {
    Language,
    ModelId,
    ProgressPhase,
    Quality,
    ThreadCount,
    TimestampMode,
} from "../../tensor/core/types";
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
} from "../../tensor/core/types";
