// Package entry, shared by Node and the browser.
import { getVoice } from "./transcription/core/voice";
import { createApi } from "./transcription/core/api";
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
export {
    listVoices,
    getVoiceSpec,
    getVoiceSample,
    loadVoice,
    prepareVoice,
    synthesize,
    isVoiceCached,
    removeVoice,
    PREVIEW,
    PREVIEW_RATE,
    VoiceFamily,
    VoiceGender,
    VoiceId,
} from "./speech";
export type {
    LoadVoiceOptions,
    SynthesizeVoiceOptions,
    VoiceAudio,
    VoiceDownloadProgress,
    VoiceFile,
    VoiceSample,
    VoiceSpec,
} from "./speech";
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
