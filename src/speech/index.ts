// Text to speech voice library.
export { VoiceFamily, VoiceGender, VoiceId } from "./types";
export { listVoices, getVoiceSpec, PREVIEW, PREVIEW_RATE } from "./voices";
export { getVoiceSample } from "./sample";
export { loadVoice, isVoiceCached, removeVoice } from "./download";
export { prepareVoice, synthesize } from "./engine";
export { cloneVoice } from "./cloning";
export type {
    CloneVoiceOptions,
    LoadVoiceOptions,
    SynthesizeVoiceOptions,
    VoiceAudio,
    VoiceClone,
    VoiceCloneProgress,
    VoiceDownloadProgress,
    VoiceFile,
    VoiceSample,
    VoiceSpec,
} from "./types";
