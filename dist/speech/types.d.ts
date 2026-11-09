import { ProgressPhase } from "../tensor/core/types";
/** Voice choices in the bundled library. */
export declare enum VoiceId {
    /** Male 100MB, MOSS-TTS-Nano-100M, multilingual. */
    Male = "male",
    /** Female 100MB, MOSS-TTS-Nano-100M, multilingual. */
    Female = "female"
}
/** Gender label of a library voice. */
export declare enum VoiceGender {
    Female = "female",
    Male = "male"
}
/** Model family behind a library voice. */
export declare enum VoiceFamily {
    Moss = "moss"
}
/** One file of a voice model, downloaded on first use. */
export interface VoiceFile {
    /** Hugging Face repository that holds the file. */
    repo: string;
    /** Path of the file inside the repository. */
    file: string;
    /** Expected size in bytes, zero when unknown. */
    bytes: number;
}
/** One voice in the library. */
export interface VoiceSpec {
    /** Stable voice identifier. */
    id: VoiceId;
    /** Human readable name. */
    label: string;
    /** Gender label. */
    gender: VoiceGender;
    /** Model family. */
    family: VoiceFamily;
    /** Speaker or preset name inside the model. */
    speaker: string;
    /** Native sample rate of the model output. */
    sampleRate: number;
    /** Sample rate of the bundled preview clip. */
    previewRate: number;
    /** Weight license. */
    license: string;
    /** Primary model repository. */
    repository: string;
    /** Every file needed to synthesize. */
    files: VoiceFile[];
    /** Fixed line rendered for the preview clip. */
    preview: string;
}
/** Decoded preview clip of one voice. */
export interface VoiceSample {
    /** Voice the clip belongs to. */
    voice: VoiceId;
    /** WAV bytes. */
    bytes: Uint8Array;
    /** Sample rate of the clip. */
    sampleRate: number;
    /** Clip MIME type. */
    mimeType: "audio/wav";
}
/** Progress of a lazy voice model download. */
export interface VoiceDownloadProgress {
    /** Voice being downloaded. */
    voice: VoiceId;
    /** File currently in flight. */
    file: string;
    /** Bytes received across every file. */
    loaded: number;
    /** Expected bytes across every file. */
    total: number;
    /** Fraction of the total, zero when unknown. */
    ratio: number;
    /** Step of the load. */
    phase: ProgressPhase;
}
/** Options for loading one voice model. */
export interface LoadVoiceOptions {
    /** Voice to load. */
    voice: VoiceId;
    /** Override for the Hugging Face host. */
    weightsBaseUrl?: string;
    /** Called as the download advances. */
    onProgress?: (progress: VoiceDownloadProgress) => void;
}
/** Options for synthesizing text with one voice. */
export interface SynthesizeVoiceOptions extends LoadVoiceOptions {
    /** Text to speak. */
    text: string;
    /** Pitch shift in semitones, zero keeps the source voice. */
    pitch?: number;
    /** Speaking rate multiplier, one keeps the source timing. */
    rate?: number;
    /** Linear output gain, MOSS defaults to two. */
    gain?: number;
    /** Deterministic sampling seed, MOSS only. */
    seed?: number;
    /** Sampling temperature, MOSS only. */
    temperature?: number;
}
/** Mono PCM produced by a voice. */
export interface VoiceAudio {
    /** Samples in the range -1 to 1. */
    pcm: Float32Array;
    /** Sample rate of the samples. */
    sampleRate: number;
}
