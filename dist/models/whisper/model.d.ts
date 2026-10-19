import { Language, TimestampMode } from "../../tensor/types";
import type { LoadedModel, StoreModelOptions, Transcript } from "../../tensor/types";
declare const 
/** Load a Whisper model from a store. */
loadWhisper: (options: StoreModelOptions) => Promise<LoadedModel>, 
/** Transcribe arbitrary length audio with a Whisper model. */
transcribeWhisper: (options: {
    model: LoadedModel;
    pcm: Float32Array;
    language?: Language;
    timestamps: TimestampMode;
    lowLatency?: boolean;
}) => Transcript;
export { loadWhisper, transcribeWhisper };
