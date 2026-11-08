import { Language } from "../../tensor/core/types";
import type { LoadedSenseVoice, LoadSenseVoiceOptions, Transcript } from "../../tensor/core/types";
declare const 
/** Load a SenseVoice model from a store. */
loadSenseVoice: (options: LoadSenseVoiceOptions) => Promise<LoadedSenseVoice>, 
/** Transcribe one buffer with a SenseVoice model. */
transcribeSenseVoice: (options: {
    model: LoadedSenseVoice;
    pcm: Float32Array;
    language?: Language;
    itn?: boolean;
}) => Transcript;
export { loadSenseVoice, transcribeSenseVoice };
