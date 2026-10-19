import type { SenseVoiceEncodeOptions, SenseVoiceEncodeResult, SenseVoiceLogitsOptions } from "../../tensor/types";
declare const 
/** Encode prefixes plus features into hidden states. */
encodeSenseVoice: (options: SenseVoiceEncodeOptions) => SenseVoiceEncodeResult, 
/** Run the CTC head over hidden states. */
senseVoiceLogits: (options: SenseVoiceLogitsOptions) => Float32Array;
export { encodeSenseVoice, senseVoiceLogits };
