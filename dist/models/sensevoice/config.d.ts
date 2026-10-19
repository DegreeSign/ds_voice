import type { GgufFile, SenseVoiceConfig, SenseVoiceFrontend, SenseVoiceWeights } from "../../tensor/types";
declare const 
/** Build the normalized encoder shape. */
readSenseVoiceConfig: (options: {
    file: GgufFile;
}) => SenseVoiceConfig, 
/** Build the kaldi fbank frontend settings. */
readSenseVoiceFrontend: (options: {
    file: GgufFile;
    weights: SenseVoiceWeights;
}) => SenseVoiceFrontend;
export { readSenseVoiceConfig, readSenseVoiceFrontend };
