import type { GgufFile, SenseVoiceWeights } from "../../tensor/types";
declare const 
/** Build a weight store over a parsed GGUF file. */
createGgufWeights: (options: {
    file: GgufFile;
}) => SenseVoiceWeights;
export { createGgufWeights };
