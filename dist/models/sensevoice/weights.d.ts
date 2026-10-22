import { GGUF_CACHE_BYTES } from "../../formats/ggufTensor";
import type { CreateGgufWeightsOptions, SenseVoiceWeights } from "../../tensor/types";
declare const 
/** Build a weight store over a parsed GGUF file. */
createGgufWeights: (options: CreateGgufWeightsOptions) => SenseVoiceWeights;
export { GGUF_CACHE_BYTES, createGgufWeights };
