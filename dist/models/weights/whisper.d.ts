import type { CreateGgufWeightsOptions, TensorStore } from "../../tensor/core/types";
declare const 
/** Map one ggml tensor name to its Hugging Face name. */
mapName: (name: string) => string | null, 
/** Build a tensor store over a Whisper GGUF file. */
createWhisperWeights: (options: CreateGgufWeightsOptions) => TensorStore;
export { createWhisperWeights, mapName };
