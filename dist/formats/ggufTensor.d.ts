import type { ReadGgufTensorOptions } from "../tensor/types";
declare const 
/** Default F32 cache budget in bytes. */
GGUF_CACHE_BYTES: number, 
/** Largest automatic cache budget in bytes. */
GGUF_CACHE_CAP_BYTES: number, 
/** Budget that holds every tensor of a checkpoint once. */
fullCacheBytes: (total: number) => number, 
/** ggml type code for f32. */
GGML_F32 = 0, 
/** ggml type code for f16. */
GGML_F16 = 1, 
/** ggml type code for Q8_0. */
GGML_Q8_0 = 8, 
/** ggml type code for Q4_K. */
GGML_Q4_K = 12, 
/** ggml type code for Q6_K. */
GGML_Q6_K = 14, 
/** Read one GGUF tensor as f32, row major. */
readGgufTensor: (options: ReadGgufTensorOptions) => Float32Array;
/** Human readable name of a supported ggml type. */
declare const ggmlTypeName: (type: number) => string;
export { GGUF_CACHE_BYTES, GGUF_CACHE_CAP_BYTES, fullCacheBytes, GGML_F32, GGML_F16, GGML_Q8_0, GGML_Q4_K, GGML_Q6_K, readGgufTensor, ggmlTypeName, };
