import type { ReadGgufTensorOptions } from "../tensor/types";
declare const 
/** Default F32 cache budget in bytes. */
GGUF_CACHE_BYTES: number, 
/** ggml type code for f32. */
GGML_F32 = 0, 
/** ggml type code for f16. */
GGML_F16 = 1, 
/** ggml type code for Q8_0. */
GGML_Q8_0 = 8, 
/** Read one GGUF tensor as f32, row major. */
readGgufTensor: (options: ReadGgufTensorOptions) => Float32Array;
/** Human readable name of a supported ggml type. */
declare const ggmlTypeName: (type: number) => string;
export { GGUF_CACHE_BYTES, GGML_F32, GGML_F16, GGML_Q8_0, readGgufTensor, ggmlTypeName, };
