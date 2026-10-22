import type { AttentionOptions, Conv1dOptions, Conv2dOptions, LinearOptions, PointwiseOptions, RelativeBiasOptions, SiluOptions, WasmKernel } from "./types";
declare const 
/** Request the worker count used by the next pool. */
setThreadCount: (threads?: number) => void, 
/** Worker threads used by the current decode. */
kernelThreads: () => number, 
/** Create the shared kernel and worker pool before a decode. */
prepareThreads: (threads?: number) => Promise<void>, 
/** Shared kernel instance, null when the engine lacks SIMD. */
getKernel: () => WasmKernel | null, 
/** Reserve an f32 arena region the weights are decoded into. */
reserveFloats: (floats: number) => Float32Array | null, 
/** Run a linear layer with the SIMD kernel, or null to fall back. */
wasmLinear: (options: LinearOptions) => Float32Array | null, 
/** Run multi-head attention with the SIMD kernel, or null to fall back. */
wasmAttention: (options: AttentionOptions) => Float32Array | null, 
/** Run a 1d convolution with the SIMD kernel, or null to fall back. */
wasmConv1d: (options: Conv1dOptions) => {
    data: Float32Array;
    length: number;
} | null, 
/** Run a 2d convolution with the SIMD kernel, or null to fall back. */
wasmConv2d: (options: Conv2dOptions) => {
    data: Float32Array;
    height: number;
    width: number;
} | null, 
/** Apply SiLU in place with the SIMD kernel. */
wasmSilu: (options: SiluOptions) => boolean, 
/** Run a 1x1 stride one convolution with the SIMD kernel. */
wasmPointwise: (options: PointwiseOptions) => Float32Array | null, 
/** Build the relative position bias with the SIMD kernel. */
wasmRelativeBias: (options: RelativeBiasOptions) => Float32Array | null;
export { getKernel, kernelThreads, prepareThreads, reserveFloats, setThreadCount, wasmAttention, wasmConv1d, wasmConv2d, wasmLinear, wasmPointwise, wasmRelativeBias, wasmSilu, };
