import type { AttentionOptions, LayerNormOptions, LinearOptions } from "./types";
declare const 
/** Apply GELU in place. */
geluInPlace: (options: {
    values: Float32Array;
    length: number;
}) => void, 
/** Rectified linear unit in place. */
reluInPlace: (options: {
    values: Float32Array;
    offset?: number;
    length?: number;
}) => void, 
/** SiLU activation in place. */
siluInPlace: (options: {
    values: Float32Array;
    offset?: number;
    length?: number;
}) => void, 
/** Elementwise sum returning a new array. */
addArrays: (options: {
    a: Float32Array;
    b: Float32Array;
    scaleB?: number;
}) => Float32Array, 
/** Elementwise add into the first array. */
addInPlace: (options: {
    a: Float32Array;
    b: Float32Array;
    scaleB?: number;
}) => void, 
/** Layer normalization over the last dimension. */
layerNorm: (options: LayerNormOptions) => Float32Array, 
/** In place softmax over one contiguous row. */
softmaxRow: (options: {
    values: Float32Array;
    offset: number;
    length: number;
}) => void, 
/** Linear layer, SIMD fast path with a pure JavaScript fallback. */
linear: (options: LinearOptions) => Float32Array, 
/** Multi-head attention, SIMD fast path with a fallback. */
attention: (options: AttentionOptions) => Float32Array, 
/** Single query cross attention returning head averaged weights. */
crossAttentionOne: (options: {
    q: Float32Array;
    k: Float32Array;
    v: Float32Array;
    heads: number;
    headDim: number;
    seqK: number;
    captureHeads?: number[];
    weightsOut?: Float32Array;
}) => Float32Array, 
/** Argument max over a range. */
argmax: (options: {
    values: Float32Array;
    start?: number;
    end?: number;
}) => number;
export { geluInPlace, reluInPlace, siluInPlace, addArrays, addInPlace, linear, layerNorm, softmaxRow, attention, crossAttentionOne, argmax, };
