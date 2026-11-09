declare const 
/** Matrix multiply with an optional bias row. */
linear: (options: {
    input: Float32Array;
    rows: number;
    inDim: number;
    weight: Float32Array;
    outDim: number;
    bias?: Float32Array;
    out?: Float32Array;
}) => Float32Array, 
/** Layer normalization over the channel axis of every frame. */
layerNorm: (options: {
    input: Float32Array;
    frames: number;
    channels: number;
    weight?: Float32Array;
    bias?: Float32Array;
    eps?: number;
    out?: Float32Array;
}) => Float32Array, 
/** GELU with the tanh approximation, in place. */
geluInPlace: (values: Float32Array) => void, 
/** GELU with the exact error function, in place. */
geluErfInPlace: (values: Float32Array) => void;
export { geluErfInPlace, geluInPlace, layerNorm, linear, };
