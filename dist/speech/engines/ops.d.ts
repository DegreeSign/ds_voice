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
/** One dimensional convolution, frame major, stride and dilation. */
conv1d: (options: {
    input: Float32Array;
    frames: number;
    channels: number;
    weight: Float32Array;
    bias?: Float32Array;
    outChannels: number;
    kernel: number;
    stride?: number;
    padding?: number;
    dilation?: number;
}) => Float32Array, 
/** Transposed convolution built from a matmul and a frame scatter. */
convTranspose1d: (options: {
    input: Float32Array;
    frames: number;
    channels: number;
    weight: Float32Array;
    bias?: Float32Array;
    outChannels: number;
    kernel: number;
    stride: number;
    padding: number;
    outputPadding?: number;
}) => Float32Array, 
/** Depthwise transposed convolution over the channel axis. */
depthwiseConvTranspose1d: (options: {
    input: Float32Array;
    frames: number;
    channels: number;
    weight: Float32Array;
    bias?: Float32Array;
    kernel: number;
    stride: number;
    padding: number;
    outputPadding?: number;
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
/** Instance normalization over the frame axis of every channel. */
instanceNorm: (options: {
    input: Float32Array;
    frames: number;
    channels: number;
    eps?: number;
}) => Float32Array, 
/** Apply an adaptive instance normalization in place. */
adainInPlace: (options: {
    input: Float32Array;
    frames: number;
    channels: number;
    style: Float32Array;
    weight: Float32Array;
    bias: Float32Array;
}) => void, 
/** Apply an adaptive layer normalization in place. */
adalnInPlace: (options: {
    input: Float32Array;
    frames: number;
    channels: number;
    style: Float32Array;
    weight: Float32Array;
    bias: Float32Array;
}) => void, 
/** Leaky rectified linear unit in place. */
leakyInPlace: (options: {
    values: Float32Array;
    slope: number;
}) => void, 
/** Snake activation with one alpha per channel, in place. */
snakeInPlace: (options: {
    values: Float32Array;
    frames: number;
    channels: number;
    alpha: Float32Array;
}) => void, 
/** Sigmoid of one value. */
sigmoid: (value: number) => number, 
/** SiLU activation in place. */
siluInPlace: (values: Float32Array) => void, 
/** GELU with the tanh approximation, in place. */
geluInPlace: (values: Float32Array) => void, 
/** Repeat every frame twice, nearest neighbour. */
upsample2: (options: {
    input: Float32Array;
    frames: number;
    channels: number;
}) => Float32Array, 
/** Reflection pad one frame on the left. */
reflectPadLeft: (options: {
    input: Float32Array;
    frames: number;
    channels: number;
}) => Float32Array, 
/** Concatenate channel blocks frame by frame. */
concatChannels: (options: {
    parts: Array<{
        data: Float32Array;
        channels: number;
    }>;
    frames: number;
}) => Float32Array, 
/** Repeat feature rows by phoneme durations. */
alignRepeat: (options: {
    input: Float32Array;
    rows: number;
    channels: number;
    durations: number[];
}) => Float32Array, 
/** Round half to even, matching torch.round. */
roundHalfEven: (value: number) => number;
export { adainInPlace, adalnInPlace, alignRepeat, concatChannels, conv1d, convTranspose1d, depthwiseConvTranspose1d, geluInPlace, instanceNorm, layerNorm, leakyInPlace, linear, reflectPadLeft, roundHalfEven, sigmoid, siluInPlace, snakeInPlace, upsample2, };
