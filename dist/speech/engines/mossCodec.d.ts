/** One layer of an audio decoder transformer. */
interface MossCodecLayer {
    norm1Weight: Float32Array;
    norm1Bias: Float32Array;
    norm2Weight: Float32Array;
    norm2Bias: Float32Array;
    scale1: Float32Array;
    scale2: Float32Array;
    qkv: Float32Array;
    out: Float32Array;
    up: Float32Array;
    down: Float32Array;
}
/** One projected transformer of the audio decoder. */
interface MossCodecStage {
    inputDim: number;
    outputDim: number;
    input: Float32Array;
    output: Float32Array;
    layers: MossCodecLayer[];
    context: number;
}
/** One residual LFQ quantizer. */
interface MossCodecQuantizer {
    codebook: Float32Array;
    outWeight: Float32Array;
    outBias: Float32Array;
}
/** Loaded audio tokenizer decoder. */
interface MossCodec {
    outputWeight: Float32Array;
    outputBias: Float32Array;
    quantizers: MossCodecQuantizer[];
    stages: Map<number, MossCodecStage>;
}
/** Parse the codec GGUF into a decoder. */
declare const loadMossCodec: (bytes: Uint8Array) => MossCodec;
/** Decode one run of audio code frames into mono PCM. */
declare const decodeMossAudio: (options: {
    codec: MossCodec;
    codes: Uint16Array;
    frames: number;
}) => Float32Array;
export { decodeMossAudio, loadMossCodec };
export type { MossCodec };
