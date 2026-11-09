declare const 
/** Largest reference kept for one clone. */
MAX_REFERENCE_SAMPLES: number, 
/** Smallest reference accepted for one clone. */
MIN_REFERENCE_SAMPLES = 48000;
/** One transformer block of the encoder. */
interface MossEncoderLayer {
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
/** One encoder transformer stage plus its projections. */
interface MossEncoderStage {
    inputDim: number;
    outputDim: number;
    context: number;
    input: Float32Array;
    output: Float32Array;
    layers: MossEncoderLayer[];
}
/** One residual vector quantizer of the encoder. */
interface MossEncoderQuantizer {
    inWeight: Float32Array;
    inBias: Float32Array;
    codebook: Float32Array;
    outWeight?: Float32Array;
    outBias?: Float32Array;
}
/** Every weight of the MOSS audio tokenizer encoder. */
interface MossEncoder {
    input: Float32Array;
    stages: MossEncoderStage[];
    quantizerInput: Float32Array;
    quantizerBias: Float32Array;
    quantizers: MossEncoderQuantizer[];
}
/** Load the MOSS audio tokenizer encoder from its data file. */
declare const loadMossEncoder: (bytes: Uint8Array) => MossEncoder;
/** Encode mono 48 kHz PCM into MOSS reference codes. */
declare const encodeMossAudio: (options: {
    encoder: MossEncoder;
    pcm: Float32Array;
}) => Uint16Array;
export { encodeMossAudio, loadMossEncoder, MAX_REFERENCE_SAMPLES, MIN_REFERENCE_SAMPLES, };
export type { MossEncoder, MossEncoderLayer, MossEncoderStage };
