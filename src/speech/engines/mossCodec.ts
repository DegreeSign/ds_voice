// MOSS audio tokenizer decoder for MOSS-TTS-Nano.
import { parseGguf } from "../../models/formats/gguf";
import { readGgufTensor } from "../../models/formats/ggufTensor";
import {
    geluErfInPlace,
    layerNorm,
    linear,
} from "./ops";
import type { GgufFile } from "../../tensor/core/types";
const
    /** Decoder stage shape of the audio tokenizer. */
    HEADS = 4,
    HEAD_DIM = 64,
    MODEL_DIM = 256,
    INTER_DIM = 1024,
    ROPE_BASE = 10000,
    NORM_EPS = 1e-5,
    /** Codec channel layout. */
    CODE_DIM = 768,
    RVQ_DIM = 512,
    CODEBOOK_DIM = 8,
    CODEBOOK_SIZE = 1024,
    N_QUANTIZERS = 16,
    CHANNELS = 2,
    /** Decoder module order. */
    DECODER: Array<{ patch?: number; stage?: number }> = [
        { patch: 4 },
        { stage: 1 },
        { patch: 2 },
        { stage: 3 },
        { patch: 2 },
        { stage: 5 },
        { patch: 2 },
        { stage: 7 },
        { patch: 240 },
    ],
    /** Sliding context of every decoder stage, in frames. */
    CONTEXT: Record<number, number> = {
        1: 500,
        3: 800,
        5: 1200,
        7: 1600,
    };

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

/** Read one tensor from a parsed file. */
const tensor = (options: {
    file: GgufFile;
    name: string;
}): Float32Array => {
    const info = options.file.tensors.get(options.name);
    if (!info) {
        throw new Error(`missing tensor ${options.name}`);
    }
    return readGgufTensor({ file: options.file, info });
};

/** RoPE frequencies of one head. */
const ropeFrequencies = (): Float32Array => {
    const out = new Float32Array(HEAD_DIM / 2);
    for (let i = 0; i < out.length; i += 1) {
        out[i] = Math.exp((-Math.log(ROPE_BASE) * 2 * i) / HEAD_DIM);
    }
    return out;
};

/** Apply RoPE to one head of a query or key row. */
const rotate = (options: {
    values: Float32Array;
    rowOffset: number;
    headOffset: number;
    position: number;
    frequencies: Float32Array;
}): void => {
    const { values, rowOffset, headOffset, position, frequencies } = options;
    for (let i = 0; i < HEAD_DIM / 2; i += 1) {
        const
            base = rowOffset + headOffset + i * 2,
            angle = position * frequencies[i],
            cos = Math.cos(angle),
            sin = Math.sin(angle),
            real = values[base],
            imag = values[base + 1];
        values[base] = real * cos - imag * sin;
        values[base + 1] = real * sin + imag * cos;
    }
};

/** Causal attention with an optional sliding window. */
const attend = (options: {
    input: Float32Array;
    frames: number;
    qkv: Float32Array;
    out: Float32Array;
    context: number;
}): Float32Array => {
    const
        { input, frames, qkv, out, context } = options,
        dim = MODEL_DIM,
        projected = linear({
            input,
            rows: frames,
            inDim: dim,
            weight: qkv,
            outDim: 3 * dim,
        }),
        frequencies = ropeFrequencies(),
        result = new Float32Array(frames * dim),
        scores = new Float32Array(frames);
    for (let frame = 0; frame < frames; frame += 1) {
        const base = frame * 3 * dim;
        for (let head = 0; head < HEADS; head += 1) {
            const offset = head * HEAD_DIM;
            rotate({
                values: projected,
                rowOffset: base,
                headOffset: offset,
                position: frame,
                frequencies,
            });
            rotate({
                values: projected,
                rowOffset: base + dim,
                headOffset: offset,
                position: frame,
                frequencies,
            });
        }
    }
    for (let frame = 0; frame < frames; frame += 1) {
        const
            queryBase = frame * 3 * dim,
            first = Math.max(0, frame - context + 1);
        for (let head = 0; head < HEADS; head += 1) {
            const
                queryOffset = queryBase + head * HEAD_DIM,
                scale = 1 / Math.sqrt(HEAD_DIM);
            let max = -Infinity;
            for (let key = first; key <= frame; key += 1) {
                const keyOffset = key * 3 * dim + dim + head * HEAD_DIM;
                let sum = 0;
                for (let d = 0; d < HEAD_DIM; d += 1) {
                    sum += projected[queryOffset + d] *
                        projected[keyOffset + d];
                }
                const value = sum * scale;
                scores[key] = value;
                if (value > max) {
                    max = value;
                }
            }
            let total = 0;
            for (let key = first; key <= frame; key += 1) {
                const value = Math.exp(scores[key] - max);
                scores[key] = value;
                total += value;
            }
            for (let d = 0; d < HEAD_DIM; d += 1) {
                let sum = 0;
                for (let key = first; key <= frame; key += 1) {
                    const valueOffset = key * 3 * dim + 2 * dim +
                        head * HEAD_DIM;
                    sum += scores[key] / total * projected[valueOffset + d];
                }
                result[frame * dim + head * HEAD_DIM + d] = sum;
            }
        }
    }
    return linear({
        input: result,
        rows: frames,
        inDim: dim,
        weight: out,
        outDim: dim,
    });
};

/** Run one decoder transformer stage. */
const runStage = (options: {
    stage: MossCodecStage;
    input: Float32Array;
    frames: number;
}): Float32Array => {
    const
        { stage: current, input, frames } = options,
        dim = MODEL_DIM,
        x = linear({
            input,
            rows: frames,
            inDim: current.inputDim,
            weight: current.input,
            outDim: dim,
        }),
        normed = new Float32Array(frames * dim);
    for (const layer of current.layers) {
        layerNorm({
            input: x,
            frames,
            channels: dim,
            weight: layer.norm1Weight,
            bias: layer.norm1Bias,
            eps: NORM_EPS,
            out: normed,
        });
        const attention = attend({
            input: normed,
            frames,
            qkv: layer.qkv,
            out: layer.out,
            context: current.context,
        });
        for (let i = 0; i < x.length; i += 1) {
            x[i] += attention[i] * layer.scale1[i % dim];
        }
        layerNorm({
            input: x,
            frames,
            channels: dim,
            weight: layer.norm2Weight,
            bias: layer.norm2Bias,
            eps: NORM_EPS,
            out: normed,
        });
        const up = linear({
            input: normed,
            rows: frames,
            inDim: dim,
            weight: layer.up,
            outDim: INTER_DIM,
        });
        geluErfInPlace(up);
        const down = linear({
            input: up,
            rows: frames,
            inDim: INTER_DIM,
            weight: layer.down,
            outDim: dim,
        });
        for (let i = 0; i < down.length; i += 1) {
            x[i] += down[i] * layer.scale2[i % dim];
        }
    }
    return linear({
        input: x,
        rows: frames,
        inDim: dim,
        weight: current.output,
        outDim: current.outputDim,
    });
};

/** Expand one patched pretransform frame into its samples. */
const expand = (options: {
    input: Float32Array;
    frames: number;
    channels: number;
    size: number;
}): Float32Array => {
    const
        { input, frames, channels, size } = options,
        outChannels = channels / size,
        out = new Float32Array(frames * size * outChannels);
    for (let frame = 0; frame < frames; frame += 1) {
        for (let step = 0; step < size; step += 1) {
            const target = (frame * size + step) * outChannels;
            for (let channel = 0; channel < outChannels; channel += 1) {
                out[target + channel] =
                    input[frame * channels + channel * size + step];
            }
        }
    }
    return out;
};

/** Load one decoder transformer stage. */
const loadStage = (options: {
    file: GgufFile;
    index: number;
    inputDim: number;
    outputDim: number;
    layers: number;
}): MossCodecStage => {
    const
        { file, index, inputDim, outputDim, layers } = options,
        prefix = `codec.decoder.${index}`,
        read = (name: string): Float32Array => tensor({ file, name }),
        list: MossCodecLayer[] = [];
    for (let layer = 0; layer < layers; layer += 1) {
        const base = `${prefix}.transformer.layers.${layer}`;
        list.push({
            norm1Weight: read(`${base}.norm1.weight`),
            norm1Bias: read(`${base}.norm1.bias`),
            norm2Weight: read(`${base}.norm2.weight`),
            norm2Bias: read(`${base}.norm2.bias`),
            scale1: read(`${base}.layer_scale_1.scale`),
            scale2: read(`${base}.layer_scale_2.scale`),
            qkv: read(`${base}.self_attn.in_proj.weight`),
            out: read(`${base}.self_attn.out_proj.weight`),
            up: read(`${base}.ffn.0.weight`),
            down: read(`${base}.ffn.2.weight`),
        });
    }
    return {
        inputDim,
        outputDim,
        input: read(`${prefix}.input_proj.weight`),
        output: read(`${prefix}.output_proj.weight`),
        layers: list,
        context: CONTEXT[index] ?? 0,
    };
};

/** Parse the codec GGUF into a decoder. */
const loadMossCodec = (bytes: Uint8Array): MossCodec => {
    const
        file = parseGguf({ bytes }),
        read = (name: string): Float32Array => tensor({ file, name }),
        stages = new Map<number, MossCodecStage>([
            [1, loadStage({
                file,
                index: 1,
                inputDim: 192,
                outputDim: 768,
                layers: 4,
            })],
            [3, loadStage({
                file,
                index: 3,
                inputDim: 384,
                outputDim: 768,
                layers: 2,
            })],
            [5, loadStage({
                file,
                index: 5,
                inputDim: 384,
                outputDim: 768,
                layers: 2,
            })],
            [7, loadStage({
                file,
                index: 7,
                inputDim: 384,
                outputDim: 240,
                layers: 4,
            })],
        ]),
        quantizers: MossCodecQuantizer[] = [];
    for (let index = 0; index < N_QUANTIZERS; index += 1) {
        const prefix = `codec.quantizer.quantizers.${index}`;
        quantizers.push({
            codebook: read(`${prefix}.codebook.weight`),
            outWeight: read(`${prefix}.out_proj.weight`),
            outBias: read(`${prefix}.out_proj.bias`),
        });
    }
    return {
        outputWeight: read("codec.quantizer.output_proj.weight"),
        outputBias: read("codec.quantizer.output_proj.bias"),
        quantizers,
        stages,
    };
};

/** Sum the residual codebook vectors of one frame run. */
const dequantize = (options: {
    codec: MossCodec;
    codes: Uint16Array;
    frames: number;
}): Float32Array => {
    const
        { codec, codes, frames } = options,
        latent = new Float32Array(frames * CODEBOOK_DIM),
        emb = new Float32Array(frames * RVQ_DIM);
    for (let index = 0; index < codec.quantizers.length; index += 1) {
        const quantizer = codec.quantizers[index];
        for (let frame = 0; frame < frames; frame += 1) {
            const
                code = codes[frame * N_QUANTIZERS + index],
                source = (code % CODEBOOK_SIZE) * CODEBOOK_DIM;
            latent.set(
                quantizer.codebook.subarray(source, source + CODEBOOK_DIM),
                frame * CODEBOOK_DIM,
            );
        }
        const part = linear({
            input: latent,
            rows: frames,
            inDim: CODEBOOK_DIM,
            weight: quantizer.outWeight,
            outDim: RVQ_DIM,
            bias: quantizer.outBias,
        });
        for (let i = 0; i < part.length; i += 1) {
            emb[i] += part[i];
        }
    }
    return linear({
        input: emb,
        rows: frames,
        inDim: RVQ_DIM,
        weight: codec.outputWeight,
        outDim: CODE_DIM,
        bias: codec.outputBias,
    });
};

/** Decode one run of audio code frames into mono PCM. */
const decodeMossAudio = (options: {
    codec: MossCodec;
    codes: Uint16Array;
    frames: number;
}): Float32Array => {
    const { codec, codes } = options;
    if (options.frames === 0) {
        return new Float32Array(0);
    }
    let
        values = dequantize({ codec, codes, frames: options.frames }),
        frames = options.frames,
        channels = CODE_DIM;
    for (const module of DECODER) {
        if (module.patch) {
            values = expand({
                input: values,
                frames,
                channels,
                size: module.patch,
            });
            frames *= module.patch;
            channels /= module.patch;
            continue;
        }
        const current = codec.stages.get(module.stage ?? 0);
        if (!current) {
            throw new Error(`missing codec stage ${module.stage}`);
        }
        values = runStage({
            stage: current,
            input: values,
            frames,
        });
        channels = current.outputDim;
    }
    const pcm = new Float32Array(frames / CHANNELS);
    for (let i = 0; i < pcm.length; i += 1) {
        pcm[i] = (values[i * CHANNELS] + values[i * CHANNELS + 1]) / 2;
    }
    return pcm;
};

export { decodeMossAudio, loadMossCodec };
export type { MossCodec };
