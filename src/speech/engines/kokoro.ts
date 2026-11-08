// Kokoro 82M text to speech engine.
import { parseGguf } from "../../models/formats/gguf";
import { readGgufTensor } from "../../models/formats/ggufTensor";
import { prepareThreads } from "../../tensor/kernel/wasm";
import {
    adainInPlace,
    adalnInPlace,
    alignRepeat,
    concatChannels,
    conv1d,
    convTranspose1d,
    depthwiseConvTranspose1d,
    geluInPlace,
    layerNorm,
    leakyInPlace,
    linear,
    reflectPadLeft,
    roundHalfEven,
    sigmoid,
    snakeInPlace,
    upsample2,
} from "./ops";
import { phonemizeEnglish } from "./g2p";
import { getVoiceSpec } from "../voices";
import type { GgufFile, GgufValue } from "../../tensor/core/types";
import type { VoiceId, VoiceAudio } from "../types";
const
    /** Channels of the StyleTTS2 features. */
    HIDDEN = 512,
    /** Channels of the style vector halves. */
    STYLE = 128,
    /** Duration bins predicted for one phoneme. */
    MAX_DUR = 50,
    /** Native sample rate of the model. */
    SAMPLE_RATE = 24000,
    /** Shared Bert dimensions. */
    BERT_EMBED = 128,
    BERT_HIDDEN = 768,
    BERT_HEADS = 12,
    BERT_HEAD_DIM = BERT_HIDDEN / BERT_HEADS,
    BERT_LAYERS = 12,
    BERT_FF = 2048,
    BERT_MAX_POS = 512,
    BERT_EPS = 1e-12,
    /** Text encoder and decoder constants. */
    TEXT_LAYERS = 3,
    TEXT_EPS = 1e-5,
    DECODER_CHANNELS = 1024,
    DECODER_RESIDUAL = 64,
    BEFORE_GENERATOR = DECODER_CHANNELS + 2 + DECODER_RESIDUAL,
    GENERATOR_CHANNELS = 512,
    /** Short time Fourier transform constants. */
    N_FFT = 20,
    HOP = 5,
    BINS = N_FFT / 2 + 1,
    /** Upsampling applied to one predicted frame. */
    FRAME_RATE = 10 * 6 * HOP,
    /** Harmonic source constants. */
    HARMONICS = 8,
    SOURCE_DIM = HARMONICS + 1,
    SINE_AMP = 0.1,
    NOISE_STD = 0.003,
    VOICED_FLOOR = 10,
    TWO_PI = Math.PI * 2,
    /** Number of style rows in a voice pack. */
    MAX_PHONEMES = 510,
    /** Default seed of the harmonic noise. */
    NOISE_SEED = 0x12345,
    /** Mersenne twister noise source with normal draws. */
    createNoise = (seed: number): {
        uniform: () => number;
        normal: () => number;
    } => {
        const
            state = new Uint32Array(624),
            temper = (value: number): number => {
                let y = value;
                y ^= y >>> 11;
                y ^= (y << 7) & 0x9d2c5680;
                y ^= (y << 15) & 0xefc60000;
                y ^= y >>> 18;
                return y >>> 0;
            };
        state[0] = seed >>> 0;
        for (let i = 1; i < 624; i += 1) {
            state[i] = (Math.imul(
                1812433253,
                state[i - 1] ^ (state[i - 1] >>> 30),
            ) + i) >>> 0;
        }
        let index = 624;
        const next = (): number => {
            if (index >= 624) {
                for (let i = 0; i < 624; i += 1) {
                    const
                        y = (state[i] & 0x80000000) |
                            (state[(i + 1) % 624] & 0x7fffffff);
                    state[i] = state[(i + 397) % 624] ^ (y >>> 1) ^
                        (y & 1 ? 0x9908b0df : 0);
                }
                index = 0;
            }
            return temper(state[index++]);
        };
        let
            /** Second normal draw of a pair. */
            saved = 0,
            /** True while saved holds a value. */
            hasSaved = false;
        return {
            uniform: (): number => next() / 4294967296,
            normal: (): number => {
                if (hasSaved) {
                    hasSaved = false;
                    return saved;
                }
                let x = 0;
                let y = 0;
                let radius = 0;
                do {
                    x = 2 * (next() / 4294967296) - 1;
                    y = 2 * (next() / 4294967296) - 1;
                    radius = x * x + y * y;
                } while (radius > 1 || radius === 0);
                const multiplier = Math.sqrt(-2 * Math.log(radius) / radius);
                saved = x * multiplier;
                hasSaved = true;
                return y * multiplier;
            },
        };
    },
    /** Linear attention over all frames of one sequence. */
    attention = (options: {
        q: Float32Array;
        k: Float32Array;
        v: Float32Array;
        frames: number;
        dim: number;
        heads: number;
    }): Float32Array => {
        const
            { q, k, v, frames, dim, heads } = options,
            headDim = dim / heads,
            scale = 1 / Math.sqrt(headDim),
            out = new Float32Array(frames * dim),
            scores = new Float32Array(frames);
        for (let head = 0; head < heads; head += 1) {
            const base = head * headDim;
            for (let position = 0; position < frames; position += 1) {
                const query = position * dim + base;
                let max = -Infinity;
                for (let source = 0; source < frames; source += 1) {
                    const key = source * dim + base;
                    let sum = 0;
                    for (let d = 0; d < headDim; d += 1) {
                        sum += q[query + d] * k[key + d];
                    }
                    sum *= scale;
                    scores[source] = sum;
                    if (sum > max) {
                        max = sum;
                    }
                }
                let total = 0;
                for (let source = 0; source < frames; source += 1) {
                    const value = Math.exp(scores[source] - max);
                    scores[source] = value;
                    total += value;
                }
                for (let d = 0; d < headDim; d += 1) {
                    let sum = 0;
                    for (let source = 0; source < frames; source += 1) {
                        sum += scores[source] / total *
                            v[source * dim + base + d];
                    }
                    out[position * dim + base + d] = sum;
                }
            }
        }
        return out;
    },
    /** One direction of a long short term memory layer. */
    lstmDirection = (options: {
        input: Float32Array;
        frames: number;
        inDim: number;
        hidden: number;
        weightInput: Float32Array;
        weightHidden: Float32Array;
        biasInput: Float32Array;
        biasHidden: Float32Array;
        reverse: boolean;
    }): Float32Array => {
        const
            { input, frames, inDim, hidden } = options,
            { weightInput, weightHidden, biasInput, biasHidden } = options,
            gates = 4 * hidden,
            projected = linear({
                input,
                rows: frames,
                inDim,
                weight: weightInput,
                bias: biasInput,
                outDim: gates,
            }),
            out = new Float32Array(frames * hidden),
            cell = new Float32Array(hidden),
            state = new Float32Array(hidden),
            steps = new Float32Array(gates);
        for (let step = 0; step < frames; step += 1) {
            const frame = options.reverse ? frames - 1 - step : step;
            const base = frame * gates;
            for (let unit = 0; unit < gates; unit += 1) {
                let sum = projected[base + unit] + biasHidden[unit];
                const start = unit * hidden;
                for (let h = 0; h < hidden; h += 1) {
                    sum += state[h] * weightHidden[start + h];
                }
                steps[unit] = sum;
            }
            for (let h = 0; h < hidden; h += 1) {
                const inputGate = sigmoid(steps[h]);
                const forgetGate = sigmoid(steps[hidden + h]);
                const candidate = Math.tanh(steps[2 * hidden + h]);
                const outputGate = sigmoid(steps[3 * hidden + h]);
                const next = forgetGate * cell[h] + inputGate * candidate;
                cell[h] = next;
                state[h] = outputGate * Math.tanh(next);
                out[frame * hidden + h] = state[h];
            }
        }
        return out;
    },
    /** Bidirectional long short term memory over one sequence. */
    lstmBidir = (options: {
        input: Float32Array;
        frames: number;
        inDim: number;
        hidden: number;
        forward: LstmWeights;
        backward: LstmWeights;
    }): Float32Array => {
        const
            { input, frames, inDim, hidden } = options,
            forward = lstmDirection({
                input,
                frames,
                inDim,
                hidden,
                weightInput: options.forward.input,
                weightHidden: options.forward.hidden,
                biasInput: options.forward.inputBias,
                biasHidden: options.forward.hiddenBias,
                reverse: false,
            }),
            backward = lstmDirection({
                input,
                frames,
                inDim,
                hidden,
                weightInput: options.backward.input,
                weightHidden: options.backward.hidden,
                biasInput: options.backward.inputBias,
                biasHidden: options.backward.hiddenBias,
                reverse: true,
            }),
            out = new Float32Array(frames * hidden * 2);
        for (let frame = 0; frame < frames; frame += 1) {
            out.set(forward.subarray(frame * hidden, (frame + 1) * hidden),
                frame * hidden * 2);
            out.set(backward.subarray(frame * hidden, (frame + 1) * hidden),
                frame * hidden * 2 + hidden);
        }
        return out;
    },
    /** Bias free one by one convolution used as a projection. */
    project = (options: {
        input: Float32Array;
        frames: number;
        channels: number;
        weight: Float32Array;
        bias?: Float32Array;
    }): Float32Array => linear({
        input: options.input,
        rows: options.frames,
        inDim: options.channels,
        weight: options.weight,
        bias: options.bias,
        outDim: options.weight.length / options.channels,
    });

/** One direction of an LSTM parameter set. */
interface LstmWeights {
    input: Float32Array;
    hidden: Float32Array;
    inputBias: Float32Array;
    hiddenBias: Float32Array;
}

/** Adaptive residual block weights. */
interface ResBlockWeights {
    norm1: Float32Array;
    norm1Bias: Float32Array;
    norm2: Float32Array;
    norm2Bias: Float32Array;
    conv1: Float32Array;
    conv1Bias: Float32Array;
    conv2: Float32Array;
    conv2Bias: Float32Array;
    pool?: Float32Array;
    poolBias?: Float32Array;
    shortcut?: Float32Array;
}

/** Generator residual block weights. */
interface GeneratorBlockWeights {
    kernel: number;
    norm1: Float32Array[];
    norm1Bias: Float32Array[];
    norm2: Float32Array[];
    norm2Bias: Float32Array[];
    alpha1: Float32Array[];
    alpha2: Float32Array[];
    conv1: Float32Array[];
    conv1Bias: Float32Array[];
    conv2: Float32Array[];
    conv2Bias: Float32Array[];
}

/** Every tensor of one loaded Kokoro model. */
interface KokoroWeights {
    tokenEmbed: Float32Array;
    positionEmbed: Float32Array;
    typeEmbed: Float32Array;
    embedNorm: Float32Array;
    embedNormBias: Float32Array;
    embedProject: Float32Array;
    embedProjectBias: Float32Array;
    bert: {
        query: Float32Array;
        queryBias: Float32Array;
        key: Float32Array;
        keyBias: Float32Array;
        value: Float32Array;
        valueBias: Float32Array;
        output: Float32Array;
        outputBias: Float32Array;
        attentionNorm: Float32Array;
        attentionNormBias: Float32Array;
        feed: Float32Array;
        feedBias: Float32Array;
        down: Float32Array;
        downBias: Float32Array;
        feedNorm: Float32Array;
        feedNormBias: Float32Array;
    };
    bertProject: Float32Array;
    bertProjectBias: Float32Array;
    text: {
        embed: Float32Array;
        conv: Array<{
            weight: Float32Array;
            bias: Float32Array;
            norm: Float32Array;
            normBias: Float32Array;
        }>;
        forward: LstmWeights;
        backward: LstmWeights;
    };
    duration: {
        encoder: Array<{
            forward: LstmWeights;
            backward: LstmWeights;
            norm: Float32Array;
            normBias: Float32Array;
        }>;
        forward: LstmWeights;
        backward: LstmWeights;
        project: Float32Array;
        projectBias: Float32Array;
    };
    shared: {
        forward: LstmWeights;
        backward: LstmWeights;
    };
    pitch: Array<ResBlockWeights>;
    energy: Array<ResBlockWeights>;
    pitchProject: Float32Array;
    pitchProjectBias: Float32Array;
    energyProject: Float32Array;
    energyProjectBias: Float32Array;
    decoder: {
        pitchConv: Float32Array;
        pitchConvBias: Float32Array;
        energyConv: Float32Array;
        energyConvBias: Float32Array;
        residual: Float32Array;
        residualBias: Float32Array;
        encode: ResBlockWeights;
        decode: ResBlockWeights[];
        upsample: Array<{
            weight: Float32Array;
            bias: Float32Array;
        }>;
        noiseConv: Array<{
            weight: Float32Array;
            bias: Float32Array;
            stride: number;
            padding: number;
        }>;
        noiseResidual: GeneratorBlockWeights[];
        residualBlocks: GeneratorBlockWeights[];
        source: Float32Array;
        sourceBias: Float32Array;
        output: Float32Array;
        outputBias: Float32Array;
    };
}

/** One cached model plus its token table. */
interface KokoroModel {
    voice: VoiceId;
    weights: KokoroWeights;
    tokens: Map<string, number>;
    pack: Float32Array;
}

let
    /** Only one model stays resident. */
    resident: KokoroModel | undefined;

const
    /** Read one tensor of a parsed file. */
    tensor = (options: {
        file: GgufFile;
        name: string;
    }): Float32Array => {
        const info = options.file.tensors.get(options.name);
        if (!info) {
            throw new Error(`missing tensor ${options.name}`);
        }
        return readGgufTensor({ file: options.file, info });
    },
    /** Read the raw token strings of a model. */
    readTokens = (file: GgufFile): Map<string, number> => {
        const list = file.metadata.get("tokenizer.ggml.tokens") as
            GgufValue[] | undefined;
        if (!Array.isArray(list)) {
            throw new Error("kokoro model has no token table");
        }
        const map = new Map<string, number>();
        list.forEach((value, index) => {
            const token = String(value);
            if (token.length > 0 && !map.has(token)) {
                map.set(token, index);
            }
        });
        return map;
    },
    /** Longest matching token at one position. */
    tokenize = (options: {
        text: string;
        tokens: Map<string, number>;
    }): number[] => {
        const { text, tokens } = options;
        const ids: number[] = [];
        for (const symbol of text) {
            const id = tokens.get(symbol);
            if (id !== undefined) {
                ids.push(id);
            }
        }
        return ids;
    },
    /** Read one LSTM direction. */
    readLstm = (options: {
        file: GgufFile;
        prefix: string;
    }): LstmWeights => {
        const read = (name: string): Float32Array =>
            tensor({ file: options.file, name: `${options.prefix}.${name}` });
        return {
            input: read("weight_ih_l0"),
            hidden: read("weight_hh_l0"),
            inputBias: read("bias_ih_l0"),
            hiddenBias: read("bias_hh_l0"),
        };
    },
    /** Read the reverse direction of one LSTM. */
    readLstmReverse = (options: {
        file: GgufFile;
        prefix: string;
    }): LstmWeights => {
        const read = (name: string): Float32Array =>
            tensor({
                file: options.file,
                name: `${options.prefix}.${name}_reverse`,
            });
        return {
            input: read("weight_ih_l0"),
            hidden: read("weight_hh_l0"),
            inputBias: read("bias_ih_l0"),
            hiddenBias: read("bias_hh_l0"),
        };
    },
    /** Read one adaptive residual block. */
    readResBlock = (options: {
        file: GgufFile;
        prefix: string;
        pool: boolean;
        shortcut: boolean;
    }): ResBlockWeights => {
        const read = (name: string): Float32Array =>
            tensor({ file: options.file, name: `${options.prefix}.${name}` });
        const block: ResBlockWeights = {
            norm1: read("adain1.weight"),
            norm1Bias: read("adain1.bias"),
            norm2: read("adain2.weight"),
            norm2Bias: read("adain2.bias"),
            conv1: read("conv1.weight"),
            conv1Bias: read("conv1.bias"),
            conv2: read("conv2.weight"),
            conv2Bias: read("conv2.bias"),
        };
        if (options.shortcut) {
            block.shortcut = read("conv1x1.weight");
        }
        if (options.pool) {
            block.pool = read("pool.weight");
            block.poolBias = read("pool.bias");
        }
        return block;
    },
    /** Transpose a ConvTranspose1d kernel into matmul layout. */
    transposeTranspose = (options: {
        weight: Float32Array;
        channels: number;
        outChannels: number;
        kernel: number;
    }): Float32Array => {
        const
            { weight, channels, outChannels, kernel } = options,
            out = new Float32Array(weight.length);
        for (let input = 0; input < channels; input += 1) {
            for (let output = 0; output < outChannels; output += 1) {
                for (let k = 0; k < kernel; k += 1) {
                    out[(output * kernel + k) * channels + input] =
                        weight[(input * outChannels + output) * kernel + k];
                }
            }
        }
        return out;
    },
    /** Read one generator residual block. */
    readGeneratorBlock = (options: {
        file: GgufFile;
        prefix: string;
        kernel: number;
    }): GeneratorBlockWeights => {
        const read = (name: string): Float32Array =>
            tensor({ file: options.file, name: `${options.prefix}.${name}` });
        const block: GeneratorBlockWeights = {
            kernel: options.kernel,
            norm1: [],
            norm1Bias: [],
            norm2: [],
            norm2Bias: [],
            alpha1: [],
            alpha2: [],
            conv1: [],
            conv1Bias: [],
            conv2: [],
            conv2Bias: [],
        };
        for (let index = 0; index < 3; index += 1) {
            block.norm1.push(read(`adain1.${index}.weight`));
            block.norm1Bias.push(read(`adain1.${index}.bias`));
            block.norm2.push(read(`adain2.${index}.weight`));
            block.norm2Bias.push(read(`adain2.${index}.bias`));
            block.alpha1.push(read(`alpha1.${index}`));
            block.alpha2.push(read(`alpha2.${index}`));
            block.conv1.push(read(`convs1.${index}.weight`));
            block.conv1Bias.push(read(`convs1.${index}.bias`));
            block.conv2.push(read(`convs2.${index}.weight`));
            block.conv2Bias.push(read(`convs2.${index}.bias`));
        }
        return block;
    },
    /** Read every tensor the synthesis path needs. */
    readWeights = (file: GgufFile): KokoroWeights => {
        const read = (name: string): Float32Array =>
            tensor({ file, name });
        const
            /** Prefix helper for one nested block. */
            readBert = (name: string): Float32Array => read(`bert.${name}`),
            textConv: KokoroWeights["text"]["conv"] = [],
            durationEncoder: KokoroWeights["duration"]["encoder"] = [],
            pitch: ResBlockWeights[] = [],
            energy: ResBlockWeights[] = [];
        for (let index = 0; index < TEXT_LAYERS; index += 1) {
            textConv.push({
                weight: read(`text_enc.cnn.${index}.conv.weight`),
                bias: read(`text_enc.cnn.${index}.conv.bias`),
                norm: read(`text_enc.cnn.${index}.ln.gamma`),
                normBias: read(`text_enc.cnn.${index}.ln.beta`),
            });
            durationEncoder.push({
                forward: readLstm({
                    file,
                    prefix: `pred.dur_enc.${index}.lstm`,
                }),
                backward: readLstmReverse({
                    file,
                    prefix: `pred.dur_enc.${index}.lstm`,
                }),
                norm: read(`pred.dur_enc.${index}.adaln.weight`),
                normBias: read(`pred.dur_enc.${index}.adaln.bias`),
            });
        }
        for (let index = 0; index < 3; index += 1) {
            pitch.push(readResBlock({
                file,
                prefix: `pred.F0.${index}`,
                pool: index === 1,
                shortcut: index === 1,
            }));
            energy.push(readResBlock({
                file,
                prefix: `pred.N.${index}`,
                pool: index === 1,
                shortcut: index === 1,
            }));
        }
        const decode: ResBlockWeights[] = [];
        for (let index = 0; index < 4; index += 1) {
            decode.push(readResBlock({
                file,
                prefix: `dec.decode.${index}`,
                pool: index === 3,
                shortcut: true,
            }));
        }
        const residualBlocks: GeneratorBlockWeights[] = [];
        for (let index = 0; index < 6; index += 1) {
            residualBlocks.push(readGeneratorBlock({
                file,
                prefix: `dec.gen.resblocks.${index}`,
                kernel: [3, 7, 11][index % 3],
            }));
        }
        return {
            tokenEmbed: readBert("embd.tok.weight"),
            positionEmbed: readBert("embd.pos.weight"),
            typeEmbed: readBert("embd.tt.weight"),
            embedNorm: readBert("embd.ln.weight"),
            embedNormBias: readBert("embd.ln.bias"),
            embedProject: readBert("embd_proj.weight"),
            embedProjectBias: readBert("embd_proj.bias"),
            bert: {
                query: readBert("attn_q.weight"),
                queryBias: readBert("attn_q.bias"),
                key: readBert("attn_k.weight"),
                keyBias: readBert("attn_k.bias"),
                value: readBert("attn_v.weight"),
                valueBias: readBert("attn_v.bias"),
                output: readBert("attn_o.weight"),
                outputBias: readBert("attn_o.bias"),
                attentionNorm: readBert("attn_ln.weight"),
                attentionNormBias: readBert("attn_ln.bias"),
                feed: readBert("ffn_up.weight"),
                feedBias: readBert("ffn_up.bias"),
                down: readBert("ffn_down.weight"),
                downBias: readBert("ffn_down.bias"),
                feedNorm: readBert("ffn_ln.weight"),
                feedNormBias: readBert("ffn_ln.bias"),
            },
            bertProject: read("bert_proj.weight"),
            bertProjectBias: read("bert_proj.bias"),
            text: {
                embed: read("text_enc.embd.weight"),
                conv: textConv,
                forward: readLstm({ file, prefix: "text_enc.lstm" }),
                backward: readLstmReverse({ file, prefix: "text_enc.lstm" }),
            },
            duration: {
                encoder: durationEncoder,
                forward: readLstm({ file, prefix: "pred.lstm" }),
                backward: readLstmReverse({ file, prefix: "pred.lstm" }),
                project: read("pred.dur_proj.weight"),
                projectBias: read("pred.dur_proj.bias"),
            },
            shared: {
                forward: readLstm({ file, prefix: "pred.shared" }),
                backward: readLstmReverse({ file, prefix: "pred.shared" }),
            },
            pitch,
            energy,
            pitchProject: read("pred.F0_proj.weight"),
            pitchProjectBias: read("pred.F0_proj.bias"),
            energyProject: read("pred.N_proj.weight"),
            energyProjectBias: read("pred.N_proj.bias"),
            decoder: {
                pitchConv: read("dec.F0_conv.weight"),
                pitchConvBias: read("dec.F0_conv.bias"),
                energyConv: read("dec.N_conv.weight"),
                energyConvBias: read("dec.N_conv.bias"),
                residual: read("dec.asr_res.weight"),
                residualBias: read("dec.asr_res.bias"),
                encode: readResBlock({
                    file,
                    prefix: "dec.encode",
                    pool: false,
                    shortcut: true,
                }),
                decode,
                upsample: [
                    {
                        weight: transposeTranspose({
                            weight: read("dec.gen.ups.0.weight"),
                            channels: 512,
                            outChannels: 256,
                            kernel: 20,
                        }),
                        bias: read("dec.gen.ups.0.bias"),
                    },
                    {
                        weight: transposeTranspose({
                            weight: read("dec.gen.ups.1.weight"),
                            channels: 256,
                            outChannels: 128,
                            kernel: 12,
                        }),
                        bias: read("dec.gen.ups.1.bias"),
                    },
                ],
                noiseConv: [
                    {
                        weight: read("dec.gen.noise_convs.0.weight"),
                        bias: read("dec.gen.noise_convs.0.bias"),
                        stride: 6,
                        padding: 3,
                    },
                    {
                        weight: read("dec.gen.noise_convs.1.weight"),
                        bias: read("dec.gen.noise_convs.1.bias"),
                        stride: 1,
                        padding: 0,
                    },
                ],
                noiseResidual: [
                    readGeneratorBlock({
                        file,
                        prefix: "dec.gen.noise_res.0",
                        kernel: 7,
                    }),
                    readGeneratorBlock({
                        file,
                        prefix: "dec.gen.noise_res.1",
                        kernel: 11,
                    }),
                ],
                residualBlocks,
                source: read("dec.gen.m_source.weight"),
                sourceBias: read("dec.gen.m_source.bias"),
                output: read("dec.gen.conv_post.weight"),
                outputBias: read("dec.gen.conv_post.bias"),
            },
        };
    },
    /** Run the twelve shared Bert layers. */
    runBert = (options: {
        ids: number[];
        weights: KokoroWeights;
    }): Float32Array => {
        const
            { ids, weights } = options,
            frames = ids.length,
            bert = weights.bert,
            x = new Float32Array(frames * BERT_EMBED);
        for (let frame = 0; frame < frames; frame += 1) {
            const
                token = ids[frame],
                position = frame,
                base = frame * BERT_EMBED;
            for (let i = 0; i < BERT_EMBED; i += 1) {
                x[base + i] = weights.tokenEmbed[token * BERT_EMBED + i] +
                    weights.positionEmbed[position * BERT_EMBED + i] +
                    weights.typeEmbed[i];
            }
        }
        const embedded = layerNorm({
            input: x,
            frames,
            channels: BERT_EMBED,
            weight: weights.embedNorm,
            bias: weights.embedNormBias,
            eps: BERT_EPS,
        });
        let state = linear({
            input: embedded,
            rows: frames,
            inDim: BERT_EMBED,
            weight: weights.embedProject,
            bias: weights.embedProjectBias,
            outDim: BERT_HIDDEN,
        });
        for (let layer = 0; layer < BERT_LAYERS; layer += 1) {
            const
                q = linear({
                    input: state,
                    rows: frames,
                    inDim: BERT_HIDDEN,
                    weight: bert.query,
                    bias: bert.queryBias,
                    outDim: BERT_HIDDEN,
                }),
                k = linear({
                    input: state,
                    rows: frames,
                    inDim: BERT_HIDDEN,
                    weight: bert.key,
                    bias: bert.keyBias,
                    outDim: BERT_HIDDEN,
                }),
                v = linear({
                    input: state,
                    rows: frames,
                    inDim: BERT_HIDDEN,
                    weight: bert.value,
                    bias: bert.valueBias,
                    outDim: BERT_HIDDEN,
                }),
                attended = attention({
                    q,
                    k,
                    v,
                    frames,
                    dim: BERT_HIDDEN,
                    heads: BERT_HEADS,
                }),
                projected = linear({
                    input: attended,
                    rows: frames,
                    inDim: BERT_HIDDEN,
                    weight: bert.output,
                    bias: bert.outputBias,
                    outDim: BERT_HIDDEN,
                });
            for (let i = 0; i < state.length; i += 1) {
                state[i] += projected[i];
            }
            state = layerNorm({
                input: state,
                frames,
                channels: BERT_HIDDEN,
                weight: bert.attentionNorm,
                bias: bert.attentionNormBias,
                eps: BERT_EPS,
            });
            const feed = linear({
                input: state,
                rows: frames,
                inDim: BERT_HIDDEN,
                weight: bert.feed,
                bias: bert.feedBias,
                outDim: BERT_FF,
            });
            geluInPlace(feed);
            const down = linear({
                input: feed,
                rows: frames,
                inDim: BERT_FF,
                weight: bert.down,
                bias: bert.downBias,
                outDim: BERT_HIDDEN,
            });
            for (let i = 0; i < state.length; i += 1) {
                state[i] += down[i];
            }
            state = layerNorm({
                input: state,
                frames,
                channels: BERT_HIDDEN,
                weight: bert.feedNorm,
                bias: bert.feedNormBias,
                eps: BERT_EPS,
            });
        }
        return linear({
            input: state,
            rows: frames,
            inDim: BERT_HIDDEN,
            weight: weights.bertProject,
            bias: weights.bertProjectBias,
            outDim: HIDDEN,
        });
    },
    /** Run the phoneme text encoder. */
    runTextEncoder = (options: {
        ids: number[];
        weights: KokoroWeights;
    }): Float32Array => {
        const
            { ids, weights } = options,
            frames = ids.length,
            text = weights.text,
            x = new Float32Array(frames * HIDDEN);
        for (let frame = 0; frame < frames; frame += 1) {
            x.set(
                text.embed.subarray(
                    ids[frame] * HIDDEN,
                    (ids[frame] + 1) * HIDDEN,
                ),
                frame * HIDDEN,
            );
        }
        let state: Float32Array = x;
        for (const block of text.conv) {
            state = conv1d({
                input: state,
                frames,
                channels: HIDDEN,
                weight: block.weight,
                bias: block.bias,
                outChannels: HIDDEN,
                kernel: 5,
                padding: 2,
            });
            layerNormInto({
                values: state,
                frames,
                channels: HIDDEN,
                weight: block.norm,
                bias: block.normBias,
                eps: TEXT_EPS,
            });
            leakyInPlace({ values: state, slope: 0.2 });
        }
        return lstmBidir({
            input: state,
            frames,
            inDim: HIDDEN,
            hidden: HIDDEN / 2,
            forward: text.forward,
            backward: text.backward,
        });
    },
    /** Layer normalization written back into the same buffer. */
    layerNormInto = (options: {
        values: Float32Array;
        frames: number;
        channels: number;
        weight: Float32Array;
        bias: Float32Array;
        eps: number;
    }): void => {
        const normalized = layerNorm({
            input: options.values,
            frames: options.frames,
            channels: options.channels,
            weight: options.weight,
            bias: options.bias,
            eps: options.eps,
        });
        options.values.set(normalized);
    },
    /** Split a style row into predictor and decoder halves. */
    styleRow = (options: {
        pack: Float32Array;
        length: number;
    }): { predictor: Float32Array; decoder: Float32Array } => {
        const
            index = Math.min(
                Math.max(options.length, 1),
                MAX_PHONEMES,
            ) - 1,
            base = index * STYLE * 2;
        return {
            decoder: options.pack.slice(base, base + STYLE),
            predictor: options.pack.slice(base + STYLE, base + STYLE * 2),
        };
    },
    /** Broadcast one style vector over every frame. */
    repeatStyle = (options: {
        style: Float32Array;
        frames: number;
        channels: number;
    }): Float32Array => {
        const out = new Float32Array(options.frames * options.channels);
        for (let frame = 0; frame < options.frames; frame += 1) {
            out.set(options.style, frame * options.channels);
        }
        return out;
    },
    /** Predict phoneme durations and the aligned predictor features. */
    runPredictor = (options: {
        bert: Float32Array;
        ids: number[];
        style: Float32Array;
        weights: KokoroWeights;
    }): { durations: number[]; features: Float32Array; frames: number } => {
        const
            { bert, ids, style, weights } = options,
            rows = ids.length,
            styleBlock = repeatStyle({
                style,
                frames: rows,
                channels: STYLE,
            });
        let state: Float32Array = concatChannels({
            parts: [
                { data: bert, channels: HIDDEN },
                { data: styleBlock, channels: STYLE },
            ],
            frames: rows,
        });
        for (const block of weights.duration.encoder) {
            state = lstmBidir({
                input: state,
                frames: rows,
                inDim: HIDDEN + STYLE,
                hidden: HIDDEN / 2,
                forward: block.forward,
                backward: block.backward,
            });
            adalnInPlace({
                input: state,
                frames: rows,
                channels: HIDDEN,
                style,
                weight: block.norm,
                bias: block.normBias,
            });
            state = concatChannels({
                parts: [
                    { data: state, channels: HIDDEN },
                    { data: styleBlock, channels: STYLE },
                ],
                frames: rows,
            });
        }
        const
            /** Duration encoder output used by the alignment. */
            features = state,
            projected = lstmBidir({
                input: features,
                frames: rows,
                inDim: HIDDEN + STYLE,
                hidden: HIDDEN / 2,
                forward: weights.duration.forward,
                backward: weights.duration.backward,
            }),
            bins = linear({
                input: projected,
                rows,
                inDim: HIDDEN,
                weight: weights.duration.project,
                bias: weights.duration.projectBias,
                outDim: MAX_DUR,
            }),
            durations: number[] = [];
        let frames = 0;
        for (let row = 0; row < rows; row += 1) {
            let total = 0;
            for (let bin = 0; bin < MAX_DUR; bin += 1) {
                total += sigmoid(bins[row * MAX_DUR + bin]);
            }
            const value = Math.max(1, roundHalfEven(total));
            durations.push(value);
            frames += value;
        }
        return { durations, features, frames };
    },
    /** One adaptive residual block of the predictor or decoder. */
    runResBlock = (options: {
        input: Float32Array;
        frames: number;
        inChannels: number;
        outChannels: number;
        weights: ResBlockWeights;
        style: Float32Array;
        upsample: boolean;
    }): { data: Float32Array; frames: number } => {
        const
            { input, inChannels, outChannels, weights, style } = options,
            work = input.slice();
        adainInPlace({
            input: work,
            frames: options.frames,
            channels: inChannels,
            style,
            weight: weights.norm1,
            bias: weights.norm1Bias,
        });
        leakyInPlace({ values: work, slope: 0.2 });
        let
            /** Working features. */
            state: Float32Array = work,
            /** Working frame count. */
            frames = options.frames;
        if (options.upsample) {
            state = depthwiseConvTranspose1d({
                input: state,
                frames,
                channels: inChannels,
                weight: weights.pool as Float32Array,
                bias: weights.poolBias,
                kernel: 3,
                stride: 2,
                padding: 1,
                outputPadding: 1,
            });
            frames *= 2;
        }
        state = conv1d({
            input: state,
            frames,
            channels: inChannels,
            weight: weights.conv1,
            bias: weights.conv1Bias,
            outChannels,
            kernel: 3,
            padding: 1,
        });
        adainInPlace({
            input: state,
            frames,
            channels: outChannels,
            style,
            weight: weights.norm2,
            bias: weights.norm2Bias,
        });
        leakyInPlace({ values: state, slope: 0.2 });
        state = conv1d({
            input: state,
            frames,
            channels: outChannels,
            weight: weights.conv2,
            bias: weights.conv2Bias,
            outChannels,
            kernel: 3,
            padding: 1,
        });
        let shortcut = options.upsample ? upsample2({
            input,
            frames: options.frames,
            channels: inChannels,
        }) : input;
        if (weights.shortcut) {
            shortcut = conv1d({
                input: shortcut,
                frames,
                channels: inChannels,
                weight: weights.shortcut,
                outChannels,
                kernel: 1,
            });
        }
        const scale = 1 / Math.SQRT2;
        for (let i = 0; i < state.length; i += 1) {
            state[i] = (state[i] + shortcut[i]) * scale;
        }
        return { data: state, frames };
    },
    /** Run one pitch or energy stack. */
    runCurve = (options: {
        input: Float32Array;
        frames: number;
        blocks: ResBlockWeights[];
        style: Float32Array;
        project: Float32Array;
        projectBias: Float32Array;
    }): Float32Array => {
        let
            /** Working features. */
            data = options.input,
            /** Working frame count. */
            frames = options.frames;
        for (let index = 0; index < options.blocks.length; index += 1) {
            const result = runResBlock({
                input: data,
                frames,
                inChannels: index === 2 ? HIDDEN / 2 : HIDDEN,
                outChannels: index === 0 ? HIDDEN : HIDDEN / 2,
                weights: options.blocks[index],
                style: options.style,
                upsample: index === 1,
            });
            data = result.data;
            frames = result.frames;
        }
        return project({
            input: data,
            frames,
            channels: HIDDEN / 2,
            weight: options.project,
            bias: options.projectBias,
        });
    },
    /** Run one generator residual block. */
    runGeneratorBlock = (options: {
        input: Float32Array;
        frames: number;
        channels: number;
        weights: GeneratorBlockWeights;
        style: Float32Array;
    }): Float32Array => {
        const
            { input, frames, channels, weights, style } = options,
            dilations = [1, 3, 5];
        let state = input.slice();
        for (let index = 0; index < 3; index += 1) {
            const work = state.slice();
            adainInPlace({
                input: work,
                frames,
                channels,
                style,
                weight: weights.norm1[index],
                bias: weights.norm1Bias[index],
            });
            snakeInPlace({
                values: work,
                frames,
                channels,
                alpha: weights.alpha1[index],
            });
            let converted = conv1d({
                input: work,
                frames,
                channels,
                weight: weights.conv1[index],
                bias: weights.conv1Bias[index],
                outChannels: channels,
                kernel: weights.kernel,
                padding: dilations[index] * (weights.kernel - 1) / 2,
                dilation: dilations[index],
            });
            adainInPlace({
                input: converted,
                frames,
                channels,
                style,
                weight: weights.norm2[index],
                bias: weights.norm2Bias[index],
            });
            snakeInPlace({
                values: converted,
                frames,
                channels,
                alpha: weights.alpha2[index],
            });
            converted = conv1d({
                input: converted,
                frames,
                channels,
                weight: weights.conv2[index],
                bias: weights.conv2Bias[index],
                outChannels: channels,
                kernel: weights.kernel,
                padding: (weights.kernel - 1) / 2,
            });
            for (let i = 0; i < state.length; i += 1) {
                state[i] += converted[i];
            }
        }
        return state;
    },
    /** Build the harmonic source spectrum of the generator. */
    makeHarmonic = (options: {
        pitch: Float32Array;
        frames: number;
        source: Float32Array;
        sourceBias: Float32Array;
        noise: { uniform: () => number; normal: () => number };
    }): { data: Float32Array; frames: number } => {
        const
            { pitch, frames, noise } = options,
            rows = frames * 2,
            high = rows * FRAME_RATE,
            f0 = new Float32Array(high),
            rad = new Float32Array(high * SOURCE_DIM);
        for (let i = 0; i < rows; i += 1) {
            f0.fill(pitch[i], i * FRAME_RATE, (i + 1) * FRAME_RATE);
        }
        const float = Math.fround;
        for (let frame = 0; frame < high; frame += 1) {
            const base = frame * SOURCE_DIM;
            for (let k = 0; k < SOURCE_DIM; k += 1) {
                const
                    value = float(
                        float(f0[frame] * (k + 1)) / SAMPLE_RATE,
                    );
                rad[base + k] = float(value - Math.floor(value));
            }
        }
        for (let k = 1; k < SOURCE_DIM; k += 1) {
            rad[k] = float(rad[k] + float(noise.uniform()));
        }
        const
            /** Low rate harmonic phase increments. */
            lowRad = new Float32Array(rows * SOURCE_DIM);
        for (let frame = 0; frame < rows; frame += 1) {
            const
                source = float(float((frame + 0.5) * FRAME_RATE) - 0.5),
                clamped = Math.max(0, Math.min(high - 1, source)),
                lower = Math.floor(clamped),
                upper = Math.min(lower + 1, high - 1),
                frac = float(clamped - lower);
            for (let k = 0; k < SOURCE_DIM; k += 1) {
                lowRad[frame * SOURCE_DIM + k] = float(
                    float(rad[lower * SOURCE_DIM + k] * (1 - frac)) +
                    float(rad[upper * SOURCE_DIM + k] * frac),
                );
            }
        }
        const
            /** Cumulative harmonic phase. */
            phase = new Float32Array(rows * SOURCE_DIM);
        for (let k = 0; k < SOURCE_DIM; k += 1) {
            let total = 0;
            for (let frame = 0; frame < rows; frame += 1) {
                total = float(total + lowRad[frame * SOURCE_DIM + k]);
                phase[frame * SOURCE_DIM + k] = float(total * TWO_PI);
            }
        }
        const
            /** Per harmonic sine wave. */
            sines = new Float32Array(high * SOURCE_DIM);
        for (let frame = 0; frame < high; frame += 1) {
            const
                source = float(float((frame + 0.5) / FRAME_RATE) - 0.5),
                clamped = Math.max(0, Math.min(rows - 1, source)),
                lower = Math.floor(clamped),
                upper = Math.min(lower + 1, rows - 1),
                frac = float(clamped - lower);
            for (let k = 0; k < SOURCE_DIM; k += 1) {
                const value = float(
                    float(
                        float(phase[lower * SOURCE_DIM + k] * FRAME_RATE) *
                        (1 - frac),
                    ) +
                    float(
                        float(phase[upper * SOURCE_DIM + k] * FRAME_RATE) *
                        frac,
                    ),
                );
                sines[frame * SOURCE_DIM + k] = float(
                    float(Math.sin(value)) * SINE_AMP,
                );
            }
        }
        const
            /** Merged harmonic source. */
            source = new Float32Array(high);
        for (let frame = 0; frame < high; frame += 1) {
            const
                voiced = f0[frame] > VOICED_FLOOR ? 1 : 0,
                amplitude = float(
                    float(voiced * NOISE_STD) +
                    float(float(1 - voiced) * SINE_AMP / 3),
                ),
                base = frame * SOURCE_DIM;
            let sum = options.sourceBias[0];
            for (let k = 0; k < SOURCE_DIM; k += 1) {
                const wave = float(
                    float(sines[base + k] * voiced) +
                    float(amplitude * float(noise.normal())),
                );
                sum = float(sum + float(wave * options.source[k]));
            }
            source[frame] = float(Math.tanh(sum));
        }
        return { data: source, frames: high };
    },
    /** Magnitude and phase of the harmonic source. */
    harmonicSpectrum = (options: {
        source: Float32Array;
        frames: number;
    }): { data: Float32Array; frames: number } => {
        const
            { source } = options,
            padded = new Float32Array(options.frames + N_FFT),
            pad = N_FFT / 2,
            frames = Math.floor(options.frames / HOP) + 1,
            data = new Float32Array(frames * BINS * 2),
            hann = new Float32Array(N_FFT),
            cosine = new Float32Array(BINS * N_FFT),
            sine = new Float32Array(BINS * N_FFT);
        for (let n = 0; n < N_FFT; n += 1) {
            hann[n] = 0.5 - 0.5 * Math.cos(TWO_PI * n / N_FFT);
        }
        padded.set(source, pad);
        for (let i = 0; i < pad; i += 1) {
            if (i + 1 < source.length) {
                padded[pad - 1 - i] = source[i + 1];
            }
            if (source.length - 2 - i >= 0) {
                padded[pad + source.length + i] =
                    source[source.length - 2 - i];
            }
        }
        for (let k = 0; k < BINS; k += 1) {
            for (let n = 0; n < N_FFT; n += 1) {
                const angle = -TWO_PI * k * n / N_FFT;
                cosine[k * N_FFT + n] = Math.cos(angle);
                sine[k * N_FFT + n] = Math.sin(angle);
            }
        }
        for (let frame = 0; frame < frames; frame += 1) {
            const base = frame * BINS * 2;
            for (let k = 0; k < BINS; k += 1) {
                let real = 0;
                let imaginary = 0;
                for (let n = 0; n < N_FFT; n += 1) {
                    const value = padded[frame * HOP + n] * hann[n];
                    real += value * cosine[k * N_FFT + n];
                    imaginary += value * sine[k * N_FFT + n];
                }
                data[base + k] = Math.sqrt(real * real + imaginary * imaginary);
                data[base + BINS + k] = Math.atan2(imaginary, real);
            }
        }
        return { data, frames };
    },
    /** Inverse short time Fourier transform with window overlap. */
    inverseStft = (options: {
        magnitude: Float32Array;
        phase: Float32Array;
        frames: number;
    }): Float32Array => {
        const
            { magnitude, phase, frames } = options,
            pad = N_FFT / 2,
            padded = (frames - 1) * HOP + N_FFT,
            audio = padded - 2 * pad,
            values = new Float32Array(padded),
            weights = new Float32Array(padded),
            hann = new Float32Array(N_FFT),
            cosine = new Float32Array(BINS * N_FFT),
            sine = new Float32Array(BINS * N_FFT);
        for (let n = 0; n < N_FFT; n += 1) {
            hann[n] = 0.5 - 0.5 * Math.cos(TWO_PI * n / N_FFT);
        }
        for (let k = 0; k < BINS; k += 1) {
            for (let n = 0; n < N_FFT; n += 1) {
                const angle = TWO_PI * k * n / N_FFT;
                cosine[k * N_FFT + n] = Math.cos(angle);
                sine[k * N_FFT + n] = Math.sin(angle);
            }
        }
        for (let frame = 0; frame < frames; frame += 1) {
            const base = frame * BINS;
            for (let n = 0; n < N_FFT; n += 1) {
                let sum = magnitude[base];
                const last = magnitude[base + BINS - 1] *
                    Math.cos(phase[base + BINS - 1]);
                sum += n % 2 === 0 ? last : -last;
                for (let k = 1; k < BINS - 1; k += 1) {
                    const
                        radius = magnitude[base + k],
                        angle = phase[base + k],
                        real = radius * Math.cos(angle),
                        imaginary = radius * Math.sin(angle);
                    sum += 2 * (
                        real * cosine[k * N_FFT + n] -
                        imaginary * sine[k * N_FFT + n]
                    );
                }
                values[frame * HOP + n] += sum / N_FFT * hann[n];
                weights[frame * HOP + n] += hann[n] * hann[n];
            }
        }
        const out = new Float32Array(audio);
        for (let i = 0; i < audio; i += 1) {
            const weight = weights[pad + i];
            out[i] = weight > 1e-11 ? values[pad + i] / weight : 0;
        }
        return out;
    },
    /** Run the decoder body and generator for one token sequence. */
    runAudio = (options: {
        ids: number[];
        weights: KokoroWeights;
        pack: Float32Array;
    }): Float32Array => {
        const
            { ids, weights, pack } = options,
            padded = [0, ...ids, 0],
            rows = padded.length,
            style = styleRow({ pack, length: ids.length }),
            bert = runBert({ ids: padded, weights }),
            textEncoded = runTextEncoder({ ids: padded, weights }),
            predictor = runPredictor({
                bert,
                ids: padded,
                style: style.predictor,
                weights,
            }),
            asr = alignRepeat({
                input: textEncoded,
                rows,
                channels: HIDDEN,
                durations: predictor.durations,
            }),
            aligned = alignRepeat({
                input: predictor.features,
                rows,
                channels: HIDDEN + STYLE,
                durations: predictor.durations,
            }),
            frames = predictor.frames,
            shared = lstmBidir({
                input: aligned,
                frames,
                inDim: HIDDEN + STYLE,
                hidden: HIDDEN / 2,
                forward: weights.shared.forward,
                backward: weights.shared.backward,
            }),
            pitchCurve = runCurve({
                input: shared,
                frames,
                blocks: weights.pitch,
                style: style.predictor,
                project: weights.pitchProject,
                projectBias: weights.pitchProjectBias,
            }),
            energyCurve = runCurve({
                input: shared,
                frames,
                blocks: weights.energy,
                style: style.predictor,
                project: weights.energyProject,
                projectBias: weights.energyProjectBias,
            }),
            pitch = conv1d({
                input: pitchCurve,
                frames: frames * 2,
                channels: 1,
                weight: weights.decoder.pitchConv,
                bias: weights.decoder.pitchConvBias,
                outChannels: 1,
                kernel: 3,
                stride: 2,
                padding: 1,
            }),
            energy = conv1d({
                input: energyCurve,
                frames: frames * 2,
                channels: 1,
                weight: weights.decoder.energyConv,
                bias: weights.decoder.energyConvBias,
                outChannels: 1,
                kernel: 3,
                stride: 2,
                padding: 1,
            }),
            residual = conv1d({
                input: asr,
                frames,
                channels: HIDDEN,
                weight: weights.decoder.residual,
                bias: weights.decoder.residualBias,
                outChannels: DECODER_RESIDUAL,
                kernel: 1,
            });
        let
            /** Decoder features. */
            state = runResBlock({
                input: concatChannels({
                    parts: [
                        { data: asr, channels: HIDDEN },
                        { data: pitch, channels: 1 },
                        { data: energy, channels: 1 },
                    ],
                    frames,
                }),
                frames,
                inChannels: HIDDEN + 2,
                outChannels: DECODER_CHANNELS,
                weights: weights.decoder.encode,
                style: style.decoder,
                upsample: false,
            }).data,
            /** Decoder frame count. */
            decoderFrames = frames;
        for (let index = 0; index < weights.decoder.decode.length; index += 1) {
            const
                /** Block input channels. */
                inChannels = BEFORE_GENERATOR,
                /** Block output channels. */
                outChannels = index === 3
                    ? GENERATOR_CHANNELS
                    : DECODER_CHANNELS;
            const result = runResBlock({
                input: concatChannels({
                    parts: [
                        { data: state, channels: DECODER_CHANNELS },
                        { data: residual, channels: DECODER_RESIDUAL },
                        { data: pitch, channels: 1 },
                        { data: energy, channels: 1 },
                    ],
                    frames: decoderFrames,
                }),
                frames: decoderFrames,
                inChannels,
                outChannels,
                weights: weights.decoder.decode[index],
                style: style.decoder,
                upsample: index === 3,
            });
            state = result.data;
            decoderFrames = result.frames;
        }
        const
            /** Deterministic noise source. */
            noise = createNoise(NOISE_SEED),
            harmonic = makeHarmonic({
                pitch: pitchCurve,
                frames,
                source: weights.decoder.source,
                sourceBias: weights.decoder.sourceBias,
                noise,
            }),
            spectrum = harmonicSpectrum({
                source: harmonic.data,
                frames: harmonic.frames,
            }),
            residualBlocks = weights.decoder.residualBlocks;
        let
            /** Generator features. */
            generator: Float32Array = state.slice(),
            /** Generator frame count. */
            generatorFrames = decoderFrames,
            /** Harmonic frame count. */
            harmonicFrames = spectrum.frames;
        for (let level = 0; level < 2; level += 1) {
            leakyInPlace({ values: generator, slope: 0.1 });
            const noise = weights.decoder.noiseConv[level];
            let source = conv1d({
                input: spectrum.data,
                frames: harmonicFrames,
                channels: BINS * 2,
                weight: noise.weight,
                bias: noise.bias,
                outChannels: level === 0 ? 256 : 128,
                kernel: level === 0 ? 12 : 1,
                stride: noise.stride,
                padding: noise.padding,
            });
            const
                /** Frames produced by the noise convolution. */
                noiseFrames = Math.floor(
                    (harmonicFrames + 2 * noise.padding -
                        (level === 0 ? 12 : 1)) / noise.stride,
                ) + 1;
            source = runGeneratorBlock({
                input: source,
                frames: noiseFrames,
                channels: level === 0 ? 256 : 128,
                weights: weights.decoder.noiseResidual[level],
                style: style.decoder,
            });
            const up = weights.decoder.upsample[level];
            generator = convTranspose1d({
                input: generator,
                frames: generatorFrames,
                channels: level === 0 ? GENERATOR_CHANNELS : 256,
                weight: up.weight,
                bias: up.bias,
                outChannels: level === 0 ? 256 : 128,
                kernel: level === 0 ? 20 : 12,
                stride: level === 0 ? 10 : 6,
                padding: level === 0 ? 5 : 3,
            });
            generatorFrames *= level === 0 ? 10 : 6;
            if (level === 1) {
                generator = reflectPadLeft({
                    input: generator,
                    frames: generatorFrames,
                    channels: 128,
                });
                generatorFrames += 1;
            }
            for (let i = 0; i < generator.length; i += 1) {
                generator[i] += source[i];
            }
            const channels = level === 0 ? 256 : 128;
            let summed: Float32Array | undefined;
            for (let index = 0; index < 3; index += 1) {
                const block = runGeneratorBlock({
                    input: generator,
                    frames: generatorFrames,
                    channels,
                    weights: residualBlocks[level * 3 + index],
                    style: style.decoder,
                });
                if (!summed) {
                    summed = block;
                } else {
                    for (let i = 0; i < summed.length; i += 1) {
                        summed[i] += block[i];
                    }
                }
            }
            if (summed) {
                generator = summed;
                for (let i = 0; i < generator.length; i += 1) {
                    generator[i] /= 3;
                }
            }
        }
        leakyInPlace({ values: generator, slope: 0.01 });
        const
            /** Predictor spectrum of the generated audio. */
            output = conv1d({
                input: generator,
                frames: generatorFrames,
                channels: 128,
                weight: weights.decoder.output,
                bias: weights.decoder.outputBias,
                outChannels: BINS * 2,
                kernel: 7,
                padding: 3,
            }),
            magnitude = new Float32Array(harmonicFrames * BINS),
            phase = new Float32Array(harmonicFrames * BINS);
        for (let frame = 0; frame < harmonicFrames; frame += 1) {
            const base = frame * BINS * 2;
            for (let bin = 0; bin < BINS; bin += 1) {
                magnitude[frame * BINS + bin] = Math.exp(output[base + bin]);
                phase[frame * BINS + bin] = Math.sin(
                    output[base + BINS + bin],
                );
            }
        }
        return inverseStft({
            magnitude,
            phase,
            frames: harmonicFrames,
        });
    },
    /** Parse the model and voice files into a cached resident set. */
    loadModel = (options: {
        voice: VoiceId;
        model: Uint8Array;
        pack: Uint8Array;
    }): KokoroModel => {
        if (resident && resident.voice === options.voice) {
            return resident;
        }
        const
            file = parseGguf({ bytes: options.model }),
            voiceFile = parseGguf({ bytes: options.pack }),
            pack = tensor({ file: voiceFile, name: "voice.pack" }),
            weights = readWeights(file);
        resident = {
            voice: options.voice,
            weights,
            tokens: readTokens(file),
            pack,
        };
        return resident;
    };

/** Load or reuse the resident Kokoro model. */
const loadKokoro = async (options: {
    voice: VoiceId;
    load: (file: string) => Promise<Uint8Array>;
}): Promise<KokoroModel> => {
    if (resident && resident.voice === options.voice) {
        return resident;
    }
    const
        spec = getVoiceSpec(options.voice),
        [model, pack] = await Promise.all([
            options.load("kokoro-82m-q8_0.gguf"),
            options.load(`kokoro-voice-${spec.speaker}.gguf`),
        ]);
    return loadModel({ voice: options.voice, model, pack });
};

/** Load Kokoro without synthesizing. */
const prepareKokoro = async (options: {
    voice: VoiceId;
    load: (file: string) => Promise<Uint8Array>;
}): Promise<void> => {
    await loadKokoro(options);
};

/** Synthesize one line with the Kokoro voice. */
const synthesizeKokoro = async (options: {
    voice: VoiceId;
    text: string;
    load: (file: string) => Promise<Uint8Array>;
}): Promise<VoiceAudio> => {
    const
        residentModel = await loadKokoro({
            voice: options.voice,
            load: options.load,
        }),
        phonemes = phonemizeEnglish(options.text),
        ids = tokenize({
            text: phonemes,
            tokens: residentModel.tokens,
        });
    if (ids.length === 0) {
        return { pcm: new Float32Array(0), sampleRate: SAMPLE_RATE };
    }
    await prepareThreads();
    const pcm = runAudio({
        ids,
        weights: residentModel.weights,
        pack: residentModel.pack,
    });
    return { pcm, sampleRate: SAMPLE_RATE };
};

export { prepareKokoro, synthesizeKokoro };
