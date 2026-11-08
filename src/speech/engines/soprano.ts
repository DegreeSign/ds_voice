// Soprano 1.1 80M text to speech engine.
import { parseGguf } from "../../models/formats/gguf";
import { readGgufTensor } from "../../models/formats/ggufTensor";
import { inverseStft } from "./fft";
import { createTokenizer } from "./tokenizer";
import type { GgufFile } from "../../tensor/core/types";
import type { SopranoTokenizer } from "./tokenizer";
import type { VoiceAudio, VoiceId } from "../types";
const
    /** Model shape of the Qwen3 backbone. */
    HIDDEN = 512,
    LAYERS = 17,
    HEADS = 4,
    KV_HEADS = 1,
    HEAD_DIM = 128,
    INTERMEDIATE = 2304,
    VOCAB = 8192,
    RMS_EPS = 1e-6,
    ROPE_THETA = 10000,
    /** Decoder shape of the Vocos vocoder. */
    DECODER_DIM = 768,
    DECODER_INTER = 2304,
    DECODER_LAYERS = 8,
    N_FFT = 2048,
    HOP = 512,
    BINS = N_FFT / 2 + 1,
    UPSCALE = 4,
    TOKEN_SAMPLES = 2048,
    SAMPLE_RATE = 32000,
    /** Default generation budget. */
    MAX_TOKENS = 120,
    REPETITION_PENALTY = 1.2,
    /** Linear interpolation of the hidden states. */
    interpolate = (options: {
        hidden: Float32Array;
        tokens: number;
    }): { data: Float32Array; frames: number } => {
        const
            { hidden, tokens } = options,
            frames = UPSCALE * (tokens - 1) + 1,
            out = new Float32Array(frames * HIDDEN);
        for (let frame = 0; frame < frames; frame += 1) {
            const
                position = frame / UPSCALE,
                lower = Math.floor(position),
                upper = Math.min(tokens - 1, lower + 1),
                frac = position - lower;
            for (let c = 0; c < HIDDEN; c += 1) {
                out[frame * HIDDEN + c] = hidden[lower * HIDDEN + c] *
                    (1 - frac) + hidden[upper * HIDDEN + c] * frac;
            }
        }
        return { data: out, frames };
    },
    /** Root mean square normalization over one row. */
    rmsNorm = (options: {
        input: Float32Array;
        offset: number;
        dim: number;
        weight: Float32Array;
        out: Float32Array;
        outOffset?: number;
        eps?: number;
    }): void => {
        const
            { input, offset, dim, weight, out } = options,
            outOffset = options.outOffset ?? 0,
            eps = options.eps ?? RMS_EPS;
        let sum = 0;
        for (let i = 0; i < dim; i += 1) {
            const value = input[offset + i];
            sum += value * value;
        }
        const scale = 1 / Math.sqrt(sum / dim + eps);
        for (let i = 0; i < dim; i += 1) {
            out[outOffset + i] = input[offset + i] * scale * weight[i];
        }
    },
    /** Matrix vector product for one row. */
    rowLinear = (options: {
        input: Float32Array;
        inputOffset: number;
        weight: Float32Array;
        outDim: number;
        inDim: number;
        out: Float32Array;
        outOffset: number;
    }): void => {
        const
            { input, inputOffset, weight, outDim, inDim, out } = options,
            outOffset = options.outOffset;
        for (let o = 0; o < outDim; o += 1) {
            let sum = 0;
            const base = o * inDim;
            for (let i = 0; i < inDim; i += 1) {
                sum += input[inputOffset + i] * weight[base + i];
            }
            out[outOffset + o] = sum;
        }
    },
    /** SiLU activation. */
    silu = (value: number): number => value / (1 + Math.exp(-value)),
    /** Error function approximation. */
    erf = (value: number): number => {
        const
            sign = value < 0 ? -1 : 1,
            x = Math.abs(value),
            t = 1 / (1 + 0.3275911 * x),
            poly = t * (0.254829592 + t * (-0.284496736 + t * (
                1.421413741 + t * (-1.453152027 + t * 1.061405429)
            ))),
            result = 1 - poly * Math.exp(-x * x);
        return sign * result;
    },
    /** GELU activation, exact form. */
    gelu = (value: number): number =>
        0.5 * value * (1 + erf(value / Math.SQRT2));

/** Shapes of one loaded Soprano checkpoint. */
interface SopranoTensors {
    /** Token embedding rows, one per vocabulary entry. */
    embed: Float32Array;
    /** Output projection rows, one per vocabulary entry. */
    output: Float32Array;
    /** Final norm weight. */
    outputNorm: Float32Array;
    /** Per layer weights. */
    layers: Array<{
        attnNorm: Float32Array;
        ffnNorm: Float32Array;
        q: Float32Array;
        k: Float32Array;
        v: Float32Array;
        o: Float32Array;
        qNorm: Float32Array;
        kNorm: Float32Array;
        gate: Float32Array;
        up: Float32Array;
        down: Float32Array;
    }>;
    /** Decoder weights. */
    decoder: {
        embed: Float32Array;
        embedBias: Float32Array;
        norm: Float32Array;
        normBias: Float32Array;
        blocks: Array<{
            gamma: Float32Array;
            dw: Float32Array;
            dwBias: Float32Array;
            norm: Float32Array;
            normBias: Float32Array;
            pw1: Float32Array;
            pw1Bias: Float32Array;
            pw2: Float32Array;
            pw2Bias: Float32Array;
        }>;
        finalNorm: Float32Array;
        finalNormBias: Float32Array;
        head: Float32Array;
        headBias: Float32Array;
    };
}

/** One resident model plus its tokenizer. */
interface SopranoResident {
    voice: VoiceId;
    tensors: SopranoTensors;
    tokenizer: SopranoTokenizer;
}

let
    /** Only one model stays resident. */
    resident: SopranoResident | undefined;

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

/** Load the language and decoder weights. */
const loadTensors = (options: {
    model: Uint8Array;
    codec: Uint8Array;
}): SopranoTensors => {
    const
        model = parseGguf({ bytes: options.model }),
        codec = parseGguf({ bytes: options.codec }),
        read = (name: string): Float32Array => tensor({ file: model, name }),
        readCodec = (name: string): Float32Array =>
            tensor({ file: codec, name }),
        layers: SopranoTensors["layers"] = [];
    for (let layer = 0; layer < LAYERS; layer += 1) {
        const prefix = `blk.${layer}`;
        layers.push({
            attnNorm: read(`${prefix}.attn_norm.weight`),
            ffnNorm: read(`${prefix}.ffn_norm.weight`),
            q: read(`${prefix}.attn_q.weight`),
            k: read(`${prefix}.attn_k.weight`),
            v: read(`${prefix}.attn_v.weight`),
            o: read(`${prefix}.attn_output.weight`),
            qNorm: read(`${prefix}.attn_q_norm.weight`),
            kNorm: read(`${prefix}.attn_k_norm.weight`),
            gate: read(`${prefix}.ffn_gate.weight`),
            up: read(`${prefix}.ffn_up.weight`),
            down: read(`${prefix}.ffn_down.weight`),
        });
    }
    const blocks: SopranoTensors["decoder"]["blocks"] = [];
    for (let block = 0; block < DECODER_LAYERS; block += 1) {
        const prefix = `sop.decode.cnx.${block}`;
        blocks.push({
            gamma: readCodec(`${prefix}.gamma`),
            dw: readCodec(`${prefix}.dw.w`),
            dwBias: readCodec(`${prefix}.dw.b`),
            norm: readCodec(`${prefix}.ln.w`),
            normBias: readCodec(`${prefix}.ln.b`),
            pw1: readCodec(`${prefix}.pw1.w`),
            pw1Bias: readCodec(`${prefix}.pw1.b`),
            pw2: readCodec(`${prefix}.pw2.w`),
            pw2Bias: readCodec(`${prefix}.pw2.b`),
        });
    }
    return {
        embed: read("token_embd.weight"),
        output: read("output.weight"),
        outputNorm: read("output_norm.weight"),
        layers,
        decoder: {
            embed: readCodec("sop.decode.embed.w"),
            embedBias: readCodec("sop.decode.embed.b"),
            norm: readCodec("sop.decode.norm.w"),
            normBias: readCodec("sop.decode.norm.b"),
            blocks,
            finalNorm: readCodec("sop.decode.fln.w"),
            finalNormBias: readCodec("sop.decode.fln.b"),
            head: readCodec("sop.decode.head.out.w"),
            headBias: readCodec("sop.decode.head.out.b"),
        },
    };
};

/** Load or reuse the resident Soprano set. */
const loadSoprano = async (options: {
    voice: VoiceId;
    load: (file: string) => Promise<Uint8Array>;
}): Promise<SopranoResident> => {
    if (resident && resident.voice === options.voice) {
        return resident;
    }
    const [model, codec, tokenizerJson] = await Promise.all([
        options.load("Soprano-1.1-80M.F16.gguf"),
        options.load("codec-F32.gguf"),
        options.load("tokenizer.json"),
    ]);
    resident = {
        voice: options.voice,
        tensors: loadTensors({ model, codec }),
        tokenizer: createTokenizer(new TextDecoder().decode(tokenizerJson)),
    };
    return resident;
};

/** Load Soprano without synthesizing. */
const prepareSoprano = async (options: {
    voice: VoiceId;
    load: (file: string) => Promise<Uint8Array>;
}): Promise<void> => {
    await loadSoprano(options);
};

/** Run the decoder over the generated hidden states. */
const decode = (options: {
    tensors: SopranoTensors;
    hidden: Float32Array;
    tokens: number;
}): Float32Array => {
    const
        { tensors, hidden, tokens } = options,
        { data, frames } = interpolate({ hidden, tokens }),
        decoder = tensors.decoder,
        x = new Float32Array(frames * DECODER_DIM),
        normed = new Float32Array(frames * DECODER_DIM),
        projected = new Float32Array(frames * DECODER_INTER),
        gated = new Float32Array(frames * DECODER_INTER),
        pointwise = new Float32Array(frames * DECODER_DIM),
        conv = new Float32Array(frames * DECODER_DIM),
        head = new Float32Array(frames * (N_FFT + 2));
    for (let frame = 0; frame < frames; frame += 1) {
        rowLinear({
            input: data,
            inputOffset: frame * HIDDEN,
            weight: decoder.embed,
            outDim: DECODER_DIM,
            inDim: HIDDEN,
            out: x,
            outOffset: frame * DECODER_DIM,
        });
    }
    for (let frame = 0; frame < frames; frame += 1) {
        for (let c = 0; c < DECODER_DIM; c += 1) {
            x[frame * DECODER_DIM + c] += decoder.embedBias[c];
        }
    }
    const layerNorm = (input: Float32Array, weight: Float32Array,
        bias: Float32Array, out: Float32Array): void => {
        for (let frame = 0; frame < frames; frame += 1) {
            const offset = frame * DECODER_DIM;
            let mean = 0;
            for (let c = 0; c < DECODER_DIM; c += 1) {
                mean += input[offset + c];
            }
            mean /= DECODER_DIM;
            let variance = 0;
            for (let c = 0; c < DECODER_DIM; c += 1) {
                const diff = input[offset + c] - mean;
                variance += diff * diff;
            }
            const rstd = 1 / Math.sqrt(variance / DECODER_DIM + 1e-6);
            for (let c = 0; c < DECODER_DIM; c += 1) {
                out[offset + c] = (input[offset + c] - mean) * rstd *
                    weight[c] + bias[c];
            }
        }
    };
    layerNorm(x, decoder.norm, decoder.normBias, normed);
    x.set(normed);
    for (let block = 0; block < DECODER_LAYERS; block += 1) {
        const weights = decoder.blocks[block];
        // Depthwise convolution, kernel three, zero padding.
        for (let frame = 0; frame < frames; frame += 1) {
            for (let c = 0; c < DECODER_DIM; c += 1) {
                let sum = weights.dwBias[c];
                for (let k = 0; k < 3; k += 1) {
                    const source = frame + k - 1;
                    if (source >= 0 && source < frames) {
                        sum += x[source * DECODER_DIM + c] *
                            weights.dw[c * 3 + k];
                    }
                }
                conv[frame * DECODER_DIM + c] = sum;
            }
        }
        layerNorm(conv, weights.norm, weights.normBias, normed);
        for (let frame = 0; frame < frames; frame += 1) {
            rowLinear({
                input: normed,
                inputOffset: frame * DECODER_DIM,
                weight: weights.pw1,
                outDim: DECODER_INTER,
                inDim: DECODER_DIM,
                out: projected,
                outOffset: frame * DECODER_INTER,
            });
            for (let i = 0; i < DECODER_INTER; i += 1) {
                gated[frame * DECODER_INTER + i] = gelu(
                    projected[frame * DECODER_INTER + i] +
                    weights.pw1Bias[i],
                );
            }
            rowLinear({
                input: gated,
                inputOffset: frame * DECODER_INTER,
                weight: weights.pw2,
                outDim: DECODER_DIM,
                inDim: DECODER_INTER,
                out: pointwise,
                outOffset: frame * DECODER_DIM,
            });
            for (let c = 0; c < DECODER_DIM; c += 1) {
                const offset = frame * DECODER_DIM + c;
                x[offset] += (pointwise[offset] + weights.pw2Bias[c]) *
                    weights.gamma[c];
            }
        }
    }
    layerNorm(x, decoder.finalNorm, decoder.finalNormBias, normed);
    for (let frame = 0; frame < frames; frame += 1) {
        rowLinear({
            input: normed,
            inputOffset: frame * DECODER_DIM,
            weight: decoder.head,
            outDim: N_FFT + 2,
            inDim: DECODER_DIM,
            out: head,
            outOffset: frame * (N_FFT + 2),
        });
        for (let i = 0; i < N_FFT + 2; i += 1) {
            head[frame * (N_FFT + 2) + i] += decoder.headBias[i];
        }
    }
    const
        magnitude = new Float32Array(frames * BINS),
        phase = new Float32Array(frames * BINS);
    for (let frame = 0; frame < frames; frame += 1) {
        const offset = frame * (N_FFT + 2);
        for (let bin = 0; bin < BINS; bin += 1) {
            magnitude[frame * BINS + bin] = Math.min(
                Math.exp(head[offset + bin]),
                1e2,
            );
            phase[frame * BINS + bin] = head[offset + BINS + bin];
        }
    }
    const audio = inverseStft({
        magnitude,
        phase,
        bins: BINS,
        frames,
        nFft: N_FFT,
        hop: HOP,
    });
    return audio;
};

/** Synthesize one line with the Soprano backbone. */
const synthesizeSoprano = async (options: {
    voice: VoiceId;
    text: string;
    load: (file: string) => Promise<Uint8Array>;
    maxTokens?: number;
}): Promise<VoiceAudio> => {
    const
        loaded = await loadSoprano({
            voice: options.voice,
            load: options.load,
        }),
        { tokenizer, tensors } = loaded,
        prompt = tokenizer.prompt(options.text),
        maxTokens = options.maxTokens ?? MAX_TOKENS,
        cos = new Float32Array((prompt.length + maxTokens) * (HEAD_DIM / 2)),
        sin = new Float32Array((prompt.length + maxTokens) * (HEAD_DIM / 2));
    const
        /** Cache of keys and values per layer. */
        keyCache: Float32Array[] = [],
        valueCache: Float32Array[] = [],
        cacheSize = prompt.length + maxTokens;
    for (let layer = 0; layer < LAYERS; layer += 1) {
        keyCache.push(new Float32Array(cacheSize * HEAD_DIM));
        valueCache.push(new Float32Array(cacheSize * HEAD_DIM));
    }
    const invFreq = new Float32Array(HEAD_DIM / 2);
    for (let i = 0; i < HEAD_DIM / 2; i += 1) {
        invFreq[i] = 1 / Math.pow(ROPE_THETA, (2 * i) / HEAD_DIM);
    }
    for (let position = 0; position < cacheSize; position += 1) {
        for (let i = 0; i < HEAD_DIM / 2; i += 1) {
            const angle = position * invFreq[i];
            cos[position * (HEAD_DIM / 2) + i] = Math.cos(angle);
            sin[position * (HEAD_DIM / 2) + i] = Math.sin(angle);
        }
    }
    const
        x = new Float32Array(HIDDEN),
        xn = new Float32Array(HIDDEN),
        q = new Float32Array(HIDDEN),
        k = new Float32Array(HEAD_DIM),
        v = new Float32Array(HEAD_DIM),
        attention = new Float32Array(HIDDEN),
        gate = new Float32Array(INTERMEDIATE),
        up = new Float32Array(INTERMEDIATE),
        down = new Float32Array(HIDDEN),
        hiddenOut = new Float32Array(HIDDEN),
        logits = new Float32Array(VOCAB),
        scores = new Float32Array(cacheSize),
        /** Run one token through the cached backbone. */
        step = (token: number, position: number): Float32Array => {
            x.set(tensors.embed.subarray(token * HIDDEN, (token + 1) * HIDDEN));
            for (let layer = 0; layer < LAYERS; layer += 1) {
                const weights = tensors.layers[layer];
                rmsNorm({
                    input: x,
                    offset: 0,
                    dim: HIDDEN,
                    weight: weights.attnNorm,
                    out: xn,
                });
                rowLinear({
                    input: xn, inputOffset: 0, weight: weights.q,
                    outDim: HIDDEN, inDim: HIDDEN, out: q, outOffset: 0,
                });
                rowLinear({
                    input: xn, inputOffset: 0, weight: weights.k,
                    outDim: HEAD_DIM, inDim: HIDDEN, out: k, outOffset: 0,
                });
                rowLinear({
                    input: xn, inputOffset: 0, weight: weights.v,
                    outDim: HEAD_DIM, inDim: HIDDEN, out: v, outOffset: 0,
                });
                for (let head = 0; head < HEADS; head += 1) {
                    rmsNorm({
                        input: q, offset: head * HEAD_DIM, dim: HEAD_DIM,
                        weight: weights.qNorm, out: q,
                        outOffset: head * HEAD_DIM,
                    });
                }
                rmsNorm({
                    input: k, offset: 0, dim: HEAD_DIM,
                    weight: weights.kNorm, out: k,
                });
                const half = HEAD_DIM / 2;
                for (let head = 0; head < HEADS; head += 1) {
                    const base = head * HEAD_DIM;
                    for (let i = 0; i < half; i += 1) {
                        const
                            c = cos[position * half + i],
                            s = sin[position * half + i],
                            a = q[base + i],
                            b = q[base + half + i];
                        q[base + i] = a * c - b * s;
                        q[base + half + i] = b * c + a * s;
                    }
                }
                for (let i = 0; i < half; i += 1) {
                    const
                        c = cos[position * half + i],
                        s = sin[position * half + i],
                        a = k[i],
                        b = k[half + i];
                    k[i] = a * c - b * s;
                    k[half + i] = b * c + a * s;
                }
                keyCache[layer].set(k, position * HEAD_DIM);
                valueCache[layer].set(v, position * HEAD_DIM);
                // Grouped query attention, all cached positions.
                for (let head = 0; head < HEADS; head += 1) {
                    const base = head * HEAD_DIM;
                    let max = -Infinity;
                    for (let pos = 0; pos <= position; pos += 1) {
                        let sum = 0;
                        const keyBase = pos * HEAD_DIM;
                        for (let d = 0; d < HEAD_DIM; d += 1) {
                            sum += q[base + d] * keyCache[layer][keyBase + d];
                        }
                        sum /= Math.sqrt(HEAD_DIM);
                        scores[pos] = sum;
                        if (sum > max) {
                            max = sum;
                        }
                    }
                    let total = 0;
                    for (let pos = 0; pos <= position; pos += 1) {
                        const value = Math.exp(scores[pos] - max);
                        scores[pos] = value;
                        total += value;
                    }
                    for (let d = 0; d < HEAD_DIM; d += 1) {
                        let sum = 0;
                        for (let pos = 0; pos <= position; pos += 1) {
                            sum += scores[pos] / total *
                                valueCache[layer][pos * HEAD_DIM + d];
                        }
                        attention[base + d] = sum;
                    }
                }
                rowLinear({
                    input: attention, inputOffset: 0, weight: weights.o,
                    outDim: HIDDEN, inDim: HIDDEN, out: hiddenOut,
                    outOffset: 0,
                });
                for (let i = 0; i < HIDDEN; i += 1) {
                    x[i] += hiddenOut[i];
                }
                rmsNorm({
                    input: x, offset: 0, dim: HIDDEN,
                    weight: weights.ffnNorm, out: xn,
                });
                rowLinear({
                    input: xn, inputOffset: 0, weight: weights.gate,
                    outDim: INTERMEDIATE, inDim: HIDDEN, out: gate,
                    outOffset: 0,
                });
                rowLinear({
                    input: xn, inputOffset: 0, weight: weights.up,
                    outDim: INTERMEDIATE, inDim: HIDDEN, out: up,
                    outOffset: 0,
                });
                for (let i = 0; i < INTERMEDIATE; i += 1) {
                    gate[i] = silu(gate[i]) * up[i];
                }
                rowLinear({
                    input: gate, inputOffset: 0, weight: weights.down,
                    outDim: HIDDEN, inDim: INTERMEDIATE, out: down,
                    outOffset: 0,
                });
                for (let i = 0; i < HIDDEN; i += 1) {
                    x[i] += down[i];
                }
            }
            rmsNorm({
                input: x, offset: 0, dim: HIDDEN,
                weight: tensors.outputNorm, out: hiddenOut,
            });
            for (let tokenId = 0; tokenId < VOCAB; tokenId += 1) {
                let sum = 0;
                const base = tokenId * HIDDEN;
                for (let i = 0; i < HIDDEN; i += 1) {
                    sum += tensors.output[base + i] * hiddenOut[i];
                }
                logits[tokenId] = sum;
            }
            return hiddenOut;
        };
    let position = 0;
    let currentHidden = new Float32Array(HIDDEN);
    for (const token of prompt) {
        currentHidden = step(token, position).slice();
        position += 1;
    }
    const
        seen = new Map<number, number>(),
        hidden: number[] = [],
        generated: number[] = [];
    for (const token of prompt) {
        seen.set(token, (seen.get(token) ?? 0) + 1);
    }
    for (let stepIndex = 0; stepIndex < maxTokens; stepIndex += 1) {
        // Apply the repetition penalty over the tokens already produced.
        for (const [tokenId, count] of seen) {
            if (count > 0) {
                logits[tokenId] = logits[tokenId] > 0
                    ? logits[tokenId] / REPETITION_PENALTY
                    : logits[tokenId] * REPETITION_PENALTY;
            }
        }
        let best = 0;
        for (let tokenId = 1; tokenId < VOCAB; tokenId += 1) {
            if (logits[tokenId] > logits[best]) {
                best = tokenId;
            }
        }
        if (best === tokenizer.stop) {
            break;
        }
        seen.set(best, (seen.get(best) ?? 0) + 1);
        generated.push(best);
        for (let i = 0; i < HIDDEN; i += 1) {
            hidden.push(currentHidden[i]);
        }
        currentHidden = step(best, position).slice();
        position += 1;
    }
    const tokens = generated.length;
    if (tokens < 2) {
        return { pcm: new Float32Array(0), sampleRate: SAMPLE_RATE };
    }
    return {
        pcm: decode({
            tensors,
            hidden: Float32Array.from(hidden),
            tokens,
        }),
        sampleRate: SAMPLE_RATE,
    };
};

export { prepareSoprano, synthesizeSoprano };
