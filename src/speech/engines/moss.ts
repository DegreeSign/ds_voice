// MOSS-TTS-Nano 100M text to speech engine.
import { parseGguf } from "../../models/formats/gguf";
import { readGgufTensor } from "../../models/formats/ggufTensor";
import { createMossTokenizer } from "./mossTokenizer";
import { MOSS_REFERENCE } from "./mossReference";
import { decodeMossAudio, loadMossCodec } from "./mossCodec";
import { geluInPlace, layerNorm, linear } from "./ops";
import type { GgufFile } from "../../tensor/core/types";
import type { MossTokenizer } from "./mossTokenizer";
import type { MossCodec } from "./mossCodec";
import type { VoiceAudio, VoiceId } from "../types";
const
    /** Model shape of the global and local transformers. */
    HIDDEN = 768,
    LAYERS = 12,
    HEADS = 12,
    HEAD_DIM = 64,
    INTERMEDIATE = 3072,
    VOCAB = 16384,
    CODEBOOKS = 16,
    CODEBOOK_SIZE = 1024,
    AUDIO_PAD = 1024,
    AUDIO_START = 6,
    AUDIO_END = 7,
    AUDIO_USER_SLOT = 8,
    ASSISTANT_SLOT = 9,
    IM_START = 4,
    IM_END = 5,
    ROPE_BASE = 10000,
    NORM_EPS = 1e-5,
    /** Generation budget in audio frames. */
    MAX_FRAMES = 300,
    /** Smallest generation budget in audio frames. */
    MIN_FRAMES = 32,
    /** Generation frames allowed per text character. */
    FRAMES_PER_CHARACTER = 6,
    /** Sampling defaults of the reference inference script. */
    TEXT_TEMPERATURE = 1,
    TEXT_TOP_K = 50,
    TEXT_TOP_P = 1,
    AUDIO_TEMPERATURE = 0.8,
    AUDIO_TOP_K = 25,
    AUDIO_TOP_P = 0.95,
    AUDIO_REPETITION = 1.2,
    /** Fixed prompt pieces of the continuation template. */
    USER_PREFIX = "user\n",
    REFERENCE_PREFIX = "<user_inst>\n- Reference(s):\n",
    REFERENCE_NONE = "None",
    AFTER_REFERENCE = "\n- Instruction:\nNone\n- Tokens:\nNone\n" +
        "- Quality:\nNone\n- Sound Event:\nNone\n- Ambient Sound:\n" +
        "None\n- Language:\nNone\n- Text:\n",
    TEMPLATE_SUFFIX = "\n</user_inst>",
    ASSISTANT_TURN = "\n",
    ASSISTANT_ROLE = "assistant\n";

/** One transformer block of the MOSS checkpoint. */
interface MossLayer {
    normWeight: Float32Array;
    normBias: Float32Array;
    qkvWeight: Float32Array;
    qkvBias: Float32Array;
    outWeight: Float32Array;
    outBias: Float32Array;
    ffnNormWeight: Float32Array;
    ffnNormBias: Float32Array;
    upWeight: Float32Array;
    upBias: Float32Array;
    downWeight: Float32Array;
    downBias: Float32Array;
}

/** Every weight of one MOSS checkpoint. */
interface MossTensors {
    embed: Float32Array;
    layers: MossLayer[];
    outputNormWeight: Float32Array;
    outputNormBias: Float32Array;
    localLayers: MossLayer[];
    localOutputNormWeight: Float32Array;
    localOutputNormBias: Float32Array;
    audioEmbeddings: Float32Array[];
}

/** One resident model plus its tokenizer and codec. */
interface MossResident {
    voice: VoiceId;
    tensors: MossTensors;
    tokenizer: MossTokenizer;
    codec: MossCodec;
}

let
    /** Only one model stays resident. */
    resident: MossResident | undefined;

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

/** Read one transformer block. */
const readLayer = (options: {
    file: GgufFile;
    prefix: string;
}): MossLayer => {
    const
        { file, prefix } = options,
        read = (name: string): Float32Array => tensor({ file, name });
    return {
        normWeight: read(`${prefix}.attn_norm.weight`),
        normBias: read(`${prefix}.attn_norm.bias`),
        qkvWeight: read(`${prefix}.attn_qkv.weight`),
        qkvBias: read(`${prefix}.attn_qkv.bias`),
        outWeight: read(`${prefix}.attn_out.weight`),
        outBias: read(`${prefix}.attn_out.bias`),
        ffnNormWeight: read(`${prefix}.ffn_norm.weight`),
        ffnNormBias: read(`${prefix}.ffn_norm.bias`),
        upWeight: read(`${prefix}.ffn_up.weight`),
        upBias: read(`${prefix}.ffn_up.bias`),
        downWeight: read(`${prefix}.ffn_down.weight`),
        downBias: read(`${prefix}.ffn_down.bias`),
    };
};

/** Load every weight of one checkpoint. */
const loadTensors = (bytes: Uint8Array): MossTensors => {
    const
        file = parseGguf({ bytes }),
        read = (name: string): Float32Array => tensor({ file, name }),
        layers: MossLayer[] = [],
        localLayers: MossLayer[] = [],
        audioEmbeddings: Float32Array[] = [];
    for (let layer = 0; layer < LAYERS; layer += 1) {
        layers.push(readLayer({ file, prefix: `blk.${layer}` }));
    }
    localLayers.push(readLayer({ file, prefix: "local.blk.0" }));
    for (let channel = 0; channel < CODEBOOKS; channel += 1) {
        audioEmbeddings.push(read(`audio_embd.${channel}.weight`));
    }
    return {
        embed: read("token_embd.weight"),
        layers,
        outputNormWeight: read("output_norm.weight"),
        outputNormBias: read("output_norm.bias"),
        localLayers,
        localOutputNormWeight: read("local.output_norm.weight"),
        localOutputNormBias: read("local.output_norm.bias"),
        audioEmbeddings,
    };
};

/** RoPE frequencies of one attention head. */
const ropeFrequencies = (): Float32Array => {
    const out = new Float32Array(HEAD_DIM / 2);
    for (let i = 0; i < out.length; i += 1) {
        out[i] = Math.exp((-Math.log(ROPE_BASE) * 2 * i) / HEAD_DIM);
    }
    return out;
};

/** Apply RoPE to one row of a query or key tensor. */
const rotateRow = (options: {
    values: Float32Array;
    rowOffset: number;
    position: number;
    frequencies: Float32Array;
}): void => {
    const { values, rowOffset, position, frequencies } = options;
    for (let head = 0; head < HEADS; head += 1) {
        const base = rowOffset + head * HEAD_DIM;
        for (let i = 0; i < HEAD_DIM / 2; i += 1) {
            const
                index = base + i * 2,
                angle = position * frequencies[i],
                cos = Math.cos(angle),
                sin = Math.sin(angle),
                real = values[index],
                imag = values[index + 1];
            values[index] = real * cos - imag * sin;
            values[index + 1] = real * sin + imag * cos;
        }
    }
};

/** Run the feed forward part of one block. */
const feedForward = (options: {
    input: Float32Array;
    rows: number;
    layer: MossLayer;
}): Float32Array => {
    const
        { input, rows, layer } = options,
        up = linear({
            input,
            rows,
            inDim: HIDDEN,
            weight: layer.upWeight,
            bias: layer.upBias,
            outDim: INTERMEDIATE,
        });
    geluInPlace(up);
    return linear({
        input: up,
        rows,
        inDim: INTERMEDIATE,
        weight: layer.downWeight,
        bias: layer.downBias,
        outDim: HIDDEN,
    });
};

/** One transformer step with a key and value cache. */
const createStep = (options: {
    layers: MossLayer[];
    outputNormWeight: Float32Array;
    outputNormBias: Float32Array;
    cacheSize: number;
}): (input: Float32Array, position: number) => Float32Array => {
    const
        { layers, outputNormWeight, outputNormBias, cacheSize } = options,
        keys: Float32Array[] = [],
        values: Float32Array[] = [],
        frequencies = ropeFrequencies(),
        x = new Float32Array(HIDDEN),
        normed = new Float32Array(HIDDEN),
        qkv = new Float32Array(3 * HIDDEN),
        attention = new Float32Array(HIDDEN),
        scores = new Float32Array(cacheSize);
    for (let layer = 0; layer < layers.length; layer += 1) {
        keys.push(new Float32Array(cacheSize * HIDDEN));
        values.push(new Float32Array(cacheSize * HIDDEN));
    }
    return (input: Float32Array, position: number): Float32Array => {
        x.set(input);
        for (let layer = 0; layer < layers.length; layer += 1) {
            const weights = layers[layer];
            layerNorm({
                input: x,
                frames: 1,
                channels: HIDDEN,
                weight: weights.normWeight,
                bias: weights.normBias,
                eps: NORM_EPS,
                out: normed,
            });
            const projected = linear({
                input: normed,
                rows: 1,
                inDim: HIDDEN,
                weight: weights.qkvWeight,
                bias: weights.qkvBias,
                outDim: 3 * HIDDEN,
                out: qkv,
            });
            rotateRow({
                values: projected,
                rowOffset: 0,
                position,
                frequencies,
            });
            rotateRow({
                values: projected,
                rowOffset: HIDDEN,
                position,
                frequencies,
            });
            keys[layer].set(
                projected.subarray(HIDDEN, 2 * HIDDEN),
                position * HIDDEN,
            );
            values[layer].set(
                projected.subarray(2 * HIDDEN, 3 * HIDDEN),
                position * HIDDEN,
            );
            for (let head = 0; head < HEADS; head += 1) {
                const
                    queryOffset = head * HEAD_DIM,
                    scale = 1 / Math.sqrt(HEAD_DIM);
                let max = -Infinity;
                for (let key = 0; key <= position; key += 1) {
                    const offset = key * HIDDEN + head * HEAD_DIM;
                    let sum = 0;
                    for (let d = 0; d < HEAD_DIM; d += 1) {
                        sum += projected[queryOffset + d] *
                            keys[layer][offset + d];
                    }
                    const value = sum * scale;
                    scores[key] = value;
                    if (value > max) {
                        max = value;
                    }
                }
                let total = 0;
                for (let key = 0; key <= position; key += 1) {
                    const value = Math.exp(scores[key] - max);
                    scores[key] = value;
                    total += value;
                }
                for (let d = 0; d < HEAD_DIM; d += 1) {
                    let sum = 0;
                    for (let key = 0; key <= position; key += 1) {
                        sum += scores[key] / total *
                            values[layer][key * HIDDEN + head * HEAD_DIM + d];
                    }
                    attention[head * HEAD_DIM + d] = sum;
                }
            }
            const out = linear({
                input: attention,
                rows: 1,
                inDim: HIDDEN,
                weight: weights.outWeight,
                bias: weights.outBias,
                outDim: HIDDEN,
            });
            for (let i = 0; i < HIDDEN; i += 1) {
                x[i] += out[i];
            }
            layerNorm({
                input: x,
                frames: 1,
                channels: HIDDEN,
                weight: weights.ffnNormWeight,
                bias: weights.ffnNormBias,
                eps: NORM_EPS,
                out: normed,
            });
            const forward = feedForward({
                input: normed,
                rows: 1,
                layer: weights,
            });
            for (let i = 0; i < HIDDEN; i += 1) {
                x[i] += forward[i];
            }
        }
        const out = new Float32Array(HIDDEN);
        layerNorm({
            input: x,
            frames: 1,
            channels: HIDDEN,
            weight: outputNormWeight,
            bias: outputNormBias,
            eps: NORM_EPS,
            out,
        });
        return out;
    };
};

/** One global transformer step with a key and value cache. */
const createGlobalStep = (options: {
    tensors: MossTensors;
    cacheSize: number;
}): (options: {
    text: number;
    audio: Uint16Array | null;
    position: number;
}) => Float32Array => {
    const
        { tensors, cacheSize } = options,
        step = createStep({
            layers: tensors.layers,
            outputNormWeight: tensors.outputNormWeight,
            outputNormBias: tensors.outputNormBias,
            cacheSize,
        }),
        embedding = new Float32Array(HIDDEN);
    return (options: {
        text: number;
        audio: Uint16Array | null;
        position: number;
    }): Float32Array => {
        const { text, audio, position } = options;
        embedding.set(
            tensors.embed.subarray(text * HIDDEN, (text + 1) * HIDDEN),
        );
        if (audio) {
            for (let channel = 0; channel < CODEBOOKS; channel += 1) {
                const code = audio[channel];
                if (code === AUDIO_PAD || code >= CODEBOOK_SIZE) {
                    continue;
                }
                const
                    source = code * HIDDEN,
                    weight = tensors.audioEmbeddings[channel];
                for (let i = 0; i < HIDDEN; i += 1) {
                    embedding[i] += weight[source + i];
                }
            }
        }
        return step(embedding, position);
    };
};

/** One local transformer step over one frame prefix. */
const createLocalStep = (
    tensors: MossTensors,
): (input: Float32Array, position: number) => Float32Array =>
    createStep({
        layers: tensors.localLayers,
        outputNormWeight: tensors.localOutputNormWeight,
        outputNormBias: tensors.localOutputNormBias,
        cacheSize: 2 + CODEBOOKS,
    });

/** Deterministic random source for one text. */
const createRandom = (seed: number): (() => number) => {
    let state = seed >>> 0;
    return (): number => {
        state = (state + 0x6d2b79f5) >>> 0;
        let value = state;
        value = Math.imul(value ^ (value >>> 15), value | 1);
        value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
        return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
    };
};

/** Stable seed of one text line. */
const textSeed = (text: string): number => {
    let hash = 2166136261;
    for (let i = 0; i < text.length; i += 1) {
        hash ^= text.charCodeAt(i);
        hash = Math.imul(hash, 16777619);
    }
    return hash >>> 0;
};

/** Sample one token after top-k and top-p filtering. */
const sampleToken = (options: {
    logits: Float32Array;
    temperature: number;
    topK: number;
    topP: number;
    random: () => number;
}): number => {
    const { logits, temperature, topK, topP, random } = options;
    const
        limit = topK > 0 ? Math.min(topK, logits.length) : logits.length,
        indices: number[] = [];
    for (let index = 0; index < logits.length; index += 1) {
        const value = logits[index] / temperature;
        let at = indices.length;
        while (
            at > 0 &&
            logits[indices[at - 1]] / temperature < value
        ) {
            at -= 1;
        }
        if (at < limit) {
            indices.splice(at, 0, index);
            if (indices.length > limit) {
                indices.pop();
            }
        }
    }
    let max = -Infinity;
    for (const index of indices) {
        const value = logits[index] / temperature;
        if (value > max) {
            max = value;
        }
    }
    const weights: number[] = [];
    let total = 0;
    for (const index of indices) {
        const weight = Math.exp(logits[index] / temperature - max);
        weights.push(weight);
        total += weight;
    }
    let cutoff = weights.length;
    if (topP > 0 && topP < 1) {
        let cumulative = 0;
        for (let i = 0; i < weights.length; i += 1) {
            cumulative += weights[i] / total;
            if (cumulative > topP) {
                cutoff = i + 1;
                break;
            }
        }
    }
    let mass = 0;
    for (let i = 0; i < cutoff; i += 1) {
        mass += weights[i];
    }
    let target = random() * mass;
    for (let i = 0; i < cutoff; i += 1) {
        target -= weights[i];
        if (target <= 0) {
            return indices[i];
        }
    }
    return indices[cutoff - 1];
};

/** Apply the audio repetition penalty to one channel. */
const penalize = (options: {
    logits: Float32Array;
    seen: Uint8Array;
    channel: number;
    penalty: number;
}): void => {
    const { logits, seen, channel, penalty } = options;
    const base = channel * CODEBOOK_SIZE;
    for (let code = 0; code < CODEBOOK_SIZE; code += 1) {
        if (!seen[base + code]) {
            continue;
        }
        const value = logits[code];
        logits[code] = value > 0 ? value / penalty : value * penalty;
    }
};

/** Build the continuation prompt ids. */
const promptIds = (options: {
    tokenizer: MossTokenizer;
    text: string;
}): number[] => {
    const { tokenizer, text } = options;
    return [
        IM_START,
        ...tokenizer.encode(USER_PREFIX),
        ...tokenizer.encode(REFERENCE_PREFIX),
        ...tokenizer.encode(REFERENCE_NONE),
        ...tokenizer.encode(AFTER_REFERENCE),
        ...tokenizer.encode(text),
        ...tokenizer.encode(TEMPLATE_SUFFIX),
        IM_END,
        ...tokenizer.encode(ASSISTANT_TURN),
        IM_START,
        ...tokenizer.encode(ASSISTANT_ROLE),
        AUDIO_START,
    ];
};

/** One prompt row, text token plus optional audio codes. */
interface MossRow {
    text: number;
    audio: Uint16Array | null;
}

/** Build the voice clone prompt rows. */
const promptRows = (options: {
    tokenizer: MossTokenizer;
    text: string;
    reference: Uint16Array | null;
}): MossRow[] => {
    const { tokenizer, text, reference } = options;
    if (!reference || reference.length === 0) {
        return promptIds({ tokenizer, text }).map((id) => ({
            text: id,
            audio: null,
        }));
    }
    const rows: MossRow[] = [];
    const prefix = [
        IM_START,
        ...tokenizer.encode(USER_PREFIX),
        ...tokenizer.encode(REFERENCE_PREFIX),
        AUDIO_START,
    ];
    for (const id of prefix) {
        rows.push({ text: id, audio: null });
    }
    const frames = Math.floor(reference.length / CODEBOOKS);
    for (let frame = 0; frame < frames; frame += 1) {
        rows.push({
            text: AUDIO_USER_SLOT,
            audio: reference.subarray(
                frame * CODEBOOKS,
                (frame + 1) * CODEBOOKS,
            ),
        });
    }
    const suffix = [
        AUDIO_END,
        ...tokenizer.encode(AFTER_REFERENCE),
        ...tokenizer.encode(text),
        ...tokenizer.encode(TEMPLATE_SUFFIX),
        IM_END,
        ...tokenizer.encode(ASSISTANT_TURN),
        IM_START,
        ...tokenizer.encode(ASSISTANT_ROLE),
        AUDIO_START,
    ];
    for (const id of suffix) {
        rows.push({ text: id, audio: null });
    }
    return rows;
};

/** Generate audio code frames with the global and local transformers. */
const generateFrames = (options: {
    tensors: MossTensors;
    tokenizer: MossTokenizer;
    text: string;
    maxFrames: number;
}): Uint16Array => {
    const
        { tensors, tokenizer, text } = options,
        prompt = promptRows({
            tokenizer,
            text,
            reference: MOSS_REFERENCE,
        }),
        maxFrames = Math.min(
            options.maxFrames,
            Math.max(MIN_FRAMES, Math.ceil(text.length * FRAMES_PER_CHARACTER)),
        ),
        step = createGlobalStep({
            tensors,
            cacheSize: prompt.length + maxFrames + 1,
        });
    let hidden: Float32Array = new Float32Array(HIDDEN);
    for (let position = 0; position < prompt.length; position += 1) {
        const row = prompt[position];
        hidden = step({
            text: row.text,
            audio: row.audio,
            position,
        });
    }
    const
        frames: Uint16Array[] = [],
        textScores = new Float32Array(2),
        audioScores = new Float32Array(CODEBOOK_SIZE),
        random = createRandom(textSeed(text)),
        seen = new Uint8Array(CODEBOOKS * CODEBOOK_SIZE);
    const localStep = createLocalStep(tensors);
    for (let frame = 0; frame < maxFrames; frame += 1) {
        const local = localStep(hidden, 0);
        textScores[0] = dot(tensors.embed, ASSISTANT_SLOT, local);
        textScores[1] = dot(tensors.embed, AUDIO_END, local);
        const textToken = sampleToken({
            logits: textScores,
            temperature: TEXT_TEMPERATURE,
            topK: TEXT_TOP_K,
            topP: TEXT_TOP_P,
            random,
        });
        if (textToken === 1) {
            break;
        }
        const current = tensors.embed.subarray(
            ASSISTANT_SLOT * HIDDEN,
            (ASSISTANT_SLOT + 1) * HIDDEN,
        );
        let localHidden = localStep(current, 1);
        const codes = new Uint16Array(CODEBOOKS);
        for (let channel = 0; channel < CODEBOOKS; channel += 1) {
            const values = linear({
                input: localHidden,
                rows: 1,
                inDim: HIDDEN,
                weight: tensors.audioEmbeddings[channel],
                outDim: CODEBOOK_SIZE,
                out: audioScores,
            });
            penalize({
                logits: values,
                seen,
                channel,
                penalty: AUDIO_REPETITION,
            });
            const code = sampleToken({
                logits: values,
                temperature: AUDIO_TEMPERATURE,
                topK: AUDIO_TOP_K,
                topP: AUDIO_TOP_P,
                random,
            });
            codes[channel] = code;
            seen[channel * CODEBOOK_SIZE + code] = 1;
            if (channel + 1 < CODEBOOKS) {
                const weight = tensors.audioEmbeddings[channel];
                localHidden = localStep(
                    weight.subarray(code * HIDDEN, (code + 1) * HIDDEN),
                    2 + channel,
                );
            }
        }
        frames.push(codes);
        hidden = step({
            text: ASSISTANT_SLOT,
            audio: codes,
            position: prompt.length + frame,
        });
    }
    const out = new Uint16Array(frames.length * CODEBOOKS);
    for (let frame = 0; frame < frames.length; frame += 1) {
        out.set(frames[frame], frame * CODEBOOKS);
    }
    return out;
};

/** Dot product of one embedding row and a vector. */
const dot = (
    weight: Float32Array,
    row: number,
    input: Float32Array,
): number => {
    const source = row * HIDDEN;
    let sum = 0;
    for (let i = 0; i < HIDDEN; i += 1) {
        sum += weight[source + i] * input[i];
    }
    return sum;
};

/** Load or reuse the resident MOSS set. */
const loadMoss = async (options: {
    voice: VoiceId;
    load: (file: string) => Promise<Uint8Array>;
}): Promise<MossResident> => {
    if (resident && resident.voice === options.voice) {
        return resident;
    }
    const [model, codec, tokenizerJson] = await Promise.all([
        options.load("moss_nano.gguf"),
        options.load("moss_codec.gguf"),
        options.load("tokenizer/tokenizer.json"),
    ]);
    resident = {
        voice: options.voice,
        tensors: loadTensors(model),
        codec: loadMossCodec(codec),
        tokenizer: createMossTokenizer(new TextDecoder().decode(tokenizerJson)),
    };
    return resident;
};

/** Load MOSS without synthesizing. */
const prepareMoss = async (options: {
    voice: VoiceId;
    load: (file: string) => Promise<Uint8Array>;
}): Promise<void> => {
    await loadMoss(options);
};

/** Synthesize one line with the MOSS backbone. */
const synthesizeMoss = async (options: {
    voice: VoiceId;
    text: string;
    load: (file: string) => Promise<Uint8Array>;
    maxFrames?: number;
}): Promise<VoiceAudio> => {
    const
        loaded = await loadMoss({
            voice: options.voice,
            load: options.load,
        }),
        codes = generateFrames({
            tensors: loaded.tensors,
            tokenizer: loaded.tokenizer,
            text: options.text,
            maxFrames: options.maxFrames ?? MAX_FRAMES,
        }),
        frames = codes.length / CODEBOOKS,
        pcm = decodeMossAudio({
            codec: loaded.codec,
            codes,
            frames,
        });
    for (let i = 0; i < pcm.length; i += 1) {
        if (pcm[i] > 1) {
            pcm[i] = 1;
        } else if (pcm[i] < -1) {
            pcm[i] = -1;
        }
    }
    return { pcm, sampleRate: 48000 };
};

export { prepareMoss, synthesizeMoss };
