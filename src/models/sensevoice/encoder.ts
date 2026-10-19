// SenseVoice SAN-M encoder.
// Mirrors funasr SenseVoiceEncoderSmall.
import {
    addArrays,
    addInPlace,
    attention,
    layerNorm,
    linear,
    reluInPlace,
} from "../../tensor/kernels";
import type {
    SenseVoiceConfig,
    SenseVoiceEncodeOptions,
    SenseVoiceEncodeResult,
    SenseVoiceLogitsOptions,
} from "../../tensor/types";
const
    /** Sinusoidal positional encoding, positions are one based. */
    sinusoidalPe = (options: {
        depth: number;
        frames: number;
    }): Float32Array => {
        const
            { depth, frames } = options,
            out = new Float32Array(frames * depth),
            half = depth >> 1;
        if (half <= 1 || frames <= 0) {
            return out;
        }
        const
            logIncrement = Math.log(10000) / (half - 1),
            inv = new Float64Array(half);
        for (let k = 0; k < half; k += 1) {
            inv[k] = Math.exp(k * -logIncrement);
        }
        for (let i = 0; i < frames; i += 1) {
            const
                pos = i + 1,
                row = i * depth;
            for (let k = 0; k < half; k += 1) {
                const scaled = pos * inv[k];
                out[row + k] = Math.sin(scaled);
                out[row + half + k] = Math.cos(scaled);
            }
        }
        return out;
    },
    /** Depthwise FSMN memory over the value tensor. */
    fsmnMemory = (options: {
        value: Float32Array;
        weight: Float32Array;
        seq: number;
        dims: number;
        kernel: number;
    }): Float32Array => {
        const
            { value, weight, seq, dims, kernel } = options,
            pad = (kernel - 1) >> 1,
            out = new Float32Array(seq * dims);
        for (let c = 0; c < dims; c += 1) {
            const taps = c * kernel;
            for (let t = 0; t < seq; t += 1) {
                let sum = 0;
                for (let k = 0; k < kernel; k += 1) {
                    const source = t - pad + k;
                    if (source >= 0 && source < seq) {
                        sum += weight[taps + k] * value[source * dims + c];
                    }
                }
                out[t * dims + c] = sum + value[t * dims + c];
            }
        }
        return out;
    },
    /** SAN-M attention branch, the sum of SDPA and the FSMN memory. */
    sanmAttention = (options: {
        input: Float32Array;
        weights: SenseVoiceEncodeOptions["weights"];
        base: string;
        config: SenseVoiceConfig;
        seq: number;
        inDim: number;
    }): Float32Array => {
        const
            { input, weights, base, config, seq, inDim } = options,
            dim = config.dModel,
            heads = config.heads,
            headDim = config.headDim,
            qkv = linear({
                input,
                weight: weights.f32(`${base}.attn.qkv.weight`),
                bias: weights.f32(`${base}.attn.qkv.bias`),
                rows: seq,
                inDim,
                outDim: 3 * dim,
            }),
            q = new Float32Array(seq * dim),
            k = new Float32Array(seq * dim),
            v = new Float32Array(seq * dim);
        for (let t = 0; t < seq; t += 1) {
            const row = t * 3 * dim;
            for (let i = 0; i < dim; i += 1) {
                q[t * dim + i] = qkv[row + i];
                k[t * dim + i] = qkv[row + dim + i];
                v[t * dim + i] = qkv[row + 2 * dim + i];
            }
        }
        const
            fsmn = fsmnMemory({
                value: v,
                weight: weights.f32(`${base}.attn.fsmn.weight`),
                seq,
                dims: dim,
                kernel: config.kernelSize,
            }),
            ctx = attention({
                q,
                k,
                v,
                seqQ: seq,
                seqK: seq,
                heads,
                headDim,
                scale: 1 / Math.sqrt(headDim),
            }),
            projected = linear({
                input: ctx,
                weight: weights.f32(`${base}.attn.out.weight`),
                bias: weights.f32(`${base}.attn.out.bias`),
                rows: seq,
                inDim: dim,
                outDim: dim,
            });
        addInPlace({ a: projected, b: fsmn });
        return projected;
    },
    /** SAN-M feed forward branch. */
    sanmFfn = (options: {
        input: Float32Array;
        weights: SenseVoiceEncodeOptions["weights"];
        base: string;
        config: SenseVoiceConfig;
        seq: number;
    }): Float32Array => {
        const
            { input, weights, base, config, seq } = options,
            hidden = linear({
                input,
                weight: weights.f32(`${base}.ffn.fc1.weight`),
                bias: weights.f32(`${base}.ffn.fc1.bias`),
                rows: seq,
                inDim: config.dModel,
                outDim: config.ffnDim,
            });
        reluInPlace({ values: hidden });
        return linear({
            input: hidden,
            weight: weights.f32(`${base}.ffn.fc2.weight`),
            bias: weights.f32(`${base}.ffn.fc2.bias`),
            rows: seq,
            inDim: config.ffnDim,
            outDim: config.dModel,
        });
    },
    /** Residual SAN-M block, both tiers use this shape. */
    sanmBlock = (options: {
        input: Float32Array;
        weights: SenseVoiceEncodeOptions["weights"];
        base: string;
        config: SenseVoiceConfig;
        seq: number;
    }): Float32Array => {
        const
            { input, weights, base, config, seq } = options,
            dim = config.dModel,
            normAttn = layerNorm({
                input,
                weight: weights.f32(`${base}.norm_attn.weight`),
                bias: weights.f32(`${base}.norm_attn.bias`),
                rows: seq,
                dims: dim,
                eps: config.eps,
            }),
            attn = sanmAttention({
                input: normAttn,
                weights,
                base,
                config,
                seq,
                inDim: dim,
            }),
            first = addArrays({ a: input, b: attn }),
            normFfn = layerNorm({
                input: first,
                weight: weights.f32(`${base}.norm_ffn.weight`),
                bias: weights.f32(`${base}.norm_ffn.bias`),
                rows: seq,
                dims: dim,
                eps: config.eps,
            }),
            ffn = sanmFfn({ input: normFfn, weights, base, config, seq });
        return addArrays({ a: first, b: ffn });
    },
    /** First SAN-M block, a dInput to dModel projection. */
    sanmProjection = (options: {
        input: Float32Array;
        weights: SenseVoiceEncodeOptions["weights"];
        base: string;
        config: SenseVoiceConfig;
        seq: number;
    }): Float32Array => {
        const
            { input, weights, base, config, seq } = options,
            normAttn = layerNorm({
                input,
                weight: weights.f32(`${base}.norm_attn.weight`),
                bias: weights.f32(`${base}.norm_attn.bias`),
                rows: seq,
                dims: config.dInput,
                eps: config.eps,
            }),
            attn = sanmAttention({
                input: normAttn,
                weights,
                base,
                config,
                seq,
                inDim: config.dInput,
            }),
            normFfn = layerNorm({
                input: attn,
                weight: weights.f32(`${base}.norm_ffn.weight`),
                bias: weights.f32(`${base}.norm_ffn.bias`),
                rows: seq,
                dims: config.dModel,
                eps: config.eps,
            }),
            ffn = sanmFfn({ input: normFfn, weights, base, config, seq });
        addInPlace({ a: attn, b: ffn });
        return attn;
    },
    /** Encode prefixes plus features into hidden states. */
    encodeSenseVoice = (
        options: SenseVoiceEncodeOptions,
    ): SenseVoiceEncodeResult => {
        const
            { config, weights, features, frames, prefix } = options,
            dim = config.dModel,
            dInput = config.dInput,
            seq = frames + 4,
            embed = weights.f32("enc.embed.weight"),
            input = new Float32Array(seq * dInput);
        for (let p = 0; p < prefix.length; p += 1) {
            const row = prefix[p] * dInput;
            for (let i = 0; i < dInput; i += 1) {
                input[p * dInput + i] = embed[row + i];
            }
        }
        input.set(features.subarray(0, frames * dInput), prefix.length * dInput);
        const scale = Math.sqrt(dim);
        for (let i = 0; i < input.length; i += 1) {
            input[i] *= scale;
        }
        const pe = sinusoidalPe({ depth: dInput, frames: seq });
        addInPlace({ a: input, b: pe });
        let hidden = sanmProjection({
            input,
            weights,
            base: "enc.encoders0.0",
            config,
            seq,
        });
        for (let i = 0; i < config.mainBlocks - 1; i += 1) {
            hidden = sanmBlock({
                input: hidden,
                weights,
                base: `enc.encoders.${i}`,
                config,
                seq,
            });
        }
        hidden = layerNorm({
            input: hidden,
            weight: weights.f32("enc.after_norm.weight"),
            bias: weights.f32("enc.after_norm.bias"),
            rows: seq,
            dims: dim,
            eps: config.eps,
        });
        for (let i = 0; i < config.tpBlocks; i += 1) {
            hidden = sanmBlock({
                input: hidden,
                weights,
                base: `enc.tp_encoders.${i}`,
                config,
                seq,
            });
        }
        hidden = layerNorm({
            input: hidden,
            weight: weights.f32("enc.tp_norm.weight"),
            bias: weights.f32("enc.tp_norm.bias"),
            rows: seq,
            dims: dim,
            eps: config.eps,
        });
        return { hidden, seq };
    },
    /** Run the CTC head over hidden states. */
    senseVoiceLogits = (
        options: SenseVoiceLogitsOptions,
    ): Float32Array => {
        const { config, weights, hidden, seq } = options;
        return linear({
            input: hidden,
            weight: weights.f32("ctc.head.weight"),
            bias: weights.f32("ctc.head.bias"),
            rows: seq,
            inDim: config.dModel,
            outDim: config.vocabSize,
        });
    };

export { encodeSenseVoice, senseVoiceLogits };
