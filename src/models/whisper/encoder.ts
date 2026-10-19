// Whisper encoder graph.
import {
    addArrays,
    attention,
    geluInPlace,
    layerNorm,
    linear,
} from "../../tensor/kernels";
import { wasmConv1d } from "../../tensor/wasm";
import type { EncodeOptions, EncodeResult } from "../../tensor/types";
const
    /** One dimensional convolution with padding. */
    conv1dJs = (options: {
        input: Float32Array;
        weight: Float32Array;
        bias: Float32Array;
        inChannels: number;
        outChannels: number;
        inLength: number;
        kernel: number;
        stride: number;
        padding: number;
    }): { data: Float32Array; length: number } => {
        const
            { input, weight, bias, inChannels, outChannels, inLength, kernel } =
                options,
            { stride, padding } = options,
            outLength = Math.floor((inLength + 2 * padding - kernel) / stride) + 1,
            out = new Float32Array(outChannels * outLength);
        for (let oc = 0; oc < outChannels; oc += 1) {
            const outRow = oc * outLength;
            for (let t = 0; t < outLength; t += 1) {
                let sum = bias[oc];
                const base = t * stride - padding;
                for (let ic = 0; ic < inChannels; ic += 1) {
                    const
                        inRow = ic * inLength,
                        wRow = (oc * inChannels + ic) * kernel;
                    for (let k = 0; k < kernel; k += 1) {
                        const index = base + k;
                        if (index < 0 || index >= inLength) {
                            continue;
                        }
                        sum += weight[wRow + k] * input[inRow + index];
                    }
                }
                out[outRow + t] = sum;
            }
        }
        return { data: out, length: outLength };
    },
    /** 1d convolution, SIMD fast path with a fallback. */
    conv1d = (options: {
        input: Float32Array;
        weight: Float32Array;
        bias: Float32Array;
        inChannels: number;
        outChannels: number;
        inLength: number;
        kernel: number;
        stride: number;
        padding: number;
    }): { data: Float32Array; length: number } => {
        return wasmConv1d(options) ?? conv1dJs(options);
    },
    /** Run the encoder over one mel chunk. */
    encode = (options: EncodeOptions): EncodeResult => {
        const
            { model, mel } = options,
            { config, tensors } = model,
            { dModel, encoderLayers, headDim, encoderHeads, encoderFfnDim } = config,
            get = (name: string) => tensors.get(name),
            conv1 = conv1d({
                input: mel,
                weight: get("model.encoder.conv1.weight"),
                bias: get("model.encoder.conv1.bias"),
                inChannels: config.numMelBins,
                outChannels: dModel,
                inLength: mel.length / config.numMelBins,
                kernel: 3,
                stride: 1,
                padding: 1,
            });
        geluInPlace({ values: conv1.data, length: conv1.data.length });
        const conv2 = conv1d({
            input: conv1.data,
            weight: get("model.encoder.conv2.weight"),
            bias: get("model.encoder.conv2.bias"),
            inChannels: dModel,
            outChannels: dModel,
            inLength: conv1.length,
            kernel: 3,
            stride: 2,
            padding: 1,
        });
        geluInPlace({ values: conv2.data, length: conv2.data.length });
        const
            seq = conv2.length,
            positions = get("model.encoder.embed_positions.weight");
        let hidden: Float32Array = new Float32Array(seq * dModel);
        for (let t = 0; t < seq; t += 1) {
            for (let d = 0; d < dModel; d += 1) {
                hidden[t * dModel + d] =
                    conv2.data[d * seq + t] + positions[t * dModel + d];
            }
        }
        const perLayer: Float32Array[] = [];
        for (let layer = 0; layer < encoderLayers; layer += 1) {
            hidden = encoderLayer({
                hidden,
                seq,
                layer,
                dModel,
                headDim,
                heads: encoderHeads,
                ffnDim: encoderFfnDim,
                get,
            });
            if (options.collect) {
                perLayer.push(hidden.slice());
            }
        }
        const
            final = get("model.encoder.layer_norm.weight"),
            finalBias = get("model.encoder.layer_norm.bias"),
            output = layerNorm({
                input: hidden,
                weight: final,
                bias: finalBias,
                rows: seq,
                dims: dModel,
            });
        return { output, seq, perLayer };
    },
    /** Run one pre-norm encoder layer. */
    encoderLayer = (options: {
        hidden: Float32Array;
        seq: number;
        layer: number;
        dModel: number;
        headDim: number;
        heads: number;
        ffnDim: number;
        get: (name: string) => Float32Array;
    }): Float32Array => {
        const
            { hidden, seq, layer, dModel, headDim, heads, ffnDim, get } = options,
            prefix = `model.encoder.layers.${layer}`,
            normed = layerNorm({
                input: hidden,
                weight: get(`${prefix}.self_attn_layer_norm.weight`),
                bias: get(`${prefix}.self_attn_layer_norm.bias`),
                rows: seq,
                dims: dModel,
            }),
            q = linear({
                input: normed,
                weight: get(`${prefix}.self_attn.q_proj.weight`),
                bias: get(`${prefix}.self_attn.q_proj.bias`),
                rows: seq,
                inDim: dModel,
                outDim: dModel,
            }),
            k = linear({
                input: normed,
                weight: get(`${prefix}.self_attn.k_proj.weight`),
                rows: seq,
                inDim: dModel,
                outDim: dModel,
            }),
            v = linear({
                input: normed,
                weight: get(`${prefix}.self_attn.v_proj.weight`),
                bias: get(`${prefix}.self_attn.v_proj.bias`),
                rows: seq,
                inDim: dModel,
                outDim: dModel,
            }),
            attended = attention({
                q,
                k,
                v,
                seqQ: seq,
                seqK: seq,
                headDim,
                heads,
            }),
            projected = linear({
                input: attended,
                weight: get(`${prefix}.self_attn.out_proj.weight`),
                bias: get(`${prefix}.self_attn.out_proj.bias`),
                rows: seq,
                inDim: dModel,
                outDim: dModel,
            }),
            residual = addArrays({ a: hidden, b: projected }),
            normed2 = layerNorm({
                input: residual,
                weight: get(`${prefix}.final_layer_norm.weight`),
                bias: get(`${prefix}.final_layer_norm.bias`),
                rows: seq,
                dims: dModel,
            }),
            fc1 = linear({
                input: normed2,
                weight: get(`${prefix}.fc1.weight`),
                bias: get(`${prefix}.fc1.bias`),
                rows: seq,
                inDim: dModel,
                outDim: ffnDim,
            });
        geluInPlace({ values: fc1, length: fc1.length });
        const fc2 = linear({
            input: fc1,
            weight: get(`${prefix}.fc2.weight`),
            bias: get(`${prefix}.fc2.bias`),
            rows: seq,
            inDim: ffnDim,
            outDim: dModel,
        });
        return addArrays({ a: residual, b: fc2 });
    };

export { encode };
