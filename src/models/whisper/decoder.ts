// Whisper decoder graph.
import {
    addArrays,
    argmax,
    crossAttentionOne,
    geluInPlace,
    layerNorm,
    linear,
    softmaxRow,
} from "../../tensor/kernels";
import { Language } from "../../tensor/types";
import type {
    CapturePair,
    DecoderState,
    LoadedModel,
    StepResult,
} from "../../tensor/types";
const
    /** Create decoder state and precompute cross keys and values. */
    createDecoderState = (options: {
        model: LoadedModel;
        encoderOutput: Float32Array;
        encoderSeq: number;
    }): DecoderState => {
        const
            { model, encoderOutput, encoderSeq } = options,
            { config, tensors } = model,
            { dModel, decoderLayers, headDim, decoderHeads, decoderFfnDim } = config,
            capacity = config.maxTargetPositions,
            selfKeys: Float32Array[] = [],
            selfValues: Float32Array[] = [],
            crossKeys: Float32Array[] = [],
            crossValues: Float32Array[] = [];
        for (let layer = 0; layer < decoderLayers; layer += 1) {
            const prefix = `model.decoder.layers.${layer}`;
            selfKeys.push(new Float32Array(capacity * dModel));
            selfValues.push(new Float32Array(capacity * dModel));
            crossKeys.push(
                linear({
                    input: encoderOutput,
                    weight: tensors.get(`${prefix}.encoder_attn.k_proj.weight`),
                    rows: encoderSeq,
                    inDim: dModel,
                    outDim: dModel,
                }),
            );
            crossValues.push(
                linear({
                    input: encoderOutput,
                    weight: tensors.get(`${prefix}.encoder_attn.v_proj.weight`),
                    bias: tensors.get(`${prefix}.encoder_attn.v_proj.bias`),
                    rows: encoderSeq,
                    inDim: dModel,
                    outDim: dModel,
                }),
            );
        }
        return {
            model,
            seq: encoderSeq,
            dModel,
            layers: decoderLayers,
            heads: decoderHeads,
            headDim,
            ffnDim: decoderFfnDim,
            selfKeys,
            selfValues,
            crossKeys,
            crossValues,
            position: 0,
        };
    },
    /** Normalized attention weights for one head. */
    headWeights = (options: {
        q: Float32Array;
        k: Float32Array;
        head: number;
        heads: number;
        headDim: number;
        seqK: number;
        scale: number;
    }): Float32Array => {
        const
            { q, k, head, heads, headDim, seqK, scale } = options,
            dim = heads * headDim,
            offset = head * headDim,
            weights = new Float32Array(seqK);
        for (let j = 0; j < seqK; j += 1) {
            const kRow = j * dim + offset;
            let sum = 0;
            for (let d = 0; d < headDim; d += 1) {
                sum += q[offset + d] * k[kRow + d];
            }
            weights[j] = sum * scale;
        }
        softmaxRow({ values: weights, offset: 0, length: seqK });
        return weights;
    },
    /** Run one decoder step for a single token. */
    decoderStep = (options: {
        state: DecoderState;
        tokenId: number;
        capture?: CapturePair[];
    }): StepResult => {
        const
            { state, tokenId } = options,
            { model, dModel } = state,
            get = (name: string) => model.tensors.get(name),
            embed = get("model.decoder.embed_tokens.weight"),
            positions = get("model.decoder.embed_positions.weight"),
            position = state.position;
        let hidden: Float32Array = new Float32Array(dModel);
        const
            embedRow = tokenId * dModel,
            posRow = position * dModel;
        for (let d = 0; d < dModel; d += 1) {
            hidden[d] = embed[embedRow + d] + positions[posRow + d];
        }
        const
            capturePairs = options.capture ?? [],
            captures: Float32Array[] = capturePairs.map(
                () => new Float32Array(state.seq),
            );
        for (let layer = 0; layer < state.layers; layer += 1) {
            const
                prefix = `model.decoder.layers.${layer}`,
                normed = layerNorm({
                    input: hidden,
                    weight: get(`${prefix}.self_attn_layer_norm.weight`),
                    bias: get(`${prefix}.self_attn_layer_norm.bias`),
                    rows: 1,
                    dims: dModel,
                }),
                q = linear({
                    input: normed,
                    weight: get(`${prefix}.self_attn.q_proj.weight`),
                    bias: get(`${prefix}.self_attn.q_proj.bias`),
                    rows: 1,
                    inDim: dModel,
                    outDim: dModel,
                }),
                k = linear({
                    input: normed,
                    weight: get(`${prefix}.self_attn.k_proj.weight`),
                    rows: 1,
                    inDim: dModel,
                    outDim: dModel,
                }),
                v = linear({
                    input: normed,
                    weight: get(`${prefix}.self_attn.v_proj.weight`),
                    bias: get(`${prefix}.self_attn.v_proj.bias`),
                    rows: 1,
                    inDim: dModel,
                    outDim: dModel,
                });
            state.selfKeys[layer].set(k, position * dModel);
            state.selfValues[layer].set(v, position * dModel);
            const
                keyCount = position + 1,
                attended = crossAttentionOne({
                    q,
                    k: state.selfKeys[layer],
                    v: state.selfValues[layer],
                    heads: state.heads,
                    headDim: state.headDim,
                    seqK: keyCount,
                }),
                selfOut = linear({
                    input: attended,
                    weight: get(`${prefix}.self_attn.out_proj.weight`),
                    bias: get(`${prefix}.self_attn.out_proj.bias`),
                    rows: 1,
                    inDim: dModel,
                    outDim: dModel,
                });
            hidden = addArrays({ a: hidden, b: selfOut });
            const
                normed2 = layerNorm({
                    input: hidden,
                    weight: get(`${prefix}.encoder_attn_layer_norm.weight`),
                    bias: get(`${prefix}.encoder_attn_layer_norm.bias`),
                    rows: 1,
                    dims: dModel,
                }),
                cq = linear({
                    input: normed2,
                    weight: get(`${prefix}.encoder_attn.q_proj.weight`),
                    bias: get(`${prefix}.encoder_attn.q_proj.bias`),
                    rows: 1,
                    inDim: dModel,
                    outDim: dModel,
                });
            for (let index = 0; index < capturePairs.length; index += 1) {
                const pair = capturePairs[index];
                if (pair.layer !== layer) {
                    continue;
                }
                captures[index] = headWeights({
                    q: cq,
                    k: state.crossKeys[layer],
                    head: pair.head,
                    heads: state.heads,
                    headDim: state.headDim,
                    seqK: state.seq,
                    scale: 1 / Math.sqrt(state.headDim),
                });
            }
            const
                crossOut = crossAttentionOne({
                    q: cq,
                    k: state.crossKeys[layer],
                    v: state.crossValues[layer],
                    heads: state.heads,
                    headDim: state.headDim,
                    seqK: state.seq,
                }),
                crossProjected = linear({
                    input: crossOut,
                    weight: get(`${prefix}.encoder_attn.out_proj.weight`),
                    bias: get(`${prefix}.encoder_attn.out_proj.bias`),
                    rows: 1,
                    inDim: dModel,
                    outDim: dModel,
                });
            hidden = addArrays({ a: hidden, b: crossProjected });
            const
                normed3 = layerNorm({
                    input: hidden,
                    weight: get(`${prefix}.final_layer_norm.weight`),
                    bias: get(`${prefix}.final_layer_norm.bias`),
                    rows: 1,
                    dims: dModel,
                }),
                fc1 = linear({
                    input: normed3,
                    weight: get(`${prefix}.fc1.weight`),
                    bias: get(`${prefix}.fc1.bias`),
                    rows: 1,
                    inDim: dModel,
                    outDim: state.ffnDim,
                });
            geluInPlace({ values: fc1, length: fc1.length });
            const fc2 = linear({
                input: fc1,
                weight: get(`${prefix}.fc2.weight`),
                bias: get(`${prefix}.fc2.bias`),
                rows: 1,
                inDim: state.ffnDim,
                outDim: dModel,
            });
            hidden = addArrays({ a: hidden, b: fc2 });
        }
        state.position = position + 1;
        const
            final = layerNorm({
                input: hidden,
                weight: get("model.decoder.layer_norm.weight"),
                bias: get("model.decoder.layer_norm.bias"),
                rows: 1,
                dims: dModel,
            }),
            logits = linear({
                input: final,
                weight: embed,
                rows: 1,
                inDim: dModel,
                outDim: model.config.vocabSize,
            });
        return { logits, captures };
    },
    /** Detect the language token for one encoder output. */
    detectLanguage = (options: {
        state: DecoderState;
        constants: LoadedModel["constants"];
    }): { language: Language; token: number } => {
        const
            { state, constants } = options,
            result = decoderStep({
                state,
                tokenId: constants.startOfTranscript,
            }),
            ids = [...constants.languageIds.values()],
            start = Math.min(...ids),
            end = Math.max(...ids),
            bestId = argmax({ values: result.logits, start, end: end + 1 }),
            language = (constants.idToLanguage.get(String(bestId)) as
                Language | undefined) ?? Language.en;
        return { language, token: bestId };
    };

export { createDecoderState, decoderStep, detectLanguage };
