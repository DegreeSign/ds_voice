// Whisper GGUF weight reading.
import { product } from "../../tensor/binary";
import {
    fullCacheBytes,
    ggmlTypeName,
    readGgufTensor,
} from "../../formats/ggufTensor";
import { releaseTensor } from "../../tensor/wasm";
import type {
    CreateGgufWeightsOptions,
    GgufTensorInfo,
    TensorInfo,
    TensorStore,
} from "../../tensor/types";
const
    /** Hugging Face projection name of one ggml attention part. */
    projection = (part: string): string | null => {
        if (part === "q" || part === "k" || part === "v") {
            return `${part}_proj`;
        }
        if (part === "out") {
            return "out_proj";
        }
        return null;
    },
    /** Map one ggml tensor name to its Hugging Face name. */
    mapName = (name: string): string | null => {
        switch (name) {
            case "enc.conv.0.weight":
                return "model.encoder.conv1.weight";
            case "enc.conv.0.bias":
                return "model.encoder.conv1.bias";
            case "enc.conv.1.weight":
                return "model.encoder.conv2.weight";
            case "enc.conv.1.bias":
                return "model.encoder.conv2.bias";
            case "enc.pos_emb.weight":
                return "model.encoder.embed_positions.weight";
            case "enc.final_norm.weight":
                return "model.encoder.layer_norm.weight";
            case "enc.final_norm.bias":
                return "model.encoder.layer_norm.bias";
            case "dec.token_embd.weight":
                return "model.decoder.embed_tokens.weight";
            case "dec.pos_emb.weight":
                return "model.decoder.embed_positions.weight";
            case "dec.final_norm.weight":
                return "model.decoder.layer_norm.weight";
            case "dec.final_norm.bias":
                return "model.decoder.layer_norm.bias";
            default:
                break;
        }
        const block =
            /^(enc|dec)\.blocks\.(\d+)\.(.+)\.(weight|bias)$/.exec(name);
        if (!block) {
            return null;
        }
        const
            [, family, layer, body, suffix] = block,
            prefix = family === "enc"
                ? `model.encoder.layers.${layer}`
                : `model.decoder.layers.${layer}`,
            dot = body.indexOf("."),
            head = dot >= 0 ? body.slice(0, dot) : body,
            rest = dot >= 0 ? body.slice(dot + 1) : "",
            projected = projection(rest);
        if (family === "enc") {
            if (head === "norm_attn") {
                return `${prefix}.self_attn_layer_norm.${suffix}`;
            }
            if (head === "norm_ffn") {
                return `${prefix}.final_layer_norm.${suffix}`;
            }
            if (head === "attn" && projected) {
                return `${prefix}.self_attn.${projected}.${suffix}`;
            }
            if (head === "ffn" && (rest === "fc1" || rest === "fc2")) {
                return `${prefix}.${rest}.${suffix}`;
            }
            return null;
        }
        if (head === "norm_self") {
            return `${prefix}.self_attn_layer_norm.${suffix}`;
        }
        if (head === "norm_cross") {
            return `${prefix}.encoder_attn_layer_norm.${suffix}`;
        }
        if (head === "norm_ffn") {
            return `${prefix}.final_layer_norm.${suffix}`;
        }
        if (head === "self_attn" && projected) {
            return `${prefix}.self_attn.${projected}.${suffix}`;
        }
        if (head === "cross_attn" && projected) {
            return `${prefix}.encoder_attn.${projected}.${suffix}`;
        }
        if (head === "ffn" && (rest === "fc1" || rest === "fc2")) {
            return `${prefix}.${rest}.${suffix}`;
        }
        return null;
    },
    /** Build a tensor store over a Whisper GGUF file. */
    createWhisperWeights = (
        options: CreateGgufWeightsOptions,
    ): TensorStore => {
        const
            { file } = options,
            infos = new Map<string, TensorInfo>(),
            sources = new Map<string, GgufTensorInfo>(),
            cache = new Map<string, Float32Array>();
        let totalBytes = 0;
        for (const [name, entry] of file.tensors) {
            const mapped = mapName(name);
            if (!mapped) {
                continue;
            }
            infos.set(mapped, {
                name: mapped,
                dtype: ggmlTypeName(entry.type),
                shape: [...entry.dims].reverse(),
                dataOffset: entry.offset,
                numel: product(entry.dims),
            });
            sources.set(mapped, entry);
            totalBytes += product(entry.dims) * 4;
        }
        const
            limit = options.cacheBytes === undefined
                ? fullCacheBytes(totalBytes)
                : Math.max(0, options.cacheBytes),
            info = (name: string): TensorInfo => {
                const entry = infos.get(name);
                if (!entry) {
                    throw new Error(`missing tensor ${name}`);
                }
                return entry;
            },
            remember = (name: string, value: Float32Array): void => {
                if (limit <= 0) {
                    return;
                }
                // drop any older copy first
                cache.delete(name);
                let used = value.byteLength;
                for (const item of cache.values()) {
                    used += item.byteLength;
                }
                while (used > limit && cache.size > 0) {
                    const oldest = cache.keys().next().value;
                    if (oldest === undefined) {
                        break;
                    }
                    const dropped = cache.get(oldest);
                    cache.delete(oldest);
                    used -= dropped?.byteLength ?? 0;
                    if (dropped) {
                        // free the arena copy too
                        releaseTensor(dropped);
                    }
                }
                cache.set(name, value);
            },
            get = (name: string): Float32Array => {
                const hit = cache.get(name);
                if (hit) {
                    cache.delete(name);
                    cache.set(name, hit);
                    return hit;
                }
                const source = sources.get(name);
                if (!source) {
                    throw new Error(`missing tensor ${name}`);
                }
                const value = readGgufTensor({ file, info: source });
                remember(name, value);
                return value;
            };
        return {
            count: infos.size,
            names: () => [...infos.keys()],
            has: (name: string) => infos.has(name),
            info,
            get,
        };
    };

export { createWhisperWeights, mapName };
