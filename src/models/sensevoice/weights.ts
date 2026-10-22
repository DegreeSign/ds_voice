// SenseVoice GGUF weight reading.
import {
    GGUF_CACHE_BYTES,
    readGgufTensor,
} from "../../formats/ggufTensor";
import type {
    CreateGgufWeightsOptions,
    SenseVoiceWeights,
} from "../../tensor/types";
const
    /** Build a weight store over a parsed GGUF file. */
    createGgufWeights = (
        options: CreateGgufWeightsOptions,
    ): SenseVoiceWeights => {
        const
            { file } = options,
            { tensors } = file,
            cache = new Map<string, Float32Array>(),
            limit = Math.max(0, options.cacheBytes ?? GGUF_CACHE_BYTES),
            info = (name: string) => {
                const entry = tensors.get(name);
                if (!entry) {
                    throw new Error(`missing tensor ${name}`);
                }
                return entry;
            },
            shape = (name: string): number[] => [...info(name).dims].reverse(),
            remember = (name: string, value: Float32Array): void => {
                if (limit <= 0 || value.byteLength > limit) {
                    return;
                }
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
                }
                cache.set(name, value);
            },
            f32 = (name: string): Float32Array => {
                const hit = cache.get(name);
                if (hit) {
                    cache.delete(name);
                    cache.set(name, hit);
                    return hit;
                }
                const value = readGgufTensor({ file, info: info(name) });
                remember(name, value);
                return value;
            };
        return {
            names: () => [...tensors.keys()],
            has: (name: string) => tensors.has(name),
            shape,
            f32,
        };
    };

export { GGUF_CACHE_BYTES, createGgufWeights };
