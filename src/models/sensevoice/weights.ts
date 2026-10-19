// SenseVoice GGUF weight reading.
import { f16ToF32, readF16 } from "../../tensor/binary";
import { reserveFloats } from "../../tensor/wasm";
import type { GgufFile, SenseVoiceWeights } from "../../tensor/types";
const
    /** ggml type code for f32. */
    GGML_F32 = 0,
    /** ggml type code for f16. */
    GGML_F16 = 1,
    /** ggml type code for Q8_0. */
    GGML_Q8_0 = 8,
    /** Elements packed into one Q8_0 block. */
    Q8_BLOCK = 32,
    /** Bytes in one Q8_0 block, one f16 scale plus 32 int8 values. */
    Q8_BYTES = 34,
    /** Read an f32 view, copying only when misaligned. */
    f32View = (options: {
        bytes: Uint8Array;
        base: number;
        count: number;
    }): Float32Array => {
        const
            { bytes, base, count } = options,
            absolute = bytes.byteOffset + base;
        if (absolute % 4 === 0) {
            return new Float32Array(bytes.buffer, absolute, count);
        }
        const
            view = new DataView(bytes.buffer, absolute, count * 4),
            out = new Float32Array(count);
        for (let i = 0; i < count; i += 1) {
            out[i] = view.getFloat32(i * 4, true);
        }
        return out;
    },
    /** Dequantize one Q8_0 tensor, row major [rows, cols]. */
    dequantQ8 = (options: {
        bytes: Uint8Array;
        base: number;
        rows: number;
        cols: number;
    }): Float32Array => {
        const
            { bytes, base, rows, cols } = options,
            blocks = cols / Q8_BLOCK,
            view = new DataView(
                bytes.buffer,
                bytes.byteOffset + base,
                rows * blocks * Q8_BYTES,
            ),
            out = reserveFloats(rows * cols) ??
                new Float32Array(rows * cols);
        let at = 0;
        for (let r = 0; r < rows; r += 1) {
            const row = r * cols;
            for (let b = 0; b < blocks; b += 1) {
                const scale = f16ToF32(view.getUint16(at, true));
                at += 2;
                const outBase = row + b * Q8_BLOCK;
                for (let i = 0; i < Q8_BLOCK; i += 1) {
                    out[outBase + i] = scale * view.getInt8(at + i);
                }
                at += Q8_BLOCK;
            }
        }
        return out;
    },
    /** Build a weight store over a parsed GGUF file. */
    createGgufWeights = (options: {
        file: GgufFile;
    }): SenseVoiceWeights => {
        const
            { file } = options,
            { buffer, dataOffset, tensors } = file,
            cache = new Map<string, Float32Array>(),
            info = (name: string) => {
                const entry = tensors.get(name);
                if (!entry) {
                    throw new Error(`missing tensor ${name}`);
                }
                return entry;
            },
            shape = (name: string): number[] => [...info(name).dims].reverse(),
            f32 = (name: string): Float32Array => {
                const hit = cache.get(name);
                if (hit) {
                    return hit;
                }
                const entry = info(name);
                let numel = 1;
                for (const dim of entry.dims) {
                    numel *= dim;
                }
                const base = dataOffset + entry.offset;
                let value: Float32Array;
                if (entry.type === GGML_F32) {
                    value = f32View({ bytes: buffer, base, count: numel });
                } else if (entry.type === GGML_F16) {
                    value = readF16({ bytes: buffer, offset: base, count: numel });
                } else if (entry.type === GGML_Q8_0) {
                    const cols = entry.dims[0];
                    value = dequantQ8({ bytes: buffer, base, rows: numel / cols, cols });
                } else {
                    throw new Error(`unsupported ggml type ${entry.type} for ${name}`);
                }
                cache.set(name, value);
                return value;
            };
        return {
            names: () => [...tensors.keys()],
            has: (name: string) => tensors.has(name),
            shape,
            f32,
        };
    };

export { createGgufWeights };
