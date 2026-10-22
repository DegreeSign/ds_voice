// GGUF tensor dequantization.
import { f16ToF32, product, readF16 } from "../tensor/binary";
import { reserveFloats } from "../tensor/wasm";
import type { ReadGgufTensorOptions } from "../tensor/types";
const
    /** Default F32 cache budget in bytes. */
    GGUF_CACHE_BYTES = 128 * 1024 * 1024,
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
    /** Read one GGUF tensor as f32, row major. */
    readGgufTensor = (options: ReadGgufTensorOptions): Float32Array => {
        const
            { file, info } = options,
            { buffer, dataOffset } = file,
            numel = product(info.dims),
            base = dataOffset + info.offset;
        if (info.type === GGML_F32) {
            return f32View({ bytes: buffer, base, count: numel });
        }
        if (info.type === GGML_F16) {
            return readF16({ bytes: buffer, offset: base, count: numel });
        }
        if (info.type === GGML_Q8_0) {
            const
                cols = info.dims[0],
                rows = numel / cols;
            if (cols % Q8_BLOCK !== 0) {
                throw new Error(`q8 row width ${cols}`);
            }
            return dequantQ8({ bytes: buffer, base, rows, cols });
        }
        throw new Error(`unsupported ggml type ${info.type} for ${info.name}`);
    };

/** Human readable name of a supported ggml type. */
const ggmlTypeName = (type: number): string => {
    if (type === GGML_F32) {
        return "F32";
    }
    if (type === GGML_F16) {
        return "F16";
    }
    if (type === GGML_Q8_0) {
        return "Q8_0";
    }
    return `ggml:${type}`;
};

export {
    GGUF_CACHE_BYTES,
    GGML_F32,
    GGML_F16,
    GGML_Q8_0,
    readGgufTensor,
    ggmlTypeName,
};
