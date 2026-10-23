// GGUF tensor dequantization.
import { f16ToF32, product, readF16 } from "../tensor/binary";
import { reserveFloats } from "../tensor/wasm";
import type { ReadGgufTensorOptions } from "../tensor/types";
const
    /** Default F32 cache budget in bytes. */
    GGUF_CACHE_BYTES = 128 * 1024 * 1024,
    /** Largest automatic cache budget in bytes. */
    GGUF_CACHE_CAP_BYTES = 1024 * 1024 * 1024,
    /** Budget that holds every tensor of a checkpoint once. */
    fullCacheBytes = (total: number): number => {
        if (total <= 0) {
            return GGUF_CACHE_BYTES;
        }
        return Math.min(
            GGUF_CACHE_CAP_BYTES,
            Math.max(GGUF_CACHE_BYTES, total),
        );
    },
    /** ggml type code for f32. */
    GGML_F32 = 0,
    /** ggml type code for f16. */
    GGML_F16 = 1,
    /** ggml type code for Q8_0. */
    GGML_Q8_0 = 8,
    /** ggml type code for Q4_K. */
    GGML_Q4_K = 12,
    /** ggml type code for Q6_K. */
    GGML_Q6_K = 14,
    /** Elements packed into one K-quant super block. */
    QK_K = 256,
    /** Elements packed into one Q8_0 block. */
    Q8_BLOCK = 32,
    /** Bytes in one Q8_0 block, one f16 scale plus 32 int8 values. */
    Q8_BYTES = 34,
    /** Bytes in one Q4_K super block. */
    Q4_K_BYTES = 144,
    /** Bytes in one Q6_K super block. */
    Q6_K_BYTES = 210,
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
    /** Read the scale and min pair of one Q4_K sub block. */
    scaleMinK4 = (options: {
        bytes: Uint8Array;
        base: number;
        index: number;
    }): [number, number] => {
        const
            { bytes, base, index } = options;
        if (index < 4) {
            return [bytes[base + index] & 63, bytes[base + index + 4] & 63];
        }
        const packed = bytes[base + index + 4];
        return [
            (packed & 15) | ((bytes[base + index - 4] >> 6) << 4),
            (packed >> 4) | ((bytes[base + index] >> 6) << 4),
        ];
    },
    /** Dequantize one Q4_K tensor, row major [rows, cols]. */
    dequantQ4K = (options: {
        bytes: Uint8Array;
        base: number;
        rows: number;
        cols: number;
    }): Float32Array => {
        const
            { bytes, base, rows, cols } = options,
            blocks = cols / QK_K,
            size = rows * blocks * Q4_K_BYTES,
            data = bytes.subarray(base, base + size),
            view = new DataView(data.buffer, data.byteOffset, size),
            out = reserveFloats(rows * cols) ??
                new Float32Array(rows * cols);
        let at = 0;
        for (let r = 0; r < rows; r += 1) {
            const row = r * cols;
            for (let b = 0; b < blocks; b += 1) {
                const
                    d = f16ToF32(view.getUint16(at, true)),
                    min = f16ToF32(view.getUint16(at + 2, true)),
                    scales = at + 4,
                    outBase = row + b * QK_K;
                let
                    quants = at + 16,
                    is = 0;
                for (let j = 0; j < QK_K; j += 64) {
                    const
                        [scale1, min1] = scaleMinK4({
                            bytes: data,
                            base: scales,
                            index: is,
                        }),
                        [scale2, min2] = scaleMinK4({
                            bytes: data,
                            base: scales,
                            index: is + 1,
                        }),
                        d1 = d * scale1,
                        m1 = min * min1,
                        d2 = d * scale2,
                        m2 = min * min2;
                    for (let l = 0; l < 32; l += 1) {
                        out[outBase + j + l] =
                            d1 * (data[quants + l] & 15) - m1;
                        out[outBase + j + 32 + l] =
                            d2 * (data[quants + l] >> 4) - m2;
                    }
                    quants += 32;
                    is += 2;
                }
                at += Q4_K_BYTES;
            }
        }
        return out;
    },
    /** Dequantize one Q6_K tensor, row major [rows, cols]. */
    dequantQ6K = (options: {
        bytes: Uint8Array;
        base: number;
        rows: number;
        cols: number;
    }): Float32Array => {
        const
            { bytes, base, rows, cols } = options,
            blocks = cols / QK_K,
            size = rows * blocks * Q6_K_BYTES,
            data = bytes.subarray(base, base + size),
            view = new DataView(data.buffer, data.byteOffset, size),
            out = reserveFloats(rows * cols) ??
                new Float32Array(rows * cols);
        let at = 0;
        for (let r = 0; r < rows; r += 1) {
            const row = r * cols;
            for (let b = 0; b < blocks; b += 1) {
                const
                    d = f16ToF32(view.getUint16(at + 208, true)),
                    outBase = row + b * QK_K;
                let
                    ql = at,
                    qh = at + 128,
                    scales = at + 192;
                for (let n = 0; n < QK_K; n += 128) {
                    const offset = outBase + n;
                    for (let l = 0; l < 32; l += 1) {
                        const
                            high = data[qh + l],
                            is = l >> 4,
                            low1 = data[ql + l],
                            low2 = data[ql + l + 32],
                            q1 = ((low1 & 15) | ((high & 3) << 4)) - 32,
                            q2 = ((low2 & 15) |
                                (((high >> 2) & 3) << 4)) - 32,
                            q3 = ((low1 >> 4) |
                                (((high >> 4) & 3) << 4)) - 32,
                            q4 = ((low2 >> 4) |
                                (((high >> 6) & 3) << 4)) - 32,
                            d1 = d * ((data[scales + is] << 24) >> 24),
                            d2 = d * ((data[scales + is + 2] << 24) >>
                                24),
                            d3 = d * ((data[scales + is + 4] << 24) >>
                                24),
                            d4 = d * ((data[scales + is + 6] << 24) >>
                                24);
                        out[offset + l] = d1 * q1;
                        out[offset + l + 32] = d2 * q2;
                        out[offset + l + 64] = d3 * q3;
                        out[offset + l + 96] = d4 * q4;
                    }
                    ql += 64;
                    qh += 32;
                    scales += 8;
                }
                at += Q6_K_BYTES;
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
            return readF16({
                bytes: buffer,
                offset: base,
                count: numel,
                out: reserveFloats(numel) ?? undefined,
            });
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
        if (info.type === GGML_Q4_K) {
            const
                cols = info.dims[0],
                rows = numel / cols;
            if (cols % QK_K !== 0) {
                throw new Error(`q4_k row width ${cols}`);
            }
            return dequantQ4K({ bytes: buffer, base, rows, cols });
        }
        if (info.type === GGML_Q6_K) {
            const
                cols = info.dims[0],
                rows = numel / cols;
            if (cols % QK_K !== 0) {
                throw new Error(`q6_k row width ${cols}`);
            }
            return dequantQ6K({ bytes: buffer, base, rows, cols });
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
    if (type === GGML_Q4_K) {
        return "Q4_K";
    }
    if (type === GGML_Q6_K) {
        return "Q6_K";
    }
    return `ggml:${type}`;
};

export {
    GGUF_CACHE_BYTES,
    GGUF_CACHE_CAP_BYTES,
    fullCacheBytes,
    GGML_F32,
    GGML_F16,
    GGML_Q8_0,
    GGML_Q4_K,
    GGML_Q6_K,
    readGgufTensor,
    ggmlTypeName,
};
