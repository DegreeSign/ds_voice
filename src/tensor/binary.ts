// Low level binary helpers.
import type { SafetensorsHeader, SafetensorsHeaderEntry } from "./types";
const
    /** Bytes per element for a dtype. */
    dtypeSize = (dtype: string): number => {
        switch (dtype) {
            case "F32":
            case "I32":
                return 4;
            case "F16":
            case "BF16":
            case "I16":
                return 2;
            case "I64":
                return 8;
            case "U8":
            case "I8":
            case "BOOL":
                return 1;
            default:
                throw new Error(`unsupported dtype ${dtype}`);
        }
    },
    /** Product of a shape. */
    product = (shape: number[]): number => {
        let total = 1;
        for (const dim of shape) {
            total *= dim;
        }
        return total;
    },
    /** Every ieee half as f32. */
    halfTable = ((): Float32Array => {
        const table = new Float32Array(65536);
        for (let bits = 0; bits < 65536; bits += 1) {
            const
                sign = (bits & 0x8000) >> 15,
                exponent = (bits & 0x7c00) >> 10,
                fraction = bits & 0x03ff;
            if (exponent === 0) {
                table[bits] =
                    (sign ? -1 : 1) * 2 ** -14 * (fraction / 1024);
            } else if (exponent === 0x1f) {
                table[bits] = fraction
                    ? NaN
                    : (sign ? -1 : 1) * Infinity;
            } else {
                table[bits] =
                    (sign ? -1 : 1) *
                    2 ** (exponent - 15) *
                    (1 + fraction / 1024);
            }
        }
        return table;
    })(),
    /** Convert an ieee half to f32. */
    f16ToF32 = (bits: number): number => halfTable[bits & 0xffff],
    /** Read a little endian f16 array as f32. */
    readF16 = (options: {
        bytes: Uint8Array;
        offset: number;
        count: number;
        /** Optional destination, reused when provided. */
        out?: Float32Array;
    }): Float32Array => {
        const
            { bytes, offset, count } = options,
            base = bytes.byteOffset + offset,
            out = options.out ?? new Float32Array(count);
        if (base % 2 === 0) {
            const halves = new Uint16Array(bytes.buffer, base, count);
            for (let i = 0; i < count; i += 1) {
                out[i] = halfTable[halves[i]];
            }
            return out;
        }
        const view = new DataView(bytes.buffer, base, count * 2);
        for (let i = 0; i < count; i += 1) {
            out[i] = halfTable[view.getUint16(i * 2, true)];
        }
        return out;
    },
    /** Decode the safetensors header block. */
    parseSafetensorsJson = (bytes: Uint8Array): {
        entries: Array<[string, SafetensorsHeaderEntry]>;
        dataStart: number;
        metadata: Record<string, string>;
    } => {
        const
            view = new DataView(bytes.buffer, bytes.byteOffset, 8),
            headerLen = Number(view.getBigUint64(0, true)),
            json = new TextDecoder().decode(bytes.subarray(8, 8 + headerLen)),
            parsed = JSON.parse(json) as SafetensorsHeader,
            metadata = parsed.__metadata__ ?? {},
            entries: Array<[string, SafetensorsHeaderEntry]> = [];
        for (const [name, value] of Object.entries(parsed)) {
            if (name === "__metadata__") {
                continue;
            }
            entries.push([name, value as SafetensorsHeaderEntry]);
        }
        return { entries, dataStart: 8 + headerLen, metadata };
    };

export { dtypeSize, product, f16ToF32, readF16, parseSafetensorsJson };
