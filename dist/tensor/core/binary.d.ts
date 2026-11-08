import type { SafetensorsHeaderEntry } from "./types";
declare const 
/** Bytes per element for a dtype. */
dtypeSize: (dtype: string) => number, 
/** Product of a shape. */
product: (shape: number[]) => number, 
/** Convert an ieee half to f32. */
f16ToF32: (bits: number) => number, 
/** Read a little endian f16 array as f32. */
readF16: (options: {
    bytes: Uint8Array;
    offset: number;
    count: number;
    /** Optional destination, reused when provided. */
    out?: Float32Array;
}) => Float32Array, 
/** Decode the safetensors header block. */
parseSafetensorsJson: (bytes: Uint8Array) => {
    entries: Array<[string, SafetensorsHeaderEntry]>;
    dataStart: number;
    metadata: Record<string, string>;
};
export { dtypeSize, product, f16ToF32, readF16, parseSafetensorsJson };
