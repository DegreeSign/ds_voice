// GGUF container reader.
import type {
    GgufFile,
    GgufTensorInfo,
    GgufValue,
    ParseGgufOptions,
} from "../tensor/types";
const
    /** Metadata type ids from the GGUF spec. */
    T_UINT8 = 0,
    T_INT8 = 1,
    T_UINT16 = 2,
    T_INT16 = 3,
    T_UINT32 = 4,
    T_INT32 = 5,
    T_FLOAT32 = 6,
    T_BOOL = 7,
    T_STRING = 8,
    T_ARRAY = 9,
    T_UINT64 = 10,
    T_INT64 = 11,
    T_FLOAT64 = 12,
    /** Default tensor data alignment. */
    DEFAULT_ALIGNMENT = 32,
    /** Parse a GGUF file into metadata and tensor entries. */
    parseGguf = (options: ParseGgufOptions): GgufFile => {
        const { bytes } = options;
        if (bytes.length < 8) {
            throw new Error("gguf file too small");
        }
        const magic = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);
        if (magic !== "GGUF") {
            throw new Error("not a gguf file");
        }
        const
            view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength),
            decoder = new TextDecoder();
        let offset = 4;
        const
            u8 = (): number => view.getUint8(offset++),
            u16 = (): number => {
                const value = view.getUint16(offset, true);
                offset += 2;
                return value;
            },
            u32 = (): number => {
                const value = view.getUint32(offset, true);
                offset += 4;
                return value;
            },
            u64 = (): number => {
                const value = Number(view.getBigUint64(offset, true));
                offset += 8;
                return value;
            },
            str = (): string => {
                const
                    length = u64(),
                    text = decoder.decode(bytes.subarray(offset, offset + length));
                offset += length;
                return text;
            },
            scalar = (type: number): GgufValue => {
                switch (type) {
                    case T_UINT8:
                        return u8();
                    case T_INT8:
                        return view.getInt8(offset++);
                    case T_UINT16:
                        return u16();
                    case T_INT16: {
                        const value = view.getInt16(offset, true);
                        offset += 2;
                        return value;
                    }
                    case T_UINT32:
                        return u32();
                    case T_INT32: {
                        const value = view.getInt32(offset, true);
                        offset += 4;
                        return value;
                    }
                    case T_FLOAT32: {
                        const value = view.getFloat32(offset, true);
                        offset += 4;
                        return value;
                    }
                    case T_BOOL:
                        return u8() !== 0;
                    case T_STRING:
                        return str();
                    case T_UINT64:
                        return u64();
                    case T_INT64: {
                        const value = Number(view.getBigInt64(offset, true));
                        offset += 8;
                        return value;
                    }
                    case T_FLOAT64: {
                        const value = view.getFloat64(offset, true);
                        offset += 8;
                        return value;
                    }
                    default:
                        throw new Error(`unsupported gguf metadata type ${type}`);
                }
            },
            value = (type: number): GgufValue => {
                if (type !== T_ARRAY) {
                    return scalar(type);
                }
                const
                    element = u32(),
                    count = u64(),
                    items: GgufValue[] = [];
                for (let i = 0; i < count; i += 1) {
                    items.push(value(element));
                }
                return items;
            },
            version = u32(),
            tensorCount = u64(),
            kvCount = u64(),
            metadata = new Map<string, GgufValue>();
        for (let i = 0; i < kvCount; i += 1) {
            const
                key = str(),
                type = u32();
            metadata.set(key, value(type));
        }
        const tensors = new Map<string, GgufTensorInfo>();
        for (let i = 0; i < tensorCount; i += 1) {
            const
                name = str(),
                rank = u32(),
                dims: number[] = [];
            for (let d = 0; d < rank; d += 1) {
                dims.push(u64());
            }
            const
                type = u32(),
                dataOffset = u64();
            tensors.set(name, { name, dims, type, offset: dataOffset });
        }
        const
            declared = metadata.get("general.alignment"),
            alignment = typeof declared === "number" ? declared : DEFAULT_ALIGNMENT,
            dataOffset = Math.ceil(offset / alignment) * alignment;
        return { version, metadata, tensors, dataOffset, buffer: bytes };
    };

export { parseGguf };
