// Safetensors zero copy parsing.
import { dtypeSize, parseSafetensorsJson, product } from "../tensor/binary";
import type {
    ParseSafetensorsOptions,
    SafetensorsFile,
    TensorInfo,
    TensorStore,
} from "../tensor/types";
const
    /** Parse a safetensors buffer into a tensor store. */
    parseSafetensors = (
        options: ParseSafetensorsOptions,
    ): SafetensorsFile => {
        const { bytes } = options;
        if (bytes.byteLength < 8) {
            throw new Error("safetensors too small");
        }
        const
            { entries, dataStart, metadata } = parseSafetensorsJson(bytes),
            infos = new Map<string, TensorInfo>();
        for (const [name, entry] of entries) {
            const
                numel = product(entry.shape),
                expected = numel * dtypeSize(entry.dtype),
                [begin, end] = entry.data_offsets;
            if (end - begin !== expected) {
                throw new Error(`tensor size mismatch ${name}`);
            }
            infos.set(name, {
                name,
                dtype: entry.dtype,
                shape: entry.shape,
                dataOffset: dataStart + begin,
                numel,
            });
        }
        const store = makeStore({ bytes, infos, dataStart });
        return { store, buffer: bytes, metadata };
    },
    /** Build the read only tensor store. */
    makeStore = (input: {
        bytes: Uint8Array;
        infos: Map<string, TensorInfo>;
        dataStart: number;
    }): TensorStore => {
        const { bytes, infos } = input;
        if (input.dataStart % 4 !== 0) {
            throw new Error("data start not aligned");
        }
        const
            cache = new Map<string, Float32Array>(),
            get = (name: string): Float32Array => {
                const hit = cache.get(name);
                if (hit) {
                    return hit;
                }
                const info = infos.get(name);
                if (!info) {
                    throw new Error(`missing tensor ${name}`);
                }
                if (info.dtype !== "F32") {
                    throw new Error(`tensor ${name} is ${info.dtype}`);
                }
                if (info.dataOffset % 4 !== 0) {
                    throw new Error(`tensor ${name} not aligned`);
                }
                const view = new Float32Array(
                    bytes.buffer,
                    bytes.byteOffset + info.dataOffset,
                    info.numel,
                );
                cache.set(name, view);
                return view;
            };
        return {
            count: infos.size,
            names: () => [...infos.keys()],
            has: (name: string) => infos.has(name),
            info: (name: string) => {
                const info = infos.get(name);
                if (!info) {
                    throw new Error(`missing tensor ${name}`);
                }
                return info;
            },
            get,
        };
    },
    /** Assert the Whisper base tensor invariants. */
    assertWhisperShape = (options: {
        store: TensorStore;
        vocabSize: number;
        dModel: number;
        /** Accept the quantized dtypes of a GGUF checkpoint. */
        allowQuantized?: boolean;
    }): void => {
        const { store, vocabSize, dModel } = options;
        if (store.has("proj_out")) {
            throw new Error("unexpected proj_out tensor");
        }
        const embed = store.info("model.decoder.embed_tokens.weight");
        if (embed.shape[0] !== vocabSize || embed.shape[1] !== dModel) {
            throw new Error(`embed shape ${embed.shape.join("x")}`);
        }
        for (const name of store.names()) {
            const dtype = store.info(name).dtype;
            if (options.allowQuantized) {
                if (
                    dtype !== "F32" &&
                    dtype !== "F16" &&
                    dtype !== "Q8_0"
                ) {
                    throw new Error(`unsupported tensor ${name} ${dtype}`);
                }
                continue;
            }
            if (dtype !== "F32") {
                throw new Error(`non f32 tensor ${name}`);
            }
        }
    };

export { parseSafetensors, assertWhisperShape };
