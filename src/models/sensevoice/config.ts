// SenseVoice configuration parsing.
import type {
    GgufFile,
    GgufValue,
    SenseVoiceConfig,
    SenseVoiceFrontend,
    SenseVoiceWeights,
} from "../../tensor/types";
const
    /** LayerNorm epsilon used by this family. */
    LAYER_NORM_EPSILON = 1e-12,
    /** Read one numeric metadata key. */
    readNumber = (options: {
        meta: Map<string, GgufValue>;
        key: string;
    }): number => {
        const
            { meta, key } = options,
            value = meta.get(key);
        if (typeof value !== "number") {
            throw new Error(`missing gguf key ${key}`);
        }
        return value;
    },
    /** Read one array metadata key. */
    readArray = (options: {
        meta: Map<string, GgufValue>;
        key: string;
    }): GgufValue[] => {
        const
            { meta, key } = options,
            value = meta.get(key);
        if (!Array.isArray(value)) {
            throw new Error(`missing gguf key ${key}`);
        }
        return value;
    },
    /** Build the normalized encoder shape. */
    readSenseVoiceConfig = (options: {
        file: GgufFile;
    }): SenseVoiceConfig => {
        const
            meta = options.file.metadata,
            dModel = readNumber({ meta, key: "stt.sensevoice.encoder.d_model" }),
            heads = readNumber({ meta, key: "stt.sensevoice.encoder.n_heads" }),
            languageIds = new Map<string, number>();
        languageIds.set("auto", readNumber({ meta, key: "stt.sensevoice.special.lang_auto" }));
        languageIds.set("zh", readNumber({ meta, key: "stt.sensevoice.special.lang_zh" }));
        languageIds.set("en", readNumber({ meta, key: "stt.sensevoice.special.lang_en" }));
        languageIds.set("yue", readNumber({ meta, key: "stt.sensevoice.special.lang_yue" }));
        languageIds.set("ja", readNumber({ meta, key: "stt.sensevoice.special.lang_ja" }));
        languageIds.set("ko", readNumber({ meta, key: "stt.sensevoice.special.lang_ko" }));
        languageIds.set(
            "nospeech",
            readNumber({ meta, key: "stt.sensevoice.special.lang_nospeech" }),
        );
        const vocabSize = readArray({ meta, key: "tokenizer.ggml.tokens" }).length;
        return {
            dInput: readNumber({ meta, key: "stt.sensevoice.encoder.d_input" }),
            dModel,
            heads,
            headDim: dModel / heads,
            ffnDim: readNumber({ meta, key: "stt.sensevoice.encoder.d_ff" }),
            kernelSize: readNumber({ meta, key: "stt.sensevoice.encoder.kernel_size" }),
            mainBlocks: readNumber({ meta, key: "stt.sensevoice.encoder.n_blocks" }),
            tpBlocks: readNumber({ meta, key: "stt.sensevoice.encoder.tp_blocks" }),
            vocabSize,
            eps: LAYER_NORM_EPSILON,
            languageIds,
            eventEmotion: [
                readNumber({ meta, key: "stt.sensevoice.special.event_speech" }),
                readNumber({ meta, key: "stt.sensevoice.special.emotion_neutral" }),
            ],
            withItn: readNumber({ meta, key: "stt.sensevoice.special.withitn" }),
            withoutItn: readNumber({ meta, key: "stt.sensevoice.special.woitn" }),
        };
    },
    /** Build the kaldi fbank frontend settings. */
    readSenseVoiceFrontend = (options: {
        file: GgufFile;
        weights: SenseVoiceWeights;
    }): SenseVoiceFrontend => {
        const
            meta = options.file.metadata,
            upscale = meta.get("stt.frontend.upscale_samples") === true,
            shift = options.weights.f32("frontend.cmvn.shift"),
            scale = options.weights.f32("frontend.cmvn.scale");
        return {
            samplingRate: readNumber({ meta, key: "stt.frontend.sample_rate" }),
            numMels: readNumber({ meta, key: "stt.frontend.num_mels" }),
            nFft: readNumber({ meta, key: "stt.frontend.n_fft" }),
            winLength: readNumber({ meta, key: "stt.frontend.win_length" }),
            hopLength: readNumber({ meta, key: "stt.frontend.hop_length" }),
            lfrM: readNumber({ meta, key: "stt.frontend.lfr_m" }),
            lfrN: readNumber({ meta, key: "stt.frontend.lfr_n" }),
            upscale: upscale ? 32768 : 1,
            shift: shift.slice(),
            scale: scale.slice(),
        };
    };

export { readSenseVoiceConfig, readSenseVoiceFrontend };
