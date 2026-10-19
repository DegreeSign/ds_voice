// Configuration loading and normalization.
import { checkpointKey } from "../specs";
import type {
    Constants,
    MelConfig,
    ModelConfig,
    ModelStore,
    RawGenerationConfig,
    RawModelConfig,
    RawPreprocessorConfig,
    StoreModelOptions,
} from "../../tensor/types";
const
    /** Structural token ids verified against the checkpoint. */
    TOKENS = {
        endOfText: 50257,
        startOfTranscript: 50258,
        translate: 50358,
        transcribe: 50359,
        startOfLm: 50360,
        startOfPrev: 50361,
        noSpeech: 50362,
        noTimestamps: 50363,
        timestampBegin: 50364,
        timestampEnd: 51864,
    } as const,
    /** Seconds represented by one timestamp token. */
    TIMESTAMP_STEP = 0.02,
    /** Read and parse one JSON file. */
    readJson = async <T>(options: {
        store: ModelStore;
        key: string;
    }): Promise<T> => {
        const text = await options.store.readText(options.key);
        if (text === null) {
            throw new Error(`missing ${options.key}`);
        }
        return JSON.parse(text) as T;
    },
    /** Load the three configuration files for a model. */
    loadConfigs = async (options: StoreModelOptions): Promise<{
        model: RawModelConfig;
        generation: RawGenerationConfig;
        preprocessor: RawPreprocessorConfig;
    }> => {
        const
            key = (file: string): string => checkpointKey({
                checkpoint: options.checkpoint,
                file,
            }),
            [model, generation, preprocessor] = await Promise.all([
                readJson<RawModelConfig>({
                    store: options.store,
                    key: key("config.json"),
                }),
                readJson<RawGenerationConfig>({
                    store: options.store,
                    key: key("generation_config.json"),
                }),
                readJson<RawPreprocessorConfig>({
                    store: options.store,
                    key: key("preprocessor_config.json"),
                }),
            ]);
        return { model, generation, preprocessor };
    },
    /** Normalize the raw model config. */
    toModelConfig = (raw: RawModelConfig): ModelConfig => {
        if (raw.activation_function !== "gelu") {
            throw new Error(`unexpected activation ${raw.activation_function}`);
        }
        if (raw.scale_embedding) {
            throw new Error("scale_embedding not supported");
        }
        return {
            dModel: raw.d_model,
            encoderLayers: raw.encoder_layers,
            decoderLayers: raw.decoder_layers,
            encoderHeads: raw.encoder_attention_heads,
            decoderHeads: raw.decoder_attention_heads,
            encoderFfnDim: raw.encoder_ffn_dim,
            decoderFfnDim: raw.decoder_ffn_dim,
            vocabSize: raw.vocab_size,
            numMelBins: raw.num_mel_bins,
            headDim: raw.d_model / raw.encoder_attention_heads,
            maxSourcePositions: raw.max_source_positions,
            maxTargetPositions: raw.max_target_positions,
            maxLength: raw.max_length,
            scaleEmbedding: raw.scale_embedding,
        };
    },
    /** Normalize the raw preprocessor config. */
    toMelConfig = (raw: RawPreprocessorConfig): MelConfig => {
        return {
            samplingRate: raw.sampling_rate,
            nFft: raw.n_fft,
            hopLength: raw.hop_length,
            featureSize: raw.feature_size,
            chunkLength: raw.chunk_length,
            nSamples: raw.n_samples,
            maxFrames: raw.nb_max_frames,
        };
    },
    /** Flatten the stored mel filter bank to a row major f32 array. */
    toMelFilters = (raw: RawPreprocessorConfig): Float32Array => {
        const
            rows = raw.mel_filters.length,
            cols = raw.mel_filters[0]?.length ?? 0,
            out = new Float32Array(rows * cols);
        for (let r = 0; r < rows; r += 1) {
            for (let c = 0; c < cols; c += 1) {
                out[r * cols + c] = raw.mel_filters[r][c];
            }
        }
        return out;
    },
    /** Build the token and generation constants. */
    toConstants = (raw: RawGenerationConfig): Constants => {
        const
            multilingual = raw.is_multilingual !== false,
            languageIds = new Map<string, number>(),
            idToLanguage = new Map<string, string>();
        for (const [token, id] of Object.entries(raw.lang_to_id ?? {})) {
            const code = token.replace("<|", "").replace("|>", "");
            languageIds.set(code, id);
            idToLanguage.set(String(id), code);
        }
        return {
            eosTokenId: raw.eos_token_id,
            bosTokenId: raw.bos_token_id,
            padTokenId: raw.pad_token_id,
            decoderStartTokenId: raw.decoder_start_token_id,
            isMultilingual: multilingual,
            // english models lack language
            startOfTranscript: multilingual
                ? TOKENS.startOfTranscript
                : raw.decoder_start_token_id,
            languageIds,
            idToLanguage,
            translateToken: raw.task_to_id?.translate ?? 0,
            transcribeToken: raw.task_to_id?.transcribe ?? 0,
            startOfLm: TOKENS.startOfLm,
            startOfPrev: raw.prev_sot_token_id,
            noSpeechToken: TOKENS.noSpeech,
            noTimestampsToken: raw.no_timestamps_token_id,
            timestampBegin: raw.no_timestamps_token_id + 1,
            timestampStep: TIMESTAMP_STEP,
            suppressTokens: raw.suppress_tokens,
            beginSuppressTokens: raw.begin_suppress_tokens,
            alignmentHeads: raw.alignment_heads.map((pair) => [pair[0], pair[1]]),
            maxInitialTimestampIndex: raw.max_initial_timestamp_index,
        };
    },
    /** Assert the two suppression lists match across the config files. */
    assertSuppressMatch = (options: { a: number[]; b: number[] }): void => {
        const
            { a, b } = options,
            left = [...a].sort((x, y) => x - y).join(","),
            right = [...b].sort((x, y) => x - y).join(",");
        if (left !== right) {
            throw new Error("suppress token mismatch");
        }
    };

export {
    TOKENS,
    loadConfigs,
    toModelConfig,
    toMelConfig,
    toMelFilters,
    toConstants,
    assertSuppressMatch,
};
