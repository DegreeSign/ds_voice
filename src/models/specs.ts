// Single source of truth for the supported models.
import { Language, ModelFamily, ModelId, Quality } from "../tensor/types";
import type {
    CheckpointId,
    ModelSpec,
    ModelWeight,
} from "../tensor/types";
const
    /** Hugging Face base URL. */
    HF_BASE = "https://huggingface.co",
    /** Default model selection. */
    DEFAULT_MODEL = ModelId.WhisperBase,
    /** Default weight precision. */
    DEFAULT_QUALITY = Quality.High,
    /** Files fetched for every Whisper checkpoint. */
    WHISPER_FILES = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "added_tokens.json",
        "special_tokens_map.json",
    ],
    /** Every weight variant of the multilingual Whisper checkpoint. */
    WHISPER_WEIGHTS: Record<Quality, ModelWeight> = {
        [Quality.High]: {
            file: "whisper-base-F16.gguf",
            bytes: 151_145_760,
        },
        [Quality.Medium]: {
            file: "whisper-base-Q8_0.gguf",
            bytes: 84_962_880,
        },
        [Quality.Low]: {
            file: "whisper-base-Q4_K_M.gguf",
            bytes: 58_870_848,
        },
    },
    /** Every weight variant of the English Whisper checkpoint. */
    WHISPER_EN_WEIGHTS: Record<Quality, ModelWeight> = {
        [Quality.High]: {
            file: "whisper-base.en-F16.gguf",
            bytes: 151_068_608,
        },
        [Quality.Medium]: {
            file: "whisper-base.en-Q8_0.gguf",
            bytes: 84_886_208,
        },
        [Quality.Low]: {
            file: "whisper-base.en-Q4_K_M.gguf",
            bytes: 58_794_272,
        },
    },
    /** Every weight variant of the SenseVoice checkpoint. */
    SENSEVOICE_WEIGHTS: Record<Quality, ModelWeight> = {
        [Quality.High]: {
            file: "SenseVoiceSmall-F16.gguf",
            bytes: 470_412_128,
        },
        [Quality.Medium]: {
            file: "SenseVoiceSmall-Q8_0.gguf",
            bytes: 252_684_608,
        },
        [Quality.Low]: {
            file: "SenseVoiceSmall-Q4_K_M.gguf",
            bytes: 145_738_304,
        },
    },
    /** Every supported checkpoint. */
    MODEL_SPECS: Record<CheckpointId, ModelSpec> = {
        [ModelId.SenseVoiceSmall]: {
            id: ModelId.SenseVoiceSmall,
            checkpoint: ModelId.SenseVoiceSmall,
            repo: "handy-computer/SenseVoiceSmall-gguf",
            family: ModelFamily.SenseVoice,
            quality: DEFAULT_QUALITY,
            weightFile: SENSEVOICE_WEIGHTS[Quality.High].file,
            weightBytes: SENSEVOICE_WEIGHTS[Quality.High].bytes,
            weights: SENSEVOICE_WEIGHTS,
            files: [SENSEVOICE_WEIGHTS[Quality.High].file],
        },
        [ModelId.WhisperBase]: {
            id: ModelId.WhisperBase,
            checkpoint: ModelId.WhisperBase,
            repo: "handy-computer/whisper-base-gguf",
            textRepo: "openai/whisper-base",
            family: ModelFamily.Whisper,
            quality: DEFAULT_QUALITY,
            weightFile: WHISPER_WEIGHTS[Quality.High].file,
            weightBytes: WHISPER_WEIGHTS[Quality.High].bytes,
            weights: WHISPER_WEIGHTS,
            files: [WHISPER_WEIGHTS[Quality.High].file, ...WHISPER_FILES],
        },
        "whisper-base.en": {
            id: ModelId.WhisperBase,
            checkpoint: "whisper-base.en",
            repo: "handy-computer/whisper-base.en-gguf",
            textRepo: "openai/whisper-base.en",
            family: ModelFamily.Whisper,
            quality: DEFAULT_QUALITY,
            weightFile: WHISPER_EN_WEIGHTS[Quality.High].file,
            weightBytes: WHISPER_EN_WEIGHTS[Quality.High].bytes,
            weights: WHISPER_EN_WEIGHTS,
            files: [WHISPER_EN_WEIGHTS[Quality.High].file, ...WHISPER_FILES],
        },
    },
    /** Every public model id, in spec order. */
    MODEL_IDS = [ModelId.SenseVoiceSmall, ModelId.WhisperBase],
    /** Every concrete checkpoint, in spec order. */
    CHECKPOINT_IDS = Object.keys(MODEL_SPECS) as CheckpointId[],
    /** Pick the checkpoint for one model and language. */
    resolveSpec = (options: {
        model?: ModelId;
        language?: Language;
        quality?: Quality;
    } = {}): ModelSpec => {
        const
            model = options.model ?? DEFAULT_MODEL,
            checkpoint: CheckpointId =
                model === ModelId.WhisperBase &&
                    options.language === Language.en
                    ? "whisper-base.en"
                    : model,
            base = MODEL_SPECS[checkpoint],
            quality = options.quality ?? DEFAULT_QUALITY,
            weight = base.weights[quality];
        return {
            ...base,
            quality,
            weightFile: weight.file,
            weightBytes: weight.bytes,
            files: [
                weight.file,
                ...base.files.filter((file) => file !== base.weightFile),
            ],
        };
    },
    /** Every weight file of one checkpoint. */
    weightFiles = (spec: ModelSpec): string[] =>
        Object.values(spec.weights).map((weight) => weight.file),
    /** Every file of one checkpoint across qualities. */
    checkpointFiles = (spec: ModelSpec): string[] => [
        ...new Set([...spec.files, ...weightFiles(spec)]),
    ],
    /** Store key for one checkpoint file. */
    checkpointKey = (options: {
        checkpoint: CheckpointId;
        file: string;
    }): string => {
        return `${options.checkpoint}/${options.file}`;
    };

export {
    HF_BASE,
    DEFAULT_MODEL,
    DEFAULT_QUALITY,
    MODEL_SPECS,
    MODEL_IDS,
    CHECKPOINT_IDS,
    resolveSpec,
    weightFiles,
    checkpointFiles,
    checkpointKey,
};
