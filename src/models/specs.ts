// Single source of truth for the supported models.
import { Language, ModelFamily, ModelId } from "../tensor/types";
import type { CheckpointId, ModelSpec } from "../tensor/types";
const
    /** Hugging Face base URL. */
    HF_BASE = "https://huggingface.co",
    /** Default model selection. */
    DEFAULT_MODEL = ModelId.WhisperBase,
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
    /** Weight file of the multilingual Whisper checkpoint. */
    WHISPER_WEIGHT = "whisper-base-Q8_0.gguf",
    /** Weight file of the English Whisper checkpoint. */
    WHISPER_EN_WEIGHT = "whisper-base.en-Q8_0.gguf",
    /** Every supported checkpoint. */
    MODEL_SPECS: Record<CheckpointId, ModelSpec> = {
        [ModelId.SenseVoiceSmall]: {
            id: ModelId.SenseVoiceSmall,
            checkpoint: ModelId.SenseVoiceSmall,
            repo: "handy-computer/SenseVoiceSmall-gguf",
            family: ModelFamily.SenseVoice,
            weightFile: "SenseVoiceSmall-Q8_0.gguf",
            weightBytes: 252_684_608,
            files: ["SenseVoiceSmall-Q8_0.gguf"],
        },
        [ModelId.WhisperBase]: {
            id: ModelId.WhisperBase,
            checkpoint: ModelId.WhisperBase,
            repo: "handy-computer/whisper-base-gguf",
            textRepo: "openai/whisper-base",
            family: ModelFamily.Whisper,
            weightFile: WHISPER_WEIGHT,
            weightBytes: 84_962_880,
            files: [WHISPER_WEIGHT, ...WHISPER_FILES],
        },
        "whisper-base.en": {
            id: ModelId.WhisperBase,
            checkpoint: "whisper-base.en",
            repo: "handy-computer/whisper-base.en-gguf",
            textRepo: "openai/whisper-base.en",
            family: ModelFamily.Whisper,
            weightFile: WHISPER_EN_WEIGHT,
            weightBytes: 84_886_208,
            files: [WHISPER_EN_WEIGHT, ...WHISPER_FILES],
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
    } = {}): ModelSpec => {
        const model = options.model ?? DEFAULT_MODEL;
        if (model === ModelId.WhisperBase && options.language === Language.en) {
            return MODEL_SPECS["whisper-base.en"];
        }
        return MODEL_SPECS[model];
    },
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
    MODEL_SPECS,
    MODEL_IDS,
    CHECKPOINT_IDS,
    resolveSpec,
    checkpointKey,
};
