// Single source of truth for the supported models.
import { Language, ModelFamily, ModelId } from "../tensor/types";
import type { CheckpointId, ModelSpec } from "../tensor/types";
const
    /** Hugging Face base URL. */
    HF_BASE = "https://huggingface.co",
    /** Default model selection. */
    DEFAULT_MODEL = ModelId.SenseVoiceSmall,
    /** Files fetched for every Whisper checkpoint. */
    WHISPER_FILES = [
        "model.safetensors",
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "added_tokens.json",
        "special_tokens_map.json",
    ],
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
            repo: "openai/whisper-base",
            family: ModelFamily.Whisper,
            weightFile: "model.safetensors",
            weightBytes: 290_403_936,
            files: WHISPER_FILES,
        },
        "whisper-base.en": {
            id: ModelId.WhisperBase,
            checkpoint: "whisper-base.en",
            repo: "openai/whisper-base.en",
            family: ModelFamily.Whisper,
            weightFile: "model.safetensors",
            weightBytes: 290_401_888,
            files: WHISPER_FILES,
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
