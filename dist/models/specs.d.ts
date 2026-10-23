import { Language, ModelId, Quality } from "../tensor/types";
import type { CheckpointId, ModelSpec } from "../tensor/types";
declare const 
/** Hugging Face base URL. */
HF_BASE = "https://huggingface.co", 
/** Default model selection. */
DEFAULT_MODEL = ModelId.WhisperBase, 
/** Default weight precision. */
DEFAULT_QUALITY = Quality.High, 
/** Every supported checkpoint. */
MODEL_SPECS: Record<CheckpointId, ModelSpec>, 
/** Every public model id, in spec order. */
MODEL_IDS: ModelId[], 
/** Every concrete checkpoint, in spec order. */
CHECKPOINT_IDS: CheckpointId[], 
/** Pick the checkpoint for one model and language. */
resolveSpec: (options?: {
    model?: ModelId;
    language?: Language;
    quality?: Quality;
}) => ModelSpec, 
/** Every weight file of one checkpoint. */
weightFiles: (spec: ModelSpec) => string[], 
/** Every file of one checkpoint across qualities. */
checkpointFiles: (spec: ModelSpec) => string[], 
/** Store key for one checkpoint file. */
checkpointKey: (options: {
    checkpoint: CheckpointId;
    file: string;
}) => string;
export { HF_BASE, DEFAULT_MODEL, DEFAULT_QUALITY, MODEL_SPECS, MODEL_IDS, CHECKPOINT_IDS, resolveSpec, weightFiles, checkpointFiles, checkpointKey, };
