import { Language, ModelId } from "../tensor/types";
import type { CheckpointId, ModelSpec } from "../tensor/types";
declare const 
/** Hugging Face base URL. */
HF_BASE = "https://huggingface.co", 
/** Default model selection. */
DEFAULT_MODEL = ModelId.SenseVoiceSmall, 
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
}) => ModelSpec, 
/** Store key for one checkpoint file. */
checkpointKey: (options: {
    checkpoint: CheckpointId;
    file: string;
}) => string;
export { HF_BASE, DEFAULT_MODEL, MODEL_SPECS, MODEL_IDS, CHECKPOINT_IDS, resolveSpec, checkpointKey, };
