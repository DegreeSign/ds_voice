import type { CheckpointStoreOptions, Tokenizer } from "../../tensor/core/types";
declare const 
/** Load the tokenizer for a model. */
loadTokenizer: (options: CheckpointStoreOptions) => Promise<Tokenizer>;
export { loadTokenizer };
