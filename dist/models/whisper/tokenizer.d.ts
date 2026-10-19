import type { StoreModelOptions, Tokenizer } from "../../tensor/types";
declare const 
/** Load the tokenizer for a model. */
loadTokenizer: (options: StoreModelOptions) => Promise<Tokenizer>;
export { loadTokenizer };
