/** Tokenizer loaded from one Hugging Face tokenizer.json. */
interface SopranoTokenizer {
    /** Token ids of a plain text line, without the prompt markers. */
    encode: (text: string) => number[];
    /** Full prompt ids for one sentence. */
    prompt: (text: string) => number[];
    /** Stop token id. */
    stop: number;
    /** Vocabulary size. */
    size: number;
}
/** Build the tokenizer from parsed JSON. */
declare const createTokenizer: (json: string) => SopranoTokenizer;
export { createTokenizer };
export type { SopranoTokenizer };
