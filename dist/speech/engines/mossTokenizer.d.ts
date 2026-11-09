/** Tokenizer loaded from one Hugging Face tokenizer.json. */
interface MossTokenizer {
    /** Token ids of one plain text line. */
    encode: (text: string) => number[];
    /** Text of one token id. */
    tokenText: (id: number) => string;
    /** Decode one run of token ids. */
    decode: (ids: number[]) => string;
    /** Vocabulary size. */
    size: number;
}
/** Build the tokenizer from parsed JSON. */
declare const createMossTokenizer: (json: string) => MossTokenizer;
export { createMossTokenizer };
export type { MossTokenizer };
