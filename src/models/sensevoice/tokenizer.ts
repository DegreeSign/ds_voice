// SenseVoice sentencepiece decoding.
import type {
    GgufFile,
    Language,
    SenseVoiceVocabulary,
    Tokenizer,
} from "../../tensor/types";
const
    /** SentencePiece whitespace marker. */
    METASPACE = "\u2581",
    /** GGUF token type code for control tokens. */
    CONTROL_TYPE = 3,
    /** Match a lower case language control token. */
    LANGUAGE_PATTERN = /^<\|([a-z]{2,3})\|>$/,
    /** Build a detokenizer from the GGUF token list. */
    createSenseVoiceVocabulary = (options: {
        file: GgufFile;
    }): SenseVoiceVocabulary => {
        const
            tokens = options.file.metadata.get("tokenizer.ggml.tokens"),
            types = options.file.metadata.get("tokenizer.ggml.token_type");
        if (!Array.isArray(tokens) || !Array.isArray(types)) {
            throw new Error("gguf is missing the token list");
        }
        const
            pieces: string[] = [],
            control = new Set<number>(),
            language = new Map<number, string>();
        for (let i = 0; i < tokens.length; i += 1) {
            const text = String(tokens[i]);
            pieces.push(text.split(METASPACE).join(" "));
            if (types[i] === CONTROL_TYPE) {
                control.add(i);
            }
            const match = LANGUAGE_PATTERN.exec(text);
            if (match) {
                language.set(i, match[1]);
            }
        }
        const
            tokenText = (id: number): string => {
                const text = pieces[id] ?? "";
                return control.has(id) ? "" : text;
            },
            decode = (ids: number[]): string => {
                let text = "";
                for (const id of ids) {
                    text += tokenText(id);
                }
                return text.replace(/^\s+/, "").replace(/\s+$/, "");
            },
            tokenizer: Tokenizer = {
                decode,
                tokenText,
                vocabSize: pieces.length,
            };
        return {
            tokenizer,
            isControl: (id: number) => control.has(id),
            languageOf: (id: number) =>
                (language.get(id) as Language | undefined) ?? null,
        };
    };

export { createSenseVoiceVocabulary };
