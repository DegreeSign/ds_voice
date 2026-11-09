// SentencePiece BPE tokenizer for MOSS-TTS-Nano.
const
    /** SentencePiece whitespace marker. */
    METASPACE = "\u2581",
    /** Added tokens that stay single pieces. */
    SPECIAL_TOKENS = [
        "<unk>",
        "<s>",
        "</s>",
        "<pad>",
        "<|im_start|>",
        "<|im_end|>",
        "<|audio_start|>",
        "<|audio_end|>",
        "<|vision_pad|>",
        "<|video_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|audio_pad|>",
        "<user_inst>",
        "</user_inst>",
    ],
    /** Split text on the special pieces. */
    SPECIAL_PATTERN = new RegExp(
        `(${SPECIAL_TOKENS.map((token) =>
            token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"),
        ).join("|")})`,
        "g",
    ),
    /** Match one byte fallback piece. */
    BYTE_PATTERN = /^<0x([0-9A-Fa-f]{2})>$/,
    /** Normalize text the way the SentencePiece model does. */
    normalize = (text: string): string => {
        const collapsed = text.normalize("NFKC").replace(/\s+/g, " ").trim();
        if (!collapsed) {
            return "";
        }
        return METASPACE + collapsed.replace(/ /g, METASPACE);
    },
    /** Read a tokenizer.json file. */
    parse = (json: string): {
        vocab: Map<string, number>;
        ranks: Map<string, number>;
        pieces: string[];
    } => {
        const
            data = JSON.parse(json) as {
                model: {
                    vocab: Record<string, number>;
                    merges: string[][];
                };
            },
            vocab = new Map(Object.entries(data.model.vocab)),
            ranks = new Map<string, number>(),
            pieces: string[] = [];
        for (const [piece, id] of vocab) {
            pieces[id] = piece;
        }
        for (let i = 0; i < data.model.merges.length; i += 1) {
            const [left, right] = data.model.merges[i];
            ranks.set(`${left}\u0000${right}`, i);
        }
        return { vocab, ranks, pieces };
    },
    /** Merge one run of symbols by merge rank. */
    merge = (options: {
        text: string;
        ranks: Map<string, number>;
        vocab: Map<string, number>;
        pieces: string[];
    }): number[] => {
        const { text, ranks, vocab, pieces } = options;
        let symbols = Array.from(text);
        while (symbols.length > 1) {
            let
                best = -1,
                bestRank = Infinity;
            for (let i = 0; i + 1 < symbols.length; i += 1) {
                const rank = ranks.get(
                    `${symbols[i]}\u0000${symbols[i + 1]}`,
                );
                if (rank !== undefined && rank < bestRank) {
                    bestRank = rank;
                    best = i;
                }
            }
            if (best < 0) {
                break;
            }
            symbols = [
                ...symbols.slice(0, best),
                symbols[best] + symbols[best + 1],
                ...symbols.slice(best + 2),
            ];
        }
        const ids: number[] = [];
        for (const symbol of symbols) {
            const id = vocab.get(symbol);
            if (id !== undefined) {
                ids.push(id);
                continue;
            }
            const bytes = new TextEncoder().encode(symbol);
            for (const byte of bytes) {
                const hex = byte.toString(16).toUpperCase().padStart(2, "0");
                ids.push(vocab.get(`<0x${hex}>`) ?? 0);
            }
        }
        return ids;
    };

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
const createMossTokenizer = (json: string): MossTokenizer => {
    const
        { vocab, ranks, pieces } = parse(json),
        encodePart = (text: string): number[] =>
            merge({ text, ranks, vocab, pieces }),
        encode = (text: string): number[] => {
            const normalized = normalize(text);
            if (!normalized) {
                return [];
            }
            const ids: number[] = [];
            for (const part of normalized.split(SPECIAL_PATTERN)) {
                if (!part) {
                    continue;
                }
                const special = vocab.get(part);
                if (special !== undefined) {
                    ids.push(special);
                    continue;
                }
                for (const id of encodePart(part)) {
                    ids.push(id);
                }
            }
            return ids;
        },
        tokenText = (id: number): string => pieces[id] ?? "",
        decode = (ids: number[]): string => {
            const decoder = new TextDecoder();
            const bytes: number[] = [];
            let text = "";
            const flush = (): void => {
                if (bytes.length === 0) {
                    return;
                }
                text += decoder.decode(new Uint8Array(bytes));
                bytes.length = 0;
            };
            for (const id of ids) {
                const piece = tokenText(id);
                const match = BYTE_PATTERN.exec(piece);
                if (match) {
                    bytes.push(Number.parseInt(match[1], 16));
                    continue;
                }
                flush();
                text += piece;
            }
            flush();
            return text.split(METASPACE).join(" ").replace(/^\s+/, "");
        };
    return { encode, tokenText, decode, size: pieces.length };
};

export { createMossTokenizer };
export type { MossTokenizer };
