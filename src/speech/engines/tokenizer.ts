// Byte pair tokenizer for the Soprano prompt format.
const
    /** Special token ids of the Soprano vocabulary. */
    STOP_ID = 3,
    TEXT_ID = 1,
    START_ID = 2,
    UNKNOWN_ID = 0,
    /** Lowercase and collapse whitespace. */
    normalize = (text: string): string => text.toLowerCase().replace(/\s+/g, " "),
    /** Split one segment into runs the BPE can consume. */
    preTokenize = (text: string): string[] => {
        const parts: string[] = [];
        let index = 0;
        while (index < text.length) {
            const char = text[index];
            if (/\s/.test(char)) {
                let end = index + 1;
                while (end < text.length && /\s/.test(text[end])) {
                    end += 1;
                }
                parts.push(text.slice(index, end));
                index = end;
            } else if (/\d/.test(char)) {
                parts.push(char);
                index += 1;
            } else if (/\w/.test(char)) {
                let end = index + 1;
                while (end < text.length && /\w/.test(text[end])) {
                    end += 1;
                }
                parts.push(text.slice(index, end));
                index = end;
            } else {
                let end = index + 1;
                while (
                    end < text.length &&
                    !/[\s\w]/.test(text[end])
                ) {
                    end += 1;
                }
                parts.push(text.slice(index, end));
                index = end;
            }
        }
        return parts;
    },
    /** Rank of every merge pair. */
    mergeRanks = (merges: string[][]): Map<string, number> => {
        const ranks = new Map<string, number>();
        for (let i = 0; i < merges.length; i += 1) {
            ranks.set(`${merges[i][0]} ${merges[i][1]}`, i);
        }
        return ranks;
    },
    /** Apply the ranked merges to one segment. */
    mergeSegment = (options: {
        segment: string;
        ranks: Map<string, number>;
        vocab: Map<string, number>;
    }): number[] => {
        const { segment, ranks, vocab } = options;
        let symbols = Array.from(segment);
        while (symbols.length > 1) {
            let
                best = -1,
                bestRank = Infinity;
            for (let i = 0; i + 1 < symbols.length; i += 1) {
                const rank = ranks.get(`${symbols[i]} ${symbols[i + 1]}`);
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
        return symbols.map((symbol) => vocab.get(symbol) ?? UNKNOWN_ID);
    };

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
const createTokenizer = (json: string): SopranoTokenizer => {
    const
        data = JSON.parse(json) as {
            model: {
                vocab: Record<string, number>;
                merges: string[][];
            };
        },
        vocab = new Map(Object.entries(data.model.vocab)),
        ranks = mergeRanks(data.model.merges),
        encode = (text: string): number[] => {
            const ids: number[] = [];
            for (const segment of preTokenize(normalize(text))) {
                const tokenIds = mergeSegment({ segment, ranks, vocab });
                for (const id of tokenIds) {
                    ids.push(id);
                }
            }
            return ids;
        };
    return {
        encode,
        prompt: (text: string): number[] => [
            STOP_ID,
            TEXT_ID,
            ...encode(text),
            START_ID,
        ],
        stop: STOP_ID,
        size: vocab.size,
    };
};

export { createTokenizer };
export type { SopranoTokenizer };
