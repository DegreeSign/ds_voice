// Word and sentence assembly.
import type { BuildWordsOptions, Segment, Word } from "../../tensor/types";
const
    /** Terminal punctuation that ends a sentence. */
    TERMINALS = new Set([".", "!", "?", "\u3002", "\uFF1F", "\uFF01"]),
    /** Build words from generated tokens and their times. */
    buildWords = (options: BuildWordsOptions): Word[] => {
        const
            { ids, tokenizer, timeOffset } = options,
            times = options.tokenTimes,
            words: Word[] = [];
        let current: { text: string; start: number; end: number } | null = null;
        const flush = (): void => {
            if (current && current.text.length > 0) {
                words.push({
                    word: current.text,
                    start: current.start,
                    end: current.end,
                });
            }
            current = null;
        };
        for (let i = 0; i < ids.length; i += 1) {
            const
                id = ids[i],
                text = tokenizer.tokenText(id);
            if (text === "" || /^<\|.*\|>$/.test(text)) {
                flush();
                continue;
            }
            const
                start = times[i] + timeOffset,
                end = (i + 1 < times.length ? times[i + 1] : times[i]) + timeOffset;
            if (current === null || text.startsWith(" ")) {
                flush();
                current = { text: text.replace(/^ /, ""), start, end };
            } else {
                current.text += text;
                current.end = Math.max(current.end, end);
            }
        }
        flush();
        return words;
    },
    /** Group words into sentence segments. */
    buildSegments = (options: {
        words: Word[];
        pauseThreshold?: number;
    }): Segment[] => {
        const
            { words } = options,
            pause = options.pauseThreshold ?? 0.6,
            segments: Segment[] = [];
        let
            group: Word[] = [],
            groupStart = 0;
        for (let i = 0; i < words.length; i += 1) {
            const word = words[i];
            if (group.length === 0) {
                groupStart = word.start;
            }
            group.push(word);
            const
                text = word.word,
                terminal = text.length > 0 && TERMINALS.has(text[text.length - 1]),
                gap = i + 1 < words.length ? words[i + 1].start - word.end : 0;
            if (terminal || gap > pause || i === words.length - 1) {
                segments.push({
                    start: groupStart,
                    end: word.end,
                    text: group.map((w) => w.word).join(" "),
                    words: group,
                });
                group = [];
            }
        }
        return segments;
    };

export { buildWords, buildSegments };
