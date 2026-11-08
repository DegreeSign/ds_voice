// English grapheme to phoneme conversion for the Kokoro voice.
import { decodeLexicon } from "./lexicon";
const
    /** Decoded lexicon, built on first use. */
    lexicon = (() => {
        let map: Map<string, string> | undefined;
        return (): Map<string, string> => {
            map ??= decodeLexicon();
            return map;
        };
    })(),
    /** Vowel letters used by the suffix rules. */
    VOWELS = "aeiou",
    /** Vowels that make a following t flap in American English. */
    TAUS = "AIOWYiuæɑəɛɪɹʊʌ",
    /** Punctuation carried into the phoneme stream. */
    PUNCTUATION = ";:,.!?—…\"“”()",
    /** Multi letter graphemes, longest first. */
    GRAPHEMES: ReadonlyArray<readonly [string, string]> = [
        ["tion", "ʃən"],
        ["sion", "ʒən"],
        ["ough", "ɔ"],
        ["augh", "ɔ"],
        ["eigh", "A"],
        ["igh", "I"],
        ["tch", "ʧ"],
        ["dge", "ʤ"],
        ["sch", "sk"],
        ["shr", "ʃɹ"],
        ["thr", "θɹ"],
        ["ph", "f"],
        ["sh", "ʃ"],
        ["ch", "ʧ"],
        ["th", "θ"],
        ["wh", "w"],
        ["ck", "k"],
        ["ng", "ŋ"],
        ["qu", "kw"],
        ["oo", "u"],
        ["ee", "i"],
        ["ea", "i"],
        ["ai", "A"],
        ["ay", "A"],
        ["oa", "O"],
        ["oe", "O"],
        ["ou", "W"],
        ["ow", "O"],
        ["oi", "Y"],
        ["oy", "Y"],
        ["au", "ɔ"],
        ["aw", "ɔ"],
        ["ew", "u"],
        ["ie", "i"],
        ["ei", "A"],
        ["ey", "A"],
        ["ue", "u"],
        ["ui", "u"],
        ["ar", "ɑɹ"],
        ["er", "əɹ"],
        ["ir", "əɹ"],
        ["or", "ɔɹ"],
        ["ur", "əɹ"],
        ["kn", "n"],
        ["wr", "ɹ"],
        ["mb", "m"],
    ],
    /** Single letter pronunciations. */
    LETTERS: Readonly<Record<string, string>> = {
        a: "æ",
        b: "b",
        c: "k",
        d: "d",
        e: "ɛ",
        f: "f",
        g: "ɡ",
        h: "h",
        i: "ɪ",
        j: "ʤ",
        k: "k",
        l: "l",
        m: "m",
        n: "n",
        o: "ɑ",
        p: "p",
        q: "k",
        r: "ɹ",
        s: "s",
        t: "t",
        u: "ʌ",
        v: "v",
        w: "w",
        x: "ks",
        y: "i",
        z: "z",
    },
    /** Cardinal names below twenty. */
    ONES = [
        "zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
        "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
    ],
    /** Cardinal names of the tens. */
    TENS = [
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
        "eighty", "ninety",
    ],
    /** Scale names of large groups. */
    SCALES = ["", "thousand", "million", "billion", "trillion"],
    /** Look up one word without any morphology. */
    direct = (word: string): string | undefined => {
        const map = lexicon();
        return map.get(word) ?? map.get(word.toLowerCase()) ??
            map.get(word.toUpperCase()) ??
            map.get(word[0].toUpperCase() + word.slice(1).toLowerCase());
    },
    /** True when the lexicon knows the word. */
    known = (word: string): boolean => direct(word) !== undefined,
    /** Append a plural or third person s. */
    addS = (stem: string): string => {
        const last = stem[stem.length - 1];
        if ("ptkfθ".includes(last)) {
            return `${stem}s`;
        }
        if ("szʃʒʧʤ".includes(last)) {
            return `${stem}ᵻz`;
        }
        return `${stem}z`;
    },
    /** Append a past tense d. */
    addEd = (stem: string): string => {
        const last = stem[stem.length - 1];
        if ("pkfθʃsʧ".includes(last)) {
            return `${stem}t`;
        }
        if (last === "d") {
            return `${stem}ᵻd`;
        }
        if (last !== "t") {
            return `${stem}d`;
        }
        if (stem.length > 1 && TAUS.includes(stem[stem.length - 2])) {
            return `${stem.slice(0, -1)}ɾᵻd`;
        }
        return `${stem}ᵻd`;
    },
    /** Append a progressive ing. */
    addIng = (stem: string): string => {
        const last = stem[stem.length - 1];
        if (stem.length > 1 && last === "t" &&
            TAUS.includes(stem[stem.length - 2])) {
            return `${stem.slice(0, -1)}ɾɪŋ`;
        }
        return `${stem}ɪŋ`;
    },
    /** Phonemes of a word from the lexicon or its inflections. */
    lookup = (word: string): string | undefined => {
        const exact = direct(word);
        if (exact !== undefined) {
            return exact;
        }
        if (word.endsWith("s'")) {
            const base = direct(word.slice(0, -2) + "'s");
            if (base !== undefined) {
                return base;
            }
        }
        if (word.endsWith("'") && known(word.slice(0, -1))) {
            return direct(word.slice(0, -1));
        }
        if (word.length >= 3 && word.endsWith("s") &&
            !word.endsWith("ss") && known(word.slice(0, -1))) {
            return addS(direct(word.slice(0, -1)) as string);
        }
        if (word.length > 4 && word.endsWith("es") &&
            !word.endsWith("ies") && known(word.slice(0, -2))) {
            return addS(direct(word.slice(0, -2)) as string);
        }
        if (word.length > 4 && word.endsWith("ies") &&
            known(word.slice(0, -3) + "y")) {
            return addS(direct(word.slice(0, -3) + "y") as string);
        }
        if (word.length >= 4 && word.endsWith("d") &&
            !word.endsWith("dd") && known(word.slice(0, -1))) {
            return addEd(direct(word.slice(0, -1)) as string);
        }
        if (word.length > 4 && word.endsWith("ed") &&
            !word.endsWith("eed") && known(word.slice(0, -2))) {
            return addEd(direct(word.slice(0, -2)) as string);
        }
        if (word.length >= 5 && word.endsWith("ing")) {
            const stem = word.slice(0, -3);
            if (word.length > 5 && known(stem)) {
                return addIng(direct(stem) as string);
            }
            if (known(`${stem}e`)) {
                return addIng(direct(`${stem}e`) as string);
            }
            if (word.length > 5 && known(word.slice(0, -4)) &&
                /([bcdgklmnprstvxz])\1ing$|cking$/.test(word)) {
                return addIng(direct(word.slice(0, -4)) as string);
            }
        }
        return undefined;
    },
    /** Spell a word letter by letter. */
    spell = (word: string): string | undefined => {
        const parts: string[] = [];
        for (const letter of word) {
            const value = lexicon().get(letter.toUpperCase());
            if (value === undefined) {
                return undefined;
            }
            parts.push(value);
        }
        return parts.join("");
    },
    /** Rule based fallback for words outside the lexicon. */
    fallback = (word: string): string => {
        const
            text = word.toLowerCase().replace(/[^a-z']/g, ""),
            out: string[] = [];
        let index = 0;
        while (index < text.length) {
            const rest = text.slice(index);
            if (rest === "e" && index > 1 && text[index - 1] !== "e") {
                break;
            }
            const grapheme = GRAPHEMES.find(([key]) => rest.startsWith(key));
            if (grapheme) {
                out.push(grapheme[1]);
                index += grapheme[0].length;
                continue;
            }
            const letter = rest[0];
            const next = rest[1] ?? "";
            if (letter === "c" && "eiy".includes(next)) {
                out.push("s");
            } else if (letter === "g" && "eiy".includes(next)) {
                out.push("ʤ");
            } else if (letter === "x" && index === 0) {
                out.push("z");
            } else {
                out.push(LETTERS[letter] ?? "");
            }
            index += 1;
        }
        const phonemes = out.join("");
        return `${phonemes.includes("ˈ") ? "" : "ˈ"}${phonemes}`;
    },
    /** One word as phonemes. */
    word = (text: string, next?: string): string => {
        const lower = text.toLowerCase();
        if (lower === "a") {
            return "ɐ";
        }
        if (lower === "an") {
            return "ɐn";
        }
        if (lower === "the") {
            const following = (next ?? "").toLowerCase();
            return following.length > 0 && "aeiou".includes(following[0])
                ? "ði"
                : "ðə";
        }
        const spoken = lookup(text);
        if (spoken !== undefined) {
            return spoken;
        }
        if (text.length > 1 && text === text.toUpperCase()) {
            return spell(text) ?? fallback(text);
        }
        if (text.length === 1) {
            return spell(text) ?? fallback(text);
        }
        return fallback(text);
    },
    /** Spell one integer group below one thousand. */
    group = (value: number): string[] => {
        const
            words: string[] = [],
            hundreds = Math.floor(value / 100),
            rest = value % 100;
        if (hundreds > 0) {
            words.push(ONES[hundreds], "hundred");
        }
        if (rest >= 20) {
            words.push(TENS[Math.floor(rest / 10)]);
            if (rest % 10 > 0) {
                words.push(ONES[rest % 10]);
            }
        } else if (rest > 0) {
            words.push(ONES[rest]);
        }
        return words;
    },
    /** Spell one decimal number. */
    number = (token: string): string => {
        const
            clean = token.replace(/,/g, ""),
            negative = clean.startsWith("-"),
            digits = negative ? clean.slice(1) : clean,
            [wholeText, fraction] = digits.split("."),
            whole = Number.parseInt(wholeText || "0", 10);
        if (!Number.isFinite(whole)) {
            return digits;
        }
        const
            words: string[] = [],
            scales: string[] = [];
        let remaining = whole;
        let scale = 0;
        if (remaining === 0) {
            words.push("zero");
        }
        while (remaining > 0 && scale < SCALES.length) {
            const part = remaining % 1000;
            if (part > 0) {
                scales.unshift(
                    [...group(part), SCALES[scale]].join(" ").trim(),
                );
            }
            remaining = Math.floor(remaining / 1000);
            scale += 1;
        }
        words.push(...scales);
        if (fraction) {
            words.push("point");
            for (const digit of fraction) {
                words.push(ONES[Number(digit)] ?? digit);
            }
        }
        return `${negative ? "minus " : ""}${words.join(" ")}`;
    },
    /** Phoneme stream for one English text. */
    phonemizeEnglish = (text: string): string => {
        const
            pieces: string[] = [],
            tokens = text.match(/[A-Za-z][A-Za-z']*|\d[\d.,]*|[\s\S]/g);
        if (!tokens) {
            return "";
        }
        let
            /** Whether the next word needs a space. */
            pendingSpace = false,
            /** Index of the current token. */
            index = -1;
        for (const token of tokens) {
            index += 1;
            if (/^[A-Za-z]/.test(token)) {
                const following = tokens.slice(index + 1).find(
                    (value) => /^[A-Za-z]/.test(value),
                );
                const spoken = word(token, following);
                if (pieces.length > 0 && pendingSpace) {
                    pieces.push(" ");
                }
                pieces.push(spoken);
                pendingSpace = true;
                continue;
            }
            if (/^\d/.test(token)) {
                const spoken = number(token);
                if (pieces.length > 0 && pendingSpace) {
                    pieces.push(" ");
                }
                pieces.push(phonemizeEnglish(spoken));
                pendingSpace = true;
                continue;
            }
            if (PUNCTUATION.includes(token)) {
                pieces.push(token);
                pendingSpace = true;
                continue;
            }
            if (token === "'" || token === "-" || token === "/") {
                pendingSpace = true;
                continue;
            }
            pendingSpace = true;
        }
        return pieces.join("").trim();
    };

export { phonemizeEnglish };
