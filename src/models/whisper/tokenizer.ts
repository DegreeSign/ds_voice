// Byte level BPE decoding.
import { checkpointKey } from "../specs";
import type { RawTokenizer, StoreModelOptions, Tokenizer } from "../../tensor/types";
const
    /** Printable byte ranges shared by the GPT-2 table. */
    isPrintableByte = (b: number): boolean => {
        if (b >= 33 && b <= 126) {
            return true;
        }
        if (b >= 161 && b <= 172) {
            return true;
        }
        return b >= 174 && b <= 255;
    },
    /** Build the GPT-2 byte to unicode table, indexed by byte. */
    bytesToUnicode = (): string[] => {
        const table: string[] = new Array<string>(256);
        let extra = 0;
        for (let b = 0; b < 256; b += 1) {
            if (isPrintableByte(b)) {
                table[b] = String.fromCharCode(b);
            } else {
                table[b] = String.fromCharCode(256 + extra);
                extra += 1;
            }
        }
        return table;
    },
    /** Load the tokenizer for a model. */
    loadTokenizer = async (options: StoreModelOptions): Promise<Tokenizer> => {
        const text = await options.store.readText(
            checkpointKey({ checkpoint: options.checkpoint, file: "tokenizer.json" }),
        );
        if (text === null) {
            throw new Error("missing tokenizer.json");
        }
        return createTokenizer(JSON.parse(text) as RawTokenizer);
    },
    /** Create a tokenizer from parsed data. */
    createTokenizer = (raw: RawTokenizer): Tokenizer => {
        const
            idToToken = new Map<number, string>(),
            special = new Set<number>();
        for (const [token, id] of Object.entries(raw.model.vocab)) {
            idToToken.set(id, token);
        }
        for (const entry of raw.added_tokens ?? []) {
            idToToken.set(entry.id, entry.content);
            if (entry.special) {
                special.add(entry.id);
            }
        }
        const
            table = bytesToUnicode(),
            charToByte = new Map<string, number>();
        for (let b = 0; b < table.length; b += 1) {
            charToByte.set(table[b], b);
        }
        const
            encoder = new TextEncoder(),
            decoder = new TextDecoder(),
            tokenBytes = (id: number): Uint8Array | null => {
                const token = idToToken.get(id);
                if (token === undefined || special.has(id)) {
                    return null;
                }
                const out = new Uint8Array(encoder.encode(token).length);
                let index = 0;
                for (const char of token) {
                    const byte = charToByte.get(char);
                    if (byte === undefined) {
                        continue;
                    }
                    out[index] = byte;
                    index += 1;
                }
                return out.subarray(0, index);
            },
            tokenText = (id: number): string => {
                const bytes = tokenBytes(id);
                if (!bytes) {
                    return "";
                }
                return decoder.decode(bytes);
            },
            decode = (ids: number[]): string => {
                const parts: Uint8Array[] = [];
                let total = 0;
                for (const id of ids) {
                    const bytes = tokenBytes(id);
                    if (!bytes) {
                        continue;
                    }
                    parts.push(bytes);
                    total += bytes.length;
                }
                const merged = new Uint8Array(total);
                let offset = 0;
                for (const part of parts) {
                    merged.set(part, offset);
                    offset += part.length;
                }
                return decoder.decode(merged);
            };

        return {
            decode,
            tokenText,
            vocabSize: idToToken.size,
        };
    };

export { loadTokenizer };
