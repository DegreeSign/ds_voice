// Voice packs shipped inside the package instead of a model repository.
import { male_1 } from "./assets/voicePacks";
const
    /** Base64 chunks of every bundled pack. */
    PACKS: Record<string, readonly string[]> = {
        "kokoro-voice-am_michael.gguf": male_1,
    },
    /** Decoded packs, filled on first use. */
    cache = new Map<string, Uint8Array>(),
    /** Decode one base64 payload. */
    decode = (payload: string): Uint8Array => {
        const
            text = atob(payload),
            bytes = new Uint8Array(text.length);
        for (let i = 0; i < text.length; i += 1) {
            bytes[i] = text.charCodeAt(i);
        }
        return bytes;
    },
    /** Bytes of one bundled voice file, or undefined. */
    getBundledVoiceFile = (file: string): Uint8Array | undefined => {
        const chunks = PACKS[file];
        if (!chunks) {
            return undefined;
        }
        let bytes = cache.get(file);
        if (!bytes) {
            bytes = decode(chunks.join(""));
            cache.set(file, bytes);
        }
        return bytes;
    },
    /** True when the file ships with the package. */
    isBundledVoiceFile = (file: string): boolean => file in PACKS;

export { getBundledVoiceFile, isBundledVoiceFile };
