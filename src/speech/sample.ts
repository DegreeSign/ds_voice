// Decode the bundled voice preview clips on demand.
import { female_1, female_2, male_1 } from "./samples";
import { getVoiceSpec } from "./voices";
import { VoiceId } from "./types";
import type { VoiceSample } from "./types";
const
    /** Base64 clip of every voice. */
    CLIPS: Record<VoiceId, string> = {
        [VoiceId.Female1]: female_1,
        [VoiceId.Female2]: female_2,
        [VoiceId.Male]: male_1,
    },
    /** Decoded clips, filled on first use. */
    cache = new Map<VoiceId, Uint8Array>(),
    /** Decode one base64 WAV. */
    decode = (base64: string): Uint8Array => {
        const
            text = atob(base64),
            bytes = new Uint8Array(text.length);
        for (let i = 0; i < text.length; i += 1) {
            bytes[i] = text.charCodeAt(i);
        }
        return bytes;
    },
    /** Preview clip of one voice. */
    getVoiceSample = (voice: VoiceId): VoiceSample => {
        const spec = getVoiceSpec(voice);
        let bytes = cache.get(voice);
        if (!bytes) {
            bytes = decode(CLIPS[voice]);
            cache.set(voice, bytes);
        }
        return {
            voice,
            bytes,
            sampleRate: spec.previewRate,
            mimeType: "audio/wav",
        };
    };

export { getVoiceSample };
