import type { VoiceId, VoiceAudio } from "../types";
/** Load Kokoro without synthesizing. */
declare const prepareKokoro: (options: {
    voice: VoiceId;
    load: (file: string) => Promise<Uint8Array>;
}) => Promise<void>;
/** Synthesize one line with the Kokoro voice. */
declare const synthesizeKokoro: (options: {
    voice: VoiceId;
    text: string;
    load: (file: string) => Promise<Uint8Array>;
}) => Promise<VoiceAudio>;
export { prepareKokoro, synthesizeKokoro };
