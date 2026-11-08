import type { VoiceAudio, VoiceId } from "../types";
/** Load Soprano without synthesizing. */
declare const prepareSoprano: (options: {
    voice: VoiceId;
    load: (file: string) => Promise<Uint8Array>;
}) => Promise<void>;
/** Synthesize one line with the Soprano backbone. */
declare const synthesizeSoprano: (options: {
    voice: VoiceId;
    text: string;
    load: (file: string) => Promise<Uint8Array>;
    maxTokens?: number;
}) => Promise<VoiceAudio>;
export { prepareSoprano, synthesizeSoprano };
