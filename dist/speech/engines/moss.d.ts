import { VoiceId } from "../types";
import type { VoiceAudio } from "../types";
/** Load MOSS without synthesizing. */
declare const prepareMoss: (options: {
    voice: VoiceId;
    load: (file: string) => Promise<Uint8Array>;
}) => Promise<void>;
/** Synthesize one line with the MOSS backbone. */
declare const synthesizeMoss: (options: {
    voice: VoiceId;
    text: string;
    load: (file: string) => Promise<Uint8Array>;
    maxFrames?: number;
    seed?: number;
    temperature?: number;
}) => Promise<VoiceAudio>;
export { prepareMoss, synthesizeMoss };
