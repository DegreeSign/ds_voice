import type { VoiceAudio } from "../types";
/** Apply pitch, rate and gain to one synthesized clip. */
declare const adjustVoiceAudio: (options: {
    audio: VoiceAudio;
    pitch?: number;
    rate?: number;
    gain?: number;
}) => VoiceAudio;
export { adjustVoiceAudio };
