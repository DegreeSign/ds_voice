import type { CloneVoiceOptions, VoiceClone } from "./types";
/** Record a reference and return its MOSS codes. */
declare const cloneVoice: (options?: CloneVoiceOptions) => Promise<VoiceClone>;
export { cloneVoice };
