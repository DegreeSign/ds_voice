import { VoiceId } from "./types";
import type { VoiceSpec } from "./types";
declare const 
/** Fixed line rendered for every preview clip. */
PREVIEW = "Hi, this is a voice preview. Ha ha ha!", 
/** Sample rate of the bundled preview clips. */
PREVIEW_RATE = 16000, 
/** Every voice in the library. */
VOICES: Record<VoiceId, VoiceSpec>, 
/** Every voice, in library order. */
listVoices: () => VoiceSpec[], 
/** One voice by id. */
getVoiceSpec: (voice: VoiceId) => VoiceSpec;
export { PREVIEW, PREVIEW_RATE, VOICES, listVoices, getVoiceSpec };
