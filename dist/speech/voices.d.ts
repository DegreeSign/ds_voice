import { VoiceId } from "./types";
import type { VoiceFile, VoiceSpec } from "./types";
declare const 
/** Fixed line rendered for every preview clip. */
PREVIEW = "Hi, this is a voice preview. Ha ha ha!", 
/** Sample rate of the bundled preview clips. */
PREVIEW_RATE = 16000, 
/** Encoder data used only by voice cloning. */
CLONE_FILES: VoiceFile[], 
/** Every voice in the library. */
VOICES: Record<VoiceId, VoiceSpec>, 
/** Every voice, in library order. */
listVoices: () => VoiceSpec[], 
/** One voice by id. */
getVoiceSpec: (voice: VoiceId) => VoiceSpec;
export { PREVIEW, PREVIEW_RATE, VOICES, CLONE_FILES, listVoices, getVoiceSpec };
