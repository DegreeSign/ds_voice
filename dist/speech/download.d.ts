import type { LoadVoiceOptions, VoiceId, VoiceSpec } from "./types";
declare const 
/** True when every file of a voice is stored. */
isVoiceCached: (voice: VoiceId) => Promise<boolean>, 
/** Remove every stored file of a voice. */
removeVoice: (voice: VoiceId) => Promise<void>;
/** Download a voice, keeping other cached voices. */
declare const loadVoice: (options: LoadVoiceOptions) => Promise<VoiceSpec>;
export { loadVoice, isVoiceCached, removeVoice };
