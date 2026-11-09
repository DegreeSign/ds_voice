import type { LoadVoiceOptions, VoiceCloneProgress, VoiceId, VoiceSpec } from "./types";
declare const 
/** True when every file of a voice is stored. */
isVoiceCached: (voice: VoiceId) => Promise<boolean>, 
/** Remove every stored file of a voice. */
removeVoice: (voice: VoiceId) => Promise<void>;
/** Download a voice, keeping other cached voices. */
declare const loadVoice: (options: LoadVoiceOptions) => Promise<VoiceSpec>;
declare const 
/** Download the clone codec when missing. */
loadCloneFiles: (options: {
    weightsBaseUrl?: string;
    onProgress?: (progress: VoiceCloneProgress) => void;
}) => Promise<void>, 
/** Read the stored clone codec data. */
readCloneWeights: () => Promise<Uint8Array>;
export { loadVoice, isVoiceCached, removeVoice, loadCloneFiles, readCloneWeights, };
