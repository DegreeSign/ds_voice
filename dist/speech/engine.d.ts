import type { LoadVoiceOptions, SynthesizeVoiceOptions, VoiceAudio } from "./types";
declare const 
/** Synthesize text with the selected library voice. */
synthesize: (options: SynthesizeVoiceOptions) => Promise<VoiceAudio>, 
/** Load and parse the selected voice, without speaking. */
prepareVoice: (options: LoadVoiceOptions) => Promise<void>;
export { prepareVoice, synthesize };
