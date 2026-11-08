import type { DictationController, DictationOptions } from "./types";
declare const 
/** True when the microphone API is reachable. */
isDictationSupported: () => boolean, 
/** Wire one UI to dictation. */
createDictation: (options: DictationOptions) => DictationController;
export { createDictation, isDictationSupported };
