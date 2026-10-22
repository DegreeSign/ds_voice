import type { ModelStore, Voice } from "../tensor/types";
declare const 
/** Create the voice functions for one store. */
createVoice: (input?: {
    store?: ModelStore;
}) => Voice;
/** Shared voice bound to the platform store. */
declare const getVoice: () => Voice;
export { createVoice, getVoice };
