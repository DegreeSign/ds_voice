import type { SessionOptions, TranscribeSession } from "../tensor/types";
declare const 
/** Create one streaming session. */
createSession: (input: SessionOptions) => TranscribeSession;
export { createSession };
