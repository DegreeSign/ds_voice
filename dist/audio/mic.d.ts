import type { MicrophoneCapture } from "../tensor/types";
declare const 
/** Start microphone capture, returning a stop handle. */
captureMicrophone: (options: MicrophoneCapture) => (() => Promise<void>);
export { captureMicrophone };
