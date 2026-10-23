import type { MicrophoneCapture, MicrophoneCaptureHandle } from "../tensor/types";
declare const 
/** Start microphone capture, returning a handle. */
captureMicrophone: (options: MicrophoneCapture) => MicrophoneCaptureHandle;
export { captureMicrophone };
