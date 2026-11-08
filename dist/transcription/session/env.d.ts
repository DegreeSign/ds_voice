import { Platform } from "../../tensor/core/types";
declare const 
/** Detect the runtime the package is executing in. */
detectPlatform: () => Platform, 
/** True when the microphone API is reachable. */
hasMicrophone: () => boolean;
export { detectPlatform, hasMicrophone };
