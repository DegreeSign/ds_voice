// Runtime environment detection.
import { Platform } from "../tensor/types";
const
    /** Detect the runtime the package is executing in. */
    detectPlatform = (): Platform => {
        if (
            typeof window !== "undefined" &&
            typeof window.document !== "undefined"
        ) {
            return Platform.Browser;
        }
        if (
            typeof process !== "undefined" &&
            typeof process.versions?.node === "string"
        ) {
            return Platform.Node;
        }
        return Platform.Browser;
    },
    /** True when the microphone API is reachable. */
    hasMicrophone = (): boolean => {
        return (
            typeof navigator !== "undefined" &&
            typeof navigator.mediaDevices?.getUserMedia === "function"
        );
    };

export { detectPlatform, hasMicrophone };
