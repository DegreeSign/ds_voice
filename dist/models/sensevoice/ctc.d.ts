import type { CtcDecodeOptions, CtcDecodeResult } from "../../tensor/core/types";
/** Decode one frame by frame argmax stream. */
declare const decodeCtc: (options: CtcDecodeOptions) => CtcDecodeResult;
export { decodeCtc };
