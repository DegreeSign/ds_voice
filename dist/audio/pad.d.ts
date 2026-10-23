import type { PadCenterOptions, PadCenterResult, PadWindowOptions } from "../tensor/types";
declare const 
/** Centre a buffer in zero padding. */
padCenter: (options: PadCenterOptions) => PadCenterResult, 
/** Pad a short window up to the minimum length. */
padWindow: (options: PadWindowOptions) => PadCenterResult;
export { padCenter, padWindow };
