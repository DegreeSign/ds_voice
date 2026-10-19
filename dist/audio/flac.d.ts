import type { DecodedAudio, ReadFlacOptions } from "../tensor/types";
declare const 
/** Decode a FLAC buffer into mono f32 audio. */
decodeFlac: (options: ReadFlacOptions) => DecodedAudio;
export { decodeFlac };
