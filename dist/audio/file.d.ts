import type { DecodeBytesOptions, DecodeSourceOptions, DecodedAudio } from "../tensor/types";
declare const 
/** Decode raw bytes for the current platform. */
decodeBytes: (options: DecodeBytesOptions) => Promise<DecodedAudio>, 
/** Decode a URL, blob or filesystem path. */
decodeSource: (options: DecodeSourceOptions) => Promise<DecodedAudio>;
export { decodeBytes, decodeSource };
