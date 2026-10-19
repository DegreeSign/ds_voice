import type { DecodedAudio, ReadWavOptions } from "../tensor/types";
declare const 
/** Parse a WAV buffer into mono f32 audio. */
decodeWav: (options: ReadWavOptions) => DecodedAudio;
export { decodeWav };
