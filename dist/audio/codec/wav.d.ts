import type { DecodedAudio, ReadWavOptions } from "../../tensor/core/types";
declare const 
/** Parse a WAV buffer into mono f32 audio. */
decodeWav: (options: ReadWavOptions) => DecodedAudio;
export { decodeWav };
