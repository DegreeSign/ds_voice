import type { DecodedAudio, ReadMp3Options } from "../tensor/types";
declare const 
/** Decode a MP3 buffer into mono f32 audio. */
decodeMp3: (options: ReadMp3Options) => Promise<DecodedAudio>;
export { decodeMp3 };
