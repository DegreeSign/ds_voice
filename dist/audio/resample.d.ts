import type { DownmixOptions, ResampleAudioOptions } from "../tensor/types";
declare const 
/** Average interleaved channels into mono. */
downmix: (options: DownmixOptions) => Float32Array, 
/** Resample mono f32 audio to a new rate. */
resampleAudio: (options: ResampleAudioOptions) => Float32Array;
export { downmix, resampleAudio };
