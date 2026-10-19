// Windowed sinc resampler.
import type { DownmixOptions, ResampleAudioOptions } from "../tensor/types";
const
    /** Normalized sinc. */
    sinc = (value: number): number => {
        if (value === 0) {
            return 1;
        }
        const scaled = Math.PI * value;
        return Math.sin(scaled) / scaled;
    },
    /** Hann window over the interval [-1, 1]. */
    hann = (value: number): number => {
        if (value < -1 || value > 1) {
            return 0;
        }
        return 0.5 + 0.5 * Math.cos(Math.PI * value);
    },
    /** Average interleaved channels into mono. */
    downmix = (options: DownmixOptions): Float32Array => {
        const { input, channels } = options;
        if (channels <= 1) {
            return input;
        }
        const
            frames = Math.floor(input.length / channels),
            out = new Float32Array(frames);
        for (let i = 0; i < frames; i += 1) {
            let sum = 0;
            for (let c = 0; c < channels; c += 1) {
                sum += input[i * channels + c];
            }
            out[i] = sum / channels;
        }
        return out;
    },
    /** Resample mono f32 audio to a new rate. */
    resampleAudio = (options: ResampleAudioOptions): Float32Array => {
        const { input, inputRate, outputRate } = options;
        if (inputRate === outputRate) {
            return input;
        }
        const
            ratio = outputRate / inputRate,
            outLength = Math.max(1, Math.ceil(input.length * ratio)),
            out = new Float32Array(outLength),
            half = 16,
            cutoff = Math.min(1, ratio) * 0.95;
        for (let i = 0; i < outLength; i += 1) {
            const
                center = i / ratio,
                base = Math.floor(center);
            let
                sum = 0,
                weight = 0;
            for (let k = -half + 1; k <= half; k += 1) {
                const index = base + k;
                if (index < 0 || index >= input.length) {
                    continue;
                }
                const
                    distance = center - index,
                    w = sinc(distance * cutoff) * hann(distance / half) * cutoff;
                sum += input[index] * w;
                weight += w;
            }
            out[i] = weight === 0 ? 0 : sum / weight;
        }
        return out;
    };

export { downmix, resampleAudio };
