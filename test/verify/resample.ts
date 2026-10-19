// Windowed sinc resampler.
import type { ResampleOptions } from "./types";
const
    /** Normalized sinc. */
    sinc = (x: number): number => {
        if (x === 0) {
            return 1;
        }
        const pix = Math.PI * x;
        return Math.sin(pix) / pix;
    },
    /** Hann window over the interval [-1, 1]. */
    hann = (x: number): number => {
        if (x < -1 || x > 1) {
            return 0;
        }
        return 0.5 + 0.5 * Math.cos(Math.PI * x);
    },
    /** Resample mono f32 audio to a new rate. */
    resample = (options: ResampleOptions): Float32Array => {
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

export { resample };
