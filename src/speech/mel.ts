// Whisper mel frontend.
import { nextPow2, reflect } from "../tensor/math";
import type { ComputeMelOptions, FftPlan, MelResult } from "../tensor/types";
const
    /** Iterative radix-2 FFT with precomputed tables. */
    transform = (options: {
        plan: FftPlan;
        re: Float32Array;
        im: Float32Array;
        inverse: boolean;
    }): void => {
        const
            { plan, re, im, inverse } = options,
            { m, bitrev, cosTable, sinTable } = plan;
        for (let i = 0; i < m; i += 1) {
            const j = bitrev[i];
            if (j > i) {
                const
                    tr = re[i],
                    ti = im[i];
                re[i] = re[j];
                im[i] = im[j];
                re[j] = tr;
                im[j] = ti;
            }
        }
        for (let len = 2; len <= m; len <<= 1) {
            const
                half = len >> 1,
                step = m / len;
            for (let start = 0; start < m; start += len) {
                for (let k = 0; k < half; k += 1) {
                    const
                        table = k * step,
                        wr = cosTable[table],
                        wi = inverse ? -sinTable[table] : sinTable[table],
                        a = start + k,
                        b = a + half,
                        tr = re[b] * wr - im[b] * wi,
                        ti = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
        }
        if (inverse) {
            const inv = 1 / m;
            for (let i = 0; i < m; i += 1) {
                re[i] *= inv;
                im[i] *= inv;
            }
        }
    },
    /** Build a Bluestein plan for transform length n. */
    buildFftPlan = (n: number): FftPlan => {
        const
            m = nextPow2(2 * n - 1),
            chirpRe = new Float32Array(n),
            chirpIm = new Float32Array(n);
        for (let i = 0; i < n; i += 1) {
            const angle = (Math.PI * ((i * i) % (2 * n))) / n;
            chirpRe[i] = Math.cos(angle);
            chirpIm[i] = -Math.sin(angle);
        }
        const
            kernelRe = new Float32Array(m),
            kernelIm = new Float32Array(m);
        for (let i = 0; i < n; i += 1) {
            kernelRe[i] = chirpRe[i];
            kernelIm[i] = -chirpIm[i];
            if (i > 0) {
                kernelRe[m - i] = chirpRe[i];
                kernelIm[m - i] = -chirpIm[i];
            }
        }
        const
            cosTable = new Float32Array(m / 2 + 1),
            sinTable = new Float32Array(m / 2 + 1);
        for (let k = 0; k <= m / 2; k += 1) {
            const angle = (-2 * Math.PI * k) / m;
            cosTable[k] = Math.cos(angle);
            sinTable[k] = Math.sin(angle);
        }
        const
            bitrev = new Uint32Array(m),
            bits = Math.log2(m);
        for (let i = 0; i < m; i += 1) {
            let value = 0;
            for (let b = 0; b < bits; b += 1) {
                value = (value << 1) | ((i >> b) & 1);
            }
            bitrev[i] = value;
        }
        const plan: FftPlan = {
            n,
            m,
            chirpRe,
            chirpIm,
            kernelRe,
            kernelIm,
            cosTable,
            sinTable,
            bitrev,
        };
        transform({
            plan,
            re: kernelRe,
            im: kernelIm,
            inverse: false,
        });
        return plan;
    },
    /** Real transform of one frame, returning the power spectrum. */
    powerSpectrum = (options: {
        plan: FftPlan;
        frame: Float32Array;
        window: Float32Array;
        bins: number;
        scratchRe: Float32Array;
        scratchIm: Float32Array;
        convRe: Float32Array;
        convIm: Float32Array;
        kernelRe: Float32Array;
        kernelIm: Float32Array;
        out: Float32Array;
        outOffset: number;
        stride: number;
    }): void => {
        const
            { plan, frame, window, bins } = options,
            n = plan.n,
            m = plan.m,
            { scratchRe, scratchIm } = options;
        scratchRe.fill(0);
        scratchIm.fill(0);
        for (let i = 0; i < n; i += 1) {
            const value = frame[i] * window[i];
            scratchRe[i] = value * plan.chirpRe[i];
            scratchIm[i] = value * plan.chirpIm[i];
        }
        transform({
            plan,
            re: scratchRe,
            im: scratchIm,
            inverse: false,
        });
        const
            convRe = options.convRe,
            convIm = options.convIm;
        convRe.fill(0);
        convIm.fill(0);
        for (let i = 0; i < m; i += 1) {
            const
                ar = scratchRe[i],
                ai = scratchIm[i],
                br = plan.kernelRe[i],
                bi = plan.kernelIm[i];
            convRe[i] = ar * br - ai * bi;
            convIm[i] = ar * bi + ai * br;
        }
        transform({
            plan,
            re: convRe,
            im: convIm,
            inverse: true,
        });
        for (let k = 0; k < bins; k += 1) {
            const
                cr = convRe[k],
                ci = convIm[k],
                xr = cr * plan.chirpRe[k] - ci * plan.chirpIm[k],
                xi = cr * plan.chirpIm[k] + ci * plan.chirpRe[k];
            options.out[options.outOffset + k * options.stride] = xr * xr + xi * xi;
        }
    },
    /** Filterbank energies, mel major [mels, frames]. */
    melEnergies = (options: {
        filters: Float32Array;
        power: Float32Array;
        bins: number;
        mels: number;
        frames: number;
    }): Float32Array => {
        const
            { filters, power, bins, mels, frames } = options,
            out = new Float32Array(mels * frames);
        for (let m = 0; m < mels; m += 1) {
            const row = m * bins;
            for (let f = 0; f < frames; f += 1) {
                let sum = 0;
                for (let b = 0; b < bins; b += 1) {
                    sum += filters[row + b] * power[b * frames + f];
                }
                out[m * frames + f] = sum;
            }
        }
        return out;
    },
    /** Run the full log-mel pipeline for one chunk. */
    computeMel = (options: ComputeMelOptions): MelResult => {
        const
            { pcm, mel, filters } = options,
            capped = Math.min(pcm.length, mel.nSamples),
            useSamples = options.padToChunk === false
                ? Math.max(mel.nFft, capped)
                : mel.nSamples,
            padded = new Float32Array(useSamples),
            take = Math.min(pcm.length, useSamples);
        padded.set(pcm.subarray(0, take));
        const window = new Float32Array(mel.nFft);
        for (let i = 0; i < mel.nFft; i += 1) {
            window[i] =
                0.5 - 0.5 * Math.cos((2 * Math.PI * i) / mel.nFft);
        }
        const
            plan = buildFftPlan(mel.nFft),
            bins = mel.nFft / 2 + 1,
            totalFrames = 1 + Math.floor(useSamples / mel.hopLength),
            keptFrames = totalFrames - 1,
            power = new Float32Array(bins * keptFrames),
            scratchRe = new Float32Array(plan.m),
            scratchIm = new Float32Array(plan.m),
            convRe = new Float32Array(plan.m),
            convIm = new Float32Array(plan.m),
            frame = new Float32Array(mel.nFft),
            half = mel.nFft / 2;
        for (let f = 0; f < keptFrames; f += 1) {
            const start = f * mel.hopLength - half;
            for (let i = 0; i < mel.nFft; i += 1) {
                frame[i] = padded[reflect({ index: start + i, length: useSamples })];
            }
            powerSpectrum({
                plan,
                frame,
                window,
                bins,
                scratchRe,
                scratchIm,
                convRe,
                convIm,
                kernelRe: plan.kernelRe,
                kernelIm: plan.kernelIm,
                out: power,
                outOffset: f,
                stride: keptFrames,
            });
        }
        const
            melBins = mel.featureSize,
            energies = melEnergies({
                filters,
                power,
                bins,
                mels: melBins,
                frames: keptFrames,
            }),
            values = new Float32Array(melBins * keptFrames);
        let max = -Infinity;
        for (let i = 0; i < values.length; i += 1) {
            const value = Math.log10(Math.max(energies[i], 1e-10));
            values[i] = value;
            if (value > max) {
                max = value;
            }
        }
        const floor = max - 8;
        for (let i = 0; i < values.length; i += 1) {
            const clamped = Math.max(values[i], floor);
            values[i] = (clamped + 4) / 4;
        }
        return {
            values,
            frames: keptFrames,
            melBins,
            trueFrames: keptFrames,
        };
    };

export { buildFftPlan, powerSpectrum, melEnergies, computeMel };
