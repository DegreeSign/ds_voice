// SenseVoice kaldi fbank features.
// Mirrors funasr wav frontend.
import { buildFftPlan, melEnergies, powerSpectrum } from "../../speech/mel";
import { nextPow2 } from "../../tensor/math";
import type {
    SenseVoiceFeatureOptions,
    SenseVoiceFeatureResult,
} from "../../tensor/types";
const
    /** Kaldi default preemphasis coefficient. */
    PREEMPHASIS = 0.97,
    /** Kaldi default low cutoff in hertz. */
    MEL_LOW = 20,
    /** Kaldi mel energy floor, roughly the f32 epsilon. */
    MEL_FLOOR = 1.1920928955078125e-7,
    /** HTK mel scale from hertz. */
    hzToMel = (freq: number): number => {
        return 1127 * Math.log(1 + freq / 700);
    },
    /** HTK mel scale to hertz. */
    melToHz = (mel: number): number => {
        return 700 * (Math.exp(mel / 1127) - 1);
    },
    /** Build the HTK mel filterbank, mels by bins. */
    htkFilterbank = (options: {
        sampleRate: number;
        fftSize: number;
        numMels: number;
    }): Float32Array => {
        const
            { sampleRate, fftSize, numMels } = options,
            bins = fftSize / 2 + 1,
            melLow = hzToMel(MEL_LOW),
            melHigh = hzToMel(sampleRate / 2),
            melStep = (melHigh - melLow) / (numMels + 1),
            binWidth = sampleRate / fftSize,
            filters = new Float32Array(numMels * bins);
        for (let m = 0; m < numMels; m += 1) {
            const
                left = melToHz(melLow + m * melStep),
                center = melToHz(melLow + (m + 1) * melStep),
                right = melToHz(melLow + (m + 2) * melStep);
            for (let b = 0; b < bins; b += 1) {
                const hz = b * binWidth;
                let weight = 0;
                if (hz > left && hz < right) {
                    weight = hz <= center
                        ? (hz - left) / (center - left)
                        : (right - hz) / (right - center);
                }
                filters[m * bins + b] = weight < 0 ? 0 : weight;
            }
        }
        return filters;
    },
    /** Symmetric kaldi hamming window. */
    hammingWindow = (length: number): Float32Array => {
        const out = new Float32Array(length);
        if (length <= 1) {
            out.fill(1);
            return out;
        }
        const step = (2 * Math.PI) / (length - 1);
        for (let i = 0; i < length; i += 1) {
            out[i] = 0.54 - 0.46 * Math.cos(step * i);
        }
        return out;
    },
    /** Run the kaldi fbank, LFR and CMVN pipeline. */
    computeSenseVoiceFeatures = (
        options: SenseVoiceFeatureOptions,
    ): SenseVoiceFeatureResult => {
        const
            { pcm, frontend } = options,
            { winLength, hopLength, numMels, lfrM, lfrN } = frontend;
        if (pcm.length < winLength) {
            return { values: new Float32Array(0), frames: 0 };
        }
        const
            values = pcm.length,
            frames = 1 + Math.floor((values - winLength) / hopLength),
            fftSize = nextPow2(winLength),
            bins = fftSize / 2 + 1,
            dInput = lfrM * numMels,
            filters = htkFilterbank({
                sampleRate: frontend.samplingRate,
                fftSize,
                numMels,
            }),
            window = new Float32Array(fftSize);
        window.set(hammingWindow(winLength));
        const
            plan = buildFftPlan(fftSize),
            power = new Float32Array(bins * frames),
            scratchRe = new Float32Array(plan.m),
            scratchIm = new Float32Array(plan.m),
            convRe = new Float32Array(plan.m),
            convIm = new Float32Array(plan.m),
            frame = new Float32Array(fftSize),
            work = new Float32Array(winLength);
        for (let t = 0; t < frames; t += 1) {
            const base = t * hopLength;
            for (let i = 0; i < winLength; i += 1) {
                work[i] = pcm[base + i] * frontend.upscale;
            }
            let sum = 0;
            for (let i = 0; i < winLength; i += 1) {
                sum += work[i];
            }
            const mean = sum / winLength;
            for (let i = 0; i < winLength; i += 1) {
                work[i] -= mean;
            }
            for (let i = winLength - 1; i >= 1; i -= 1) {
                work[i] -= PREEMPHASIS * work[i - 1];
            }
            work[0] *= 1 - PREEMPHASIS;
            frame.fill(0);
            for (let i = 0; i < winLength; i += 1) {
                frame[i] = work[i];
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
                outOffset: t,
                stride: frames,
            });
        }
        const
            energies = melEnergies({
                filters,
                power,
                bins,
                mels: numMels,
                frames,
            }),
            mel = new Float32Array(frames * numMels);
        for (let m = 0; m < numMels; m += 1) {
            for (let t = 0; t < frames; t += 1) {
                mel[t * numMels + m] = Math.log(Math.max(energies[m * frames + t], MEL_FLOOR));
            }
        }
        const
            leftPad = (lfrM - 1) >> 1,
            stacked = (frames + lfrN - 1) / lfrN,
            outFrames = Math.floor(stacked),
            features = new Float32Array(outFrames * dInput);
        for (let i = 0; i < outFrames; i += 1) {
            const outRow = i * dInput;
            for (let k = 0; k < lfrM; k += 1) {
                const padded = i * lfrN + k;
                let source = padded < leftPad ? 0 : padded - leftPad;
                if (source >= frames) {
                    source = frames - 1;
                }
                const
                    srcRow = source * numMels,
                    dstRow = outRow + k * numMels;
                for (let m = 0; m < numMels; m += 1) {
                    const
                        dim = k * numMels + m,
                        value = mel[srcRow + m];
                    features[dstRow + m] =
                        (value + frontend.shift[dim]) * frontend.scale[dim];
                }
            }
        }
        return { values: features, frames: outFrames };
    };

export { computeSenseVoiceFeatures };
