// Voice post-processing for pitch, rate and gain.
import type { VoiceAudio } from "../types";
const
    /** Largest analysis frame in samples. */
    MAX_FRAME = 2048,
    /** Smallest analysis frame in samples. */
    MIN_FRAME = 256,
    /** Frame length for one sample rate. */
    frameLength = (sampleRate: number): number => {
        const frame = Math.max(
            MIN_FRAME,
            Math.min(MAX_FRAME, Math.round(sampleRate * 0.04)),
        );
        return frame % 2 === 0 ? frame : frame + 1;
    },
    /** Clamp one value. */
    clamp = (value: number, min: number, max: number): number =>
        value < min ? min : value > max ? max : value,
    /** Linear resampling by one factor. */
    resample = (input: Float32Array, factor: number): Float32Array => {
        const
            length = Math.max(1, Math.round(input.length / factor)),
            out = new Float32Array(length);
        if (Math.abs(factor - 1) < 1e-6) {
            out.set(input.subarray(0, length));
            return out;
        }
        for (let i = 0; i < length; i += 1) {
            const
                position = i * factor,
                lower = Math.floor(position),
                upper = Math.min(input.length - 1, lower + 1),
                fraction = position - lower;
            out[i] = input[lower] * (1 - fraction) +
                input[upper] * fraction;
        }
        return out;
    },
    /** Overlap-add time stretch, factor is output over input length. */
    timeStretch = (options: {
        input: Float32Array;
        factor: number;
        sampleRate: number;
    }): Float32Array => {
        const
            { input, factor, sampleRate } = options,
            frame = frameLength(sampleRate),
            hop = frame / 2,
            outputLength = Math.max(1, Math.round(input.length * factor));
        if (input.length < frame * 2) {
            return input.slice();
        }
        const
            out = new Float32Array(outputLength + frame),
            window = new Float32Array(frame),
            analysisHop = hop / factor,
            search = Math.max(1, Math.round(hop / 2));
        for (let i = 0; i < frame; i += 1) {
            window[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / frame));
        }
        let previousStart = 0;
        for (
            let frameIndex = 0, outStart = 0;
            outStart < outputLength;
            frameIndex += 1, outStart += hop
        ) {
            const ideal = Math.round(frameIndex * analysisHop);
            let start = ideal;
            if (frameIndex === 0) {
                start = 0;
            } else {
                const
                    referenceStart = Math.min(
                        input.length - hop,
                        previousStart + hop,
                    ),
                    low = Math.max(0, ideal - search),
                    high = Math.min(input.length - hop, ideal + search);
                let best = -Infinity;
                for (let candidate = low; candidate <= high; candidate += 1) {
                    let dot = 0;
                    let left = 0;
                    let right = 0;
                    for (let i = 0; i < hop; i += 1) {
                        const
                            a = input[referenceStart + i],
                            b = input[candidate + i];
                        dot += a * b;
                        left += a * a;
                        right += b * b;
                    }
                    const score = dot / Math.sqrt(left * right + 1e-9);
                    if (score > best) {
                        best = score;
                        start = candidate;
                    }
                }
            }
            for (let i = 0; i < frame; i += 1) {
                const source = start + i;
                if (source >= input.length) {
                    break;
                }
                out[outStart + i] += input[source] * window[i];
            }
            previousStart = start;
        }
        return out.slice(0, outputLength);
    };

/** Apply pitch, rate and gain to one synthesized clip. */
const adjustVoiceAudio = (options: {
    audio: VoiceAudio;
    pitch?: number;
    rate?: number;
    gain?: number;
}): VoiceAudio => {
    const
        { audio } = options,
        pitch = clamp(options.pitch ?? 0, -12, 12),
        rate = clamp(options.rate ?? 1, 0.25, 4),
        gain = clamp(options.gain ?? 1, 0.1, 4);
    if (pitch === 0 && rate === 1 && gain === 1) {
        return audio;
    }
    const pitchFactor = 2 ** (pitch / 12);
    let pcm = audio.pcm;
    if (Math.abs(pitchFactor - 1) > 1e-6) {
        pcm = resample(pcm, pitchFactor);
    }
    const stretch = pitchFactor / rate;
    if (Math.abs(stretch - 1) > 1e-6) {
        pcm = timeStretch({
            input: pcm,
            factor: stretch,
            sampleRate: audio.sampleRate,
        });
    }
    if (gain !== 1 || pcm === audio.pcm) {
        pcm = new Float32Array(pcm);
        for (let i = 0; i < pcm.length; i += 1) {
            pcm[i] *= gain;
        }
    }
    for (let i = 0; i < pcm.length; i += 1) {
        pcm[i] = clamp(pcm[i], -1, 1);
    }
    return { pcm, sampleRate: audio.sampleRate };
};

export { adjustVoiceAudio };
