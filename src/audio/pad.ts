// Symmetric silence padding.
import type {
    PadCenterOptions,
    PadCenterResult,
    PadWindowOptions,
} from "../tensor/types";
const
    /** Shortest window handed to a model. */
    MIN_WINDOW_SECONDS = 1,
    /** Centre a buffer in zero padding. */
    padCenter = (options: PadCenterOptions): PadCenterResult => {
        const { pcm, minSamples } = options;
        if (pcm.length >= minSamples) {
            return { pcm, leadSamples: 0 };
        }
        const
            lead = Math.floor((minSamples - pcm.length) / 2),
            padded = new Float32Array(minSamples);
        padded.set(pcm, lead);
        return { pcm: padded, leadSamples: lead };
    },
    /** Pad a short window up to the minimum length. */
    padWindow = (options: PadWindowOptions): PadCenterResult => padCenter({
        pcm: options.pcm,
        minSamples: Math.round(MIN_WINDOW_SECONDS * options.samplingRate),
    });

export { padCenter, padWindow };
