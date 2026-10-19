// Greedy CTC decoding.
import { argmax } from "../../tensor/kernels";
import type { CtcDecodeOptions, CtcDecodeResult } from "../../tensor/types";

/** Decode one frame by frame argmax stream. */
const decodeCtc = (options: CtcDecodeOptions): CtcDecodeResult => {
    const
        { logits, frames, vocabSize, blankId } = options,
        ids: number[] = [],
        positions: number[] = [];
    let previous = -1;
    for (let t = 0; t < frames; t += 1) {
        const
            row = t * vocabSize,
            best = argmax({
                values: logits,
                start: row,
                end: row + vocabSize,
            }) - row;
        if (best === previous) {
            continue;
        }
        previous = best;
        if (best !== blankId) {
            ids.push(best);
            positions.push(t);
        }
    }
    return { ids, frames: positions };
};

export { decodeCtc };
