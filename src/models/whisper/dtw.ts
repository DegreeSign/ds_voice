// Word level DTW alignment.
// Ported from transformers 4.44.0.
import { reflect } from "../../tensor/math";
import type { DtwOptions, DtwResult } from "../../tensor/types";
const
    /** Median filter along the last dimension with reflect padding. */
    medianFilter = (options: {
        values: Float32Array;
        rows: number;
        cols: number;
        width: number;
    }): Float32Array => {
        const { values, rows, cols, width } = options;
        if (width <= 0 || width % 2 !== 1) {
            throw new Error("median width must be odd");
        }
        const pad = (width - 1) / 2;
        if (cols <= pad) {
            return values;
        }
        const
            out = new Float32Array(rows * cols),
            window = new Float64Array(width);
        for (let r = 0; r < rows; r += 1) {
            const row = r * cols;
            for (let c = 0; c < cols; c += 1) {
                for (let k = 0; k < width; k += 1) {
                    window[k] = values[row + reflect({ index: c - pad + k, length: cols })];
                }
                const sorted = [...window].sort((a, b) => a - b);
                out[row + c] = sorted[pad];
            }
        }
        return out;
    },
    /** Dynamic time warping over one cost matrix. */
    dynamicTimeWarping = (options: {
        matrix: Float64Array;
        rows: number;
        cols: number;
    }): {
        textIndices: Int32Array;
        timeIndices: Int32Array;
    } => {
        const
            { matrix, rows, cols } = options,
            width = cols + 1,
            cost = new Float64Array((rows + 1) * width).fill(Infinity),
            trace = new Int8Array((rows + 1) * width).fill(-1);
        cost[0] = 0;
        for (let j = 1; j <= cols; j += 1) {
            for (let i = 1; i <= rows; i += 1) {
                const
                    c0 = cost[(i - 1) * width + (j - 1)],
                    c1 = cost[(i - 1) * width + j],
                    c2 = cost[i * width + (j - 1)];
                let
                    c: number,
                    t: number;
                if (c0 < c1 && c0 < c2) {
                    c = c0;
                    t = 0;
                } else if (c1 < c0 && c1 < c2) {
                    c = c1;
                    t = 1;
                } else {
                    c = c2;
                    t = 2;
                }
                cost[i * width + j] = matrix[(i - 1) * cols + (j - 1)] + c;
                trace[i * width + j] = t;
            }
        }
        let
            i = rows,
            j = cols;
        const
            text: number[] = [],
            time: number[] = [];
        while (i > 0 || j > 0) {
            text.push(i - 1);
            time.push(j - 1);
            const t = i === 0 ? 2 : j === 0 ? 1 : trace[i * width + j];
            if (t === 0) {
                i -= 1;
                j -= 1;
            } else if (t === 1) {
                i -= 1;
            } else if (t === 2) {
                j -= 1;
            } else {
                throw new Error("bad dtw trace");
            }
        }
        text.reverse();
        time.reverse();
        return {
            textIndices: Int32Array.from(text),
            timeIndices: Int32Array.from(time),
        };
    },
    /** Build the alignment times from captured cross attention. */
    computeDtw = (options: DtwOptions): DtwResult => {
        const
            { attention, frameCount } = options,
            rows = attention.length,
            heads = attention[0]?.length ?? 0,
            cols = frameCount;
        if (rows === 0 || heads === 0) {
            return {
                textIndices: new Int32Array(0),
                timeIndices: new Int32Array(0),
                times: new Float64Array(0),
            };
        }
        const headRows: Float32Array[] = [];
        for (let h = 0; h < heads; h += 1) {
            headRows.push(new Float32Array(rows * cols));
        }
        for (let t = 0; t < rows; t += 1) {
            const row = attention[t];
            for (let h = 0; h < heads; h += 1) {
                headRows[h].set(row[h].subarray(0, cols), t * cols);
            }
        }
        const
            width = options.medianWidth ?? 7,
            averaged = new Float32Array(rows * cols);
        for (let h = 0; h < heads; h += 1) {
            const values = headRows[h];
            for (let c = 0; c < cols; c += 1) {
                let mean = 0;
                for (let t = 0; t < rows; t += 1) {
                    mean += values[t * cols + c];
                }
                mean /= rows;
                let variance = 0;
                for (let t = 0; t < rows; t += 1) {
                    const diff = values[t * cols + c] - mean;
                    variance += diff * diff;
                }
                variance /= rows;
                const std = Math.sqrt(variance);
                for (let t = 0; t < rows; t += 1) {
                    values[t * cols + c] = (values[t * cols + c] - mean) / std;
                }
            }
            const filtered = medianFilter({ values, rows, cols, width });
            for (let i = 0; i < averaged.length; i += 1) {
                averaged[i] += filtered[i] / heads;
            }
        }
        const matrix = new Float64Array(rows * cols);
        for (let i = 0; i < matrix.length; i += 1) {
            matrix[i] = -averaged[i];
        }
        const
            { textIndices, timeIndices } = dynamicTimeWarping({
                matrix,
                rows,
                cols,
            }),
            jumpTimes: number[] = [];
        for (let k = 0; k < textIndices.length; k += 1) {
            if (k === 0 || textIndices[k] !== textIndices[k - 1]) {
                jumpTimes.push(timeIndices[k] * 0.02);
            }
        }
        const times = new Float64Array(rows);
        for (let i = 1; i < rows; i += 1) {
            times[i] = jumpTimes[i - 1] ?? jumpTimes[jumpTimes.length - 1] ?? 0;
        }
        return { textIndices, timeIndices, times };
    };

export { computeDtw };
