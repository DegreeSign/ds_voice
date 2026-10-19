// Shared numeric kernels.
import { wasmAttention, wasmLinear, wasmSilu } from "./wasm";
import type {
    AttentionOptions,
    LayerNormOptions,
    LinearOptions,
} from "./types";
const
    /** Scaled complementary error function approximation. */
    erf = (x: number): number => {
        const
            sign = x < 0 ? -1 : 1,
            ax = Math.abs(x),
            t = 1 / (1 + 0.3275911 * ax),
            poly =
                ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t -
                    0.284496736) *
                    t +
                    0.254829592) *
                t;
        return sign * (1 - poly * Math.exp(-ax * ax));
    },
    /** Apply GELU in place. */
    geluInPlace = (options: { values: Float32Array; length: number }): void => {
        const
            { values, length } = options,
            inv = 1 / Math.SQRT2;
        for (let i = 0; i < length; i += 1) {
            const x = values[i];
            values[i] = 0.5 * x * (1 + erf(x * inv));
        }
    },
    /** Rectified linear unit in place. */
    reluInPlace = (options: {
        values: Float32Array;
        offset?: number;
        length?: number;
    }): void => {
        const
            offset = options.offset ?? 0,
            length = options.length ?? options.values.length - offset;
        for (let i = 0; i < length; i += 1) {
            const value = options.values[offset + i];
            options.values[offset + i] = value > 0 ? value : 0;
        }
    },
    /** SiLU activation in place. */
    siluInPlace = (options: {
        values: Float32Array;
        offset?: number;
        length?: number;
    }): void => {
        if (wasmSilu(options)) {
            return;
        }
        const
            offset = options.offset ?? 0,
            length = options.length ?? options.values.length - offset;
        for (let i = 0; i < length; i += 1) {
            const x = options.values[offset + i];
            options.values[offset + i] = x / (1 + Math.exp(-x));
        }
    },
    /** Elementwise sum returning a new array. */
    addArrays = (options: {
        a: Float32Array;
        b: Float32Array;
        scaleB?: number;
    }): Float32Array => {
        const
            { a, b } = options,
            scale = options.scaleB ?? 1,
            out = new Float32Array(a.length);
        for (let i = 0; i < a.length; i += 1) {
            out[i] = a[i] + b[i] * scale;
        }
        return out;
    },
    /** Elementwise add into the first array. */
    addInPlace = (options: {
        a: Float32Array;
        b: Float32Array;
        scaleB?: number;
    }): void => {
        const
            { a, b } = options,
            scale = options.scaleB ?? 1;
        for (let i = 0; i < a.length; i += 1) {
            a[i] += b[i] * scale;
        }
    },
    /** Linear layer row-major weight plus optional bias. */
    linearJs = (options: LinearOptions): Float32Array => {
        const
            { input, weight, bias, rows, inDim, outDim } = options,
            out = options.out ?? new Float32Array(rows * outDim);
        for (let i = 0; i < rows; i += 1) {
            const
                inRow = i * inDim,
                outRow = i * outDim;
            for (let j = 0; j < outDim; j += 1) {
                const wRow = j * inDim;
                let sum = bias ? bias[j] : 0;
                for (let p = 0; p < inDim; p += 1) {
                    sum += input[inRow + p] * weight[wRow + p];
                }
                out[outRow + j] = sum;
            }
        }
        return out;
    },
    /** Layer normalization over the last dimension. */
    layerNorm = (options: LayerNormOptions): Float32Array => {
        const
            { input, weight, bias, rows, dims } = options,
            eps = options.eps ?? 1e-5,
            out = options.out ?? new Float32Array(rows * dims);
        for (let r = 0; r < rows; r += 1) {
            const row = r * dims;
            let mean = 0;
            for (let i = 0; i < dims; i += 1) {
                mean += input[row + i];
            }
            mean /= dims;
            let variance = 0;
            for (let i = 0; i < dims; i += 1) {
                const diff = input[row + i] - mean;
                variance += diff * diff;
            }
            variance /= dims;
            const rstd = 1 / Math.sqrt(variance + eps);
            if (options.meanOut) {
                options.meanOut[r] = mean;
            }
            if (options.rstdOut) {
                options.rstdOut[r] = rstd;
            }
            for (let i = 0; i < dims; i += 1) {
                out[row + i] = (input[row + i] - mean) * rstd * weight[i] + bias[i];
            }
        }
        return out;
    },
    /** In place softmax over one contiguous row. */
    softmaxRow = (options: {
        values: Float32Array;
        offset: number;
        length: number;
    }): void => {
        const { values, offset, length } = options;
        let max = -Infinity;
        for (let i = 0; i < length; i += 1) {
            const v = values[offset + i];
            if (v > max) {
                max = v;
            }
        }
        let sum = 0;
        for (let i = 0; i < length; i += 1) {
            const e = Math.exp(values[offset + i] - max);
            values[offset + i] = e;
            sum += e;
        }
        const inv = 1 / sum;
        for (let i = 0; i < length; i += 1) {
            values[offset + i] *= inv;
        }
    },
    /** Query blocked attention with online softmax. */
    attentionJs = (options: AttentionOptions): Float32Array => {
        const
            { q, k, v, seqQ, seqK, headDim, heads } = options,
            scale = options.scale ?? 1 / Math.sqrt(headDim),
            dim = headDim * heads,
            out = options.out ?? new Float32Array(seqQ * dim),
            block = 64,
            scores = new Float32Array(block * seqK);
        for (let h = 0; h < heads; h += 1) {
            const headOffset = h * headDim;
            for (let start = 0; start < seqQ; start += block) {
                const rows = Math.min(block, seqQ - start);
                let max = -Infinity;
                for (let i = 0; i < rows; i += 1) {
                    const
                        qRow = (start + i) * dim + headOffset,
                        sRow = i * seqK;
                    for (let j = 0; j < seqK; j += 1) {
                        const kRow = j * dim + headOffset;
                        let sum = 0;
                        for (let d = 0; d < headDim; d += 1) {
                            sum += q[qRow + d] * k[kRow + d];
                        }
                        sum *= scale;
                        if (options.causal && j > start + i) {
                            sum = -Infinity;
                        }
                        scores[sRow + j] = sum;
                        if (sum > max) {
                            max = sum;
                        }
                    }
                }
                for (let i = 0; i < rows; i += 1) {
                    let sum = 0;
                    const sRow = i * seqK;
                    for (let j = 0; j < seqK; j += 1) {
                        const e = scores[sRow + j] === -Infinity
                            ? 0
                            : Math.exp(scores[sRow + j] - max);
                        scores[sRow + j] = e;
                        sum += e;
                    }
                    const
                        inv = 1 / sum,
                        outRow = (start + i) * dim + headOffset;
                    for (let d = 0; d < headDim; d += 1) {
                        out[outRow + d] = 0;
                    }
                    for (let j = 0; j < seqK; j += 1) {
                        const w = scores[sRow + j] * inv;
                        if (w === 0) {
                            continue;
                        }
                        const vRow = j * dim + headOffset;
                        for (let d = 0; d < headDim; d += 1) {
                            out[outRow + d] += w * v[vRow + d];
                        }
                    }
                }
            }
        }
        return out;
    },
    /** Linear layer, SIMD fast path with a pure JavaScript fallback. */
    linear = (options: LinearOptions): Float32Array => {
        return wasmLinear(options) ?? linearJs(options);
    },
    /** Multi-head attention, SIMD fast path with a fallback. */
    attention = (options: AttentionOptions): Float32Array => {
        return wasmAttention(options) ?? attentionJs(options);
    },
    /** Single query cross attention returning head averaged weights. */
    crossAttentionOne = (options: {
        q: Float32Array;
        k: Float32Array;
        v: Float32Array;
        heads: number;
        headDim: number;
        seqK: number;
        captureHeads?: number[];
        weightsOut?: Float32Array;
    }): Float32Array => {
        const
            { q, k, v, heads, headDim, seqK } = options,
            dim = heads * headDim,
            out = new Float32Array(dim),
            scale = 1 / Math.sqrt(headDim),
            capture = options.captureHeads;
        if (options.weightsOut) {
            options.weightsOut.fill(0);
        }
        let captured = 0;
        for (let h = 0; h < heads; h += 1) {
            const
                headOffset = h * headDim,
                scores = new Float32Array(seqK);
            let max = -Infinity;
            for (let j = 0; j < seqK; j += 1) {
                const kRow = j * dim + headOffset;
                let sum = 0;
                for (let d = 0; d < headDim; d += 1) {
                    sum += q[headOffset + d] * k[kRow + d];
                }
                sum *= scale;
                scores[j] = sum;
                if (sum > max) {
                    max = sum;
                }
            }
            let sum = 0;
            for (let j = 0; j < seqK; j += 1) {
                const e = Math.exp(scores[j] - max);
                scores[j] = e;
                sum += e;
            }
            const
                inv = 1 / sum,
                isCaptured = capture?.includes(h) ?? false;
            if (isCaptured && options.weightsOut) {
                captured += 1;
                for (let j = 0; j < seqK; j += 1) {
                    options.weightsOut[j] += scores[j] * inv;
                }
            }
            for (let d = 0; d < headDim; d += 1) {
                out[headOffset + d] = 0;
            }
            for (let j = 0; j < seqK; j += 1) {
                const
                    w = scores[j] * inv,
                    vRow = j * dim + headOffset;
                for (let d = 0; d < headDim; d += 1) {
                    out[headOffset + d] += w * v[vRow + d];
                }
            }
        }
        if (options.weightsOut && captured > 0) {
            const inv = 1 / captured;
            for (let j = 0; j < seqK; j += 1) {
                options.weightsOut[j] *= inv;
            }
        }
        return out;
    },
    /** Argument max over a range. */
    argmax = (options: {
        values: Float32Array;
        start?: number;
        end?: number;
    }): number => {
        const
            { values } = options,
            start = options.start ?? 0,
            end = options.end ?? values.length;
        let
            best = start,
            bestValue = -Infinity;
        for (let i = start; i < end; i += 1) {
            if (values[i] > bestValue) {
                bestValue = values[i];
                best = i;
            }
        }
        return best;
    };

export {
    geluInPlace,
    reluInPlace,
    siluInPlace,
    addArrays,
    addInPlace,
    linear,
    layerNorm,
    softmaxRow,
    attention,
    crossAttentionOne,
    argmax,
};
