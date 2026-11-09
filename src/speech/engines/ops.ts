// Shared numeric helpers for the speech synthesis engines.
import { wasmLinear } from "../../tensor/kernel/wasm";
const
    /** Matrix multiply with an optional bias row. */
    linear = (options: {
        input: Float32Array;
        rows: number;
        inDim: number;
        weight: Float32Array;
        outDim: number;
        bias?: Float32Array;
        out?: Float32Array;
    }): Float32Array => {
        const
            { input, rows, inDim, weight, outDim, bias } = options,
            out = options.out ?? new Float32Array(rows * outDim),
            fast = wasmLinear({
                input,
                weight,
                bias,
                rows,
                inDim,
                outDim,
                out,
            });
        if (fast) {
            return fast;
        }
        for (let row = 0; row < rows; row += 1) {
            const base = row * inDim;
            for (let unit = 0; unit < outDim; unit += 1) {
                const start = unit * inDim;
                let sum = bias ? bias[unit] : 0;
                for (let i = 0; i < inDim; i += 1) {
                    sum += input[base + i] * weight[start + i];
                }
                out[row * outDim + unit] = sum;
            }
        }
        return out;
    },
    /** Layer normalization over the channel axis of every frame. */
    layerNorm = (options: {
        input: Float32Array;
        frames: number;
        channels: number;
        weight?: Float32Array;
        bias?: Float32Array;
        eps?: number;
        out?: Float32Array;
    }): Float32Array => {
        const
            { input, frames, channels, weight, bias } = options,
            eps = options.eps ?? 1e-5,
            out = options.out ?? new Float32Array(input.length);
        for (let frame = 0; frame < frames; frame += 1) {
            const base = frame * channels;
            let mean = 0;
            for (let c = 0; c < channels; c += 1) {
                mean += input[base + c];
            }
            mean /= channels;
            let variance = 0;
            for (let c = 0; c < channels; c += 1) {
                const diff = input[base + c] - mean;
                variance += diff * diff;
            }
            const scale = 1 / Math.sqrt(variance / channels + eps);
            for (let c = 0; c < channels; c += 1) {
                let value = (input[base + c] - mean) * scale;
                if (weight) {
                    value *= weight[c];
                }
                if (bias) {
                    value += bias[c];
                }
                out[base + c] = value;
            }
        }
        return out;
    },
    /** GELU with the tanh approximation, in place. */
    geluInPlace = (values: Float32Array): void => {
        const factor = Math.sqrt(2 / Math.PI);
        for (let i = 0; i < values.length; i += 1) {
            const value = values[i];
            const cube = value * value * value;
            values[i] = 0.5 * value *
                (1 + Math.tanh(factor * (value + 0.044715 * cube)));
        }
    },
    /** Error function approximation. */
    erf = (value: number): number => {
        const
            sign = value < 0 ? -1 : 1,
            x = Math.abs(value),
            t = 1 / (1 + 0.3275911 * x),
            poly = t * (0.254829592 + t * (-0.284496736 + t * (
                1.421413741 + t * (-1.453152027 + t * 1.061405429)
            )));
        return sign * (1 - poly * Math.exp(-x * x));
    },
    /** GELU with the exact error function, in place. */
    geluErfInPlace = (values: Float32Array): void => {
        for (let i = 0; i < values.length; i += 1) {
            const value = values[i];
            values[i] = 0.5 * value *
                (1 + erf(value / Math.SQRT2));
        }
    };

export {
    geluErfInPlace,
    geluInPlace,
    layerNorm,
    linear,
};
