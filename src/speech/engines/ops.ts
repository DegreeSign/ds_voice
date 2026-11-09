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
    /** One dimensional convolution, frame major, stride and dilation. */
    conv1d = (options: {
        input: Float32Array;
        frames: number;
        channels: number;
        weight: Float32Array;
        bias?: Float32Array;
        outChannels: number;
        kernel: number;
        stride?: number;
        padding?: number;
        dilation?: number;
    }): Float32Array => {
        const
            { input, frames, channels, weight, bias, outChannels } = options,
            { kernel } = options,
            stride = options.stride ?? 1,
            padding = options.padding ?? 0,
            dilation = options.dilation ?? 1,
            span = dilation * (kernel - 1) + 1,
            outFrames = Math.floor(
                (frames + 2 * padding - span) / stride,
            ) + 1,
            width = channels * kernel,
            columns = new Float32Array(outFrames * width);
        for (let frame = 0; frame < outFrames; frame += 1) {
            const base = frame * stride - padding;
            for (let channel = 0; channel < channels; channel += 1) {
                const target = frame * width + channel * kernel;
                for (let k = 0; k < kernel; k += 1) {
                    const source = base + k * dilation;
                    if (source >= 0 && source < frames) {
                        columns[target + k] =
                            input[source * channels + channel];
                    }
                }
            }
        }
        return linear({
            input: columns,
            rows: outFrames,
            inDim: width,
            weight,
            bias,
            outDim: outChannels,
        });
    },
    /** Transposed convolution built from a matmul and a frame scatter. */
    convTranspose1d = (options: {
        input: Float32Array;
        frames: number;
        channels: number;
        weight: Float32Array;
        bias?: Float32Array;
        outChannels: number;
        kernel: number;
        stride: number;
        padding: number;
        outputPadding?: number;
    }): Float32Array => {
        const
            { input, frames, channels, weight, bias, outChannels } = options,
            { kernel, stride, padding } = options,
            outputPadding = options.outputPadding ?? 0,
            outFrames = (frames - 1) * stride - 2 * padding + kernel +
                outputPadding,
            width = outChannels * kernel,
            products = linear({
                input,
                rows: frames,
                inDim: channels,
                weight,
                outDim: width,
            }),
            out = new Float32Array(outFrames * outChannels);
        if (bias) {
            for (let frame = 0; frame < outFrames; frame += 1) {
                out.set(bias, frame * outChannels);
            }
        }
        for (let frame = 0; frame < frames; frame += 1) {
            for (let unit = 0; unit < outChannels; unit += 1) {
                const source = frame * width + unit * kernel;
                for (let k = 0; k < kernel; k += 1) {
                    const target = frame * stride - padding + k;
                    if (target >= 0 && target < outFrames) {
                        out[target * outChannels + unit] += products[source + k];
                    }
                }
            }
        }
        return out;
    },
    /** Depthwise transposed convolution over the channel axis. */
    depthwiseConvTranspose1d = (options: {
        input: Float32Array;
        frames: number;
        channels: number;
        weight: Float32Array;
        bias?: Float32Array;
        kernel: number;
        stride: number;
        padding: number;
        outputPadding?: number;
    }): Float32Array => {
        const
            { input, frames, channels, weight, bias } = options,
            { kernel, stride, padding } = options,
            outputPadding = options.outputPadding ?? 0,
            outFrames = (frames - 1) * stride - 2 * padding + kernel +
                outputPadding,
            out = new Float32Array(outFrames * channels);
        if (bias) {
            for (let frame = 0; frame < outFrames; frame += 1) {
                out.set(bias, frame * channels);
            }
        }
        for (let frame = 0; frame < frames; frame += 1) {
            const base = frame * channels;
            for (let channel = 0; channel < channels; channel += 1) {
                const value = input[base + channel];
                const start = channel * kernel;
                for (let k = 0; k < kernel; k += 1) {
                    const target = frame * stride - padding + k;
                    if (target >= 0 && target < outFrames) {
                        out[target * channels + channel] +=
                            value * weight[start + k];
                    }
                }
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
    /** Instance normalization over the frame axis of every channel. */
    instanceNorm = (options: {
        input: Float32Array;
        frames: number;
        channels: number;
        eps?: number;
    }): Float32Array => {
        const
            { input, frames, channels } = options,
            eps = options.eps ?? 1e-5,
            mean = new Float64Array(channels),
            spread = new Float64Array(channels),
            out = new Float32Array(input.length);
        for (let frame = 0; frame < frames; frame += 1) {
            const base = frame * channels;
            for (let c = 0; c < channels; c += 1) {
                mean[c] += input[base + c];
            }
        }
        for (let c = 0; c < channels; c += 1) {
            mean[c] /= frames;
        }
        for (let frame = 0; frame < frames; frame += 1) {
            const base = frame * channels;
            for (let c = 0; c < channels; c += 1) {
                const diff = input[base + c] - mean[c];
                spread[c] += diff * diff;
            }
        }
        for (let c = 0; c < channels; c += 1) {
            spread[c] = 1 / Math.sqrt(spread[c] / frames + eps);
        }
        for (let frame = 0; frame < frames; frame += 1) {
            const base = frame * channels;
            for (let c = 0; c < channels; c += 1) {
                out[base + c] = (input[base + c] - mean[c]) * spread[c];
            }
        }
        return out;
    },
    /** Apply an adaptive instance normalization in place. */
    adainInPlace = (options: {
        input: Float32Array;
        frames: number;
        channels: number;
        style: Float32Array;
        weight: Float32Array;
        bias: Float32Array;
    }): void => {
        const
            { input, frames, channels, style, weight, bias } = options,
            scale = linear({
                input: style,
                rows: 1,
                inDim: style.length,
                weight,
                bias,
                outDim: channels * 2,
            }),
            normed = instanceNorm({ input, frames, channels });
        for (let frame = 0; frame < frames; frame += 1) {
            const base = frame * channels;
            for (let c = 0; c < channels; c += 1) {
                const gamma = scale[c];
                const beta = scale[channels + c];
                input[base + c] = normed[base + c] +
                    normed[base + c] * gamma + beta;
            }
        }
    },
    /** Apply an adaptive layer normalization in place. */
    adalnInPlace = (options: {
        input: Float32Array;
        frames: number;
        channels: number;
        style: Float32Array;
        weight: Float32Array;
        bias: Float32Array;
    }): void => {
        const
            { input, frames, channels, style, weight, bias } = options,
            scale = linear({
                input: style,
                rows: 1,
                inDim: style.length,
                weight,
                bias,
                outDim: channels * 2,
            }),
            normed = layerNorm({ input, frames, channels });
        for (let frame = 0; frame < frames; frame += 1) {
            const base = frame * channels;
            for (let c = 0; c < channels; c += 1) {
                input[base + c] = normed[base + c] +
                    normed[base + c] * scale[c] + scale[channels + c];
            }
        }
    },
    /** Leaky rectified linear unit in place. */
    leakyInPlace = (options: {
        values: Float32Array;
        slope: number;
    }): void => {
        const { values, slope } = options;
        for (let i = 0; i < values.length; i += 1) {
            const value = values[i];
            values[i] = value >= 0 ? value : value * slope;
        }
    },
    /** Snake activation with one alpha per channel, in place. */
    snakeInPlace = (options: {
        values: Float32Array;
        frames: number;
        channels: number;
        alpha: Float32Array;
    }): void => {
        const { values, frames, channels, alpha } = options;
        for (let frame = 0; frame < frames; frame += 1) {
            const base = frame * channels;
            for (let c = 0; c < channels; c += 1) {
                const a = alpha[c];
                const sine = Math.sin(a * values[base + c]);
                values[base + c] += (sine * sine) / a;
            }
        }
    },
    /** Sigmoid of one value. */
    sigmoid = (value: number): number => 1 / (1 + Math.exp(-value)),
    /** SiLU activation in place. */
    siluInPlace = (values: Float32Array): void => {
        for (let i = 0; i < values.length; i += 1) {
            const value = values[i];
            values[i] = value / (1 + Math.exp(-value));
        }
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
    },
    /** Repeat every frame twice, nearest neighbour. */
    upsample2 = (options: {
        input: Float32Array;
        frames: number;
        channels: number;
    }): Float32Array => {
        const
            { input, frames, channels } = options,
            out = new Float32Array(frames * 2 * channels);
        for (let frame = 0; frame < frames; frame += 1) {
            const source = frame * channels;
            out.set(input.subarray(source, source + channels),
                frame * 2 * channels);
            out.set(input.subarray(source, source + channels),
                (frame * 2 + 1) * channels);
        }
        return out;
    },
    /** Reflection pad one frame on the left. */
    reflectPadLeft = (options: {
        input: Float32Array;
        frames: number;
        channels: number;
    }): Float32Array => {
        const
            { input, frames, channels } = options,
            out = new Float32Array((frames + 1) * channels);
        if (frames > 1) {
            out.set(input.subarray(channels, channels * 2), 0);
        }
        out.set(input, channels);
        return out;
    },
    /** Concatenate channel blocks frame by frame. */
    concatChannels = (options: {
        parts: Array<{ data: Float32Array; channels: number }>;
        frames: number;
    }): Float32Array => {
        const
            { parts, frames } = options,
            channels = parts.reduce((total, part) => total + part.channels, 0),
            out = new Float32Array(frames * channels);
        for (let frame = 0; frame < frames; frame += 1) {
            let offset = frame * channels;
            for (const part of parts) {
                const source = frame * part.channels;
                out.set(
                    part.data.subarray(source, source + part.channels),
                    offset,
                );
                offset += part.channels;
            }
        }
        return out;
    },
    /** Repeat feature rows by phoneme durations. */
    alignRepeat = (options: {
        input: Float32Array;
        rows: number;
        channels: number;
        durations: number[];
    }): Float32Array => {
        const
            { input, rows, channels, durations } = options,
            frames = durations.reduce((total, value) => total + value, 0),
            out = new Float32Array(frames * channels);
        let target = 0;
        for (let row = 0; row < rows; row += 1) {
            const source = row * channels;
            for (let step = 0; step < durations[row]; step += 1) {
                out.set(
                    input.subarray(source, source + channels),
                    target * channels,
                );
                target += 1;
            }
        }
        return out;
    },
    /** Round half to even, matching torch.round. */
    roundHalfEven = (value: number): number => {
        const floor = Math.floor(value);
        const diff = value - floor;
        if (diff > 0.5) {
            return floor + 1;
        }
        if (diff < 0.5) {
            return floor;
        }
        return floor % 2 === 0 ? floor : floor + 1;
    };

export {
    adainInPlace,
    adalnInPlace,
    alignRepeat,
    concatChannels,
    conv1d,
    convTranspose1d,
    depthwiseConvTranspose1d,
    geluErfInPlace,
    geluInPlace,
    instanceNorm,
    layerNorm,
    leakyInPlace,
    linear,
    reflectPadLeft,
    roundHalfEven,
    sigmoid,
    siluInPlace,
    snakeInPlace,
    upsample2,
};
