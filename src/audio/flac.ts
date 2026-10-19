// Pure TypeScript FLAC decoder.
import { downmix } from "./resample";
import type {
    DecodedAudio,
    FlacStreamInfo,
    ReadFlacOptions,
} from "../tensor/types";
const
    /** Sample size table for the frame header codes. */
    SAMPLE_SIZES = [0, 8, 12, 0, 16, 20, 24, 32],
    /** Sync code of a FLAC frame. */
    SYNC = 0x3ffe,
    /** Metadata block holding the stream information. */
    STREAMINFO = 0,
    /** Bit stream over a byte buffer. */
    createBitReader = (bytes: Uint8Array, start = 0) => {
        let position = start * 8;
        const
            length = bytes.length * 8,
            /** Align to the next byte. */
            align = (): void => {
                position = (position + 7) & ~7;
            },
            /** Read an unsigned field of at most 36 bits. */
            readBits = (count: number): number => {
                let
                    value = 0,
                    remaining = count;
                while (remaining > 0) {
                    const
                        byteIndex = position >> 3,
                        bitIndex = position & 7,
                        available = 8 - bitIndex,
                        take = Math.min(remaining, available),
                        shift = available - take,
                        bits = ((bytes[byteIndex] ?? 0) >> shift) &
                            ((1 << take) - 1);
                    value = value * 2 ** take + bits;
                    position += take;
                    remaining -= take;
                }
                return value;
            },
            /** Read a signed two's complement field. */
            readSigned = (count: number): number => {
                const
                    value = readBits(count),
                    sign = 2 ** (count - 1);
                return value >= sign ? value - 2 ** count : value;
            },
            /** Read a unary coded count of zero bits. */
            readUnary = (): number => {
                let count = 0;
                while (readBits(1) === 0) {
                    count += 1;
                }
                return count;
            },
            /** Read a UTF-8 coded frame or sample number. */
            readUtf8 = (): number => {
                const first = readBits(8);
                if ((first & 0x80) === 0) {
                    return first;
                }
                let
                    extra = 0,
                    mask = 0x40;
                while ((first & mask) !== 0) {
                    extra += 1;
                    mask >>= 1;
                }
                let value = first & ((1 << (7 - extra)) - 1);
                for (let i = 0; i < extra; i += 1) {
                    value = value * 64 + (readBits(8) & 0x3f);
                }
                return value;
            };
        return {
            get position(): number {
                return position;
            },
            set position(value: number) {
                position = value;
            },
            get length(): number {
                return length;
            },
            align,
            readBits,
            readSigned,
            readUnary,
            readUtf8,
        };
    },
    /** Skip an ID3v2 tag when present. */
    skipId3 = (bytes: Uint8Array): number => {
        if (
            bytes.length < 10 ||
            bytes[0] !== 0x49 ||
            bytes[1] !== 0x44 ||
            bytes[2] !== 0x33
        ) {
            return 0;
        }
        const
            size = ((bytes[6] & 0x7f) << 21) |
                ((bytes[7] & 0x7f) << 14) |
                ((bytes[8] & 0x7f) << 7) |
                (bytes[9] & 0x7f),
            footer = (bytes[5] & 0x10) !== 0 ? 10 : 0;
        return 10 + size + footer;
    },
    /** Decode a Rice coded residual of one subframe. */
    decodeResidual = (options: {
        reader: ReturnType<typeof createBitReader>;
        blockSize: number;
        order: number;
    }): number[] => {
        const
            { reader, blockSize, order } = options,
            method = reader.readBits(2);
        if (method > 1) {
            throw new Error("bad residual method");
        }
        const
            partitionOrder = reader.readBits(4),
            partitions = 1 << partitionOrder,
            partitionSamples = blockSize >> partitionOrder;
        if (partitionSamples * partitions !== blockSize) {
            throw new Error("bad residual partition");
        }
        const
            paramBits = method === 0 ? 4 : 5,
            escape = (1 << paramBits) - 1,
            residual: number[] = [];
        for (let p = 0; p < partitions; p += 1) {
            const count = p === 0
                ? partitionSamples - order
                : partitionSamples;
            const param = reader.readBits(paramBits);
            if (param === escape) {
                const rawBits = reader.readBits(5);
                for (let i = 0; i < count; i += 1) {
                    residual.push(
                        rawBits === 0 ? 0 : reader.readSigned(rawBits),
                    );
                }
                continue;
            }
            for (let i = 0; i < count; i += 1) {
                const
                    quotient = reader.readUnary(),
                    remainder = reader.readBits(param),
                    value = quotient * 2 ** param + remainder;
                residual.push(
                    value % 2 === 0 ? value / 2 : -(value + 1) / 2,
                );
            }
        }
        return residual;
    },
    /** Decode one fixed predictor subframe. */
    decodeFixed = (options: {
        reader: ReturnType<typeof createBitReader>;
        blockSize: number;
        order: number;
        bits: number;
    }): Float64Array => {
        const
            { reader, blockSize, order, bits } = options,
            samples = new Float64Array(blockSize);
        for (let i = 0; i < order; i += 1) {
            samples[i] = reader.readSigned(bits);
        }
        const residual = decodeResidual({ reader, blockSize, order });
        for (let i = order; i < blockSize; i += 1) {
            let predicted = 0;
            if (order === 1) {
                predicted = samples[i - 1];
            } else if (order === 2) {
                predicted = 2 * samples[i - 1] - samples[i - 2];
            } else if (order === 3) {
                predicted = 3 * samples[i - 1] - 3 * samples[i - 2] +
                    samples[i - 3];
            } else if (order === 4) {
                predicted = 4 * samples[i - 1] - 6 * samples[i - 2] +
                    4 * samples[i - 3] - samples[i - 4];
            }
            samples[i] = predicted + residual[i - order];
        }
        return samples;
    },
    /** Decode one linear predictor subframe. */
    decodeLpc = (options: {
        reader: ReturnType<typeof createBitReader>;
        blockSize: number;
        order: number;
        bits: number;
    }): Float64Array => {
        const
            { reader, blockSize, order, bits } = options,
            samples = new Float64Array(blockSize);
        for (let i = 0; i < order; i += 1) {
            samples[i] = reader.readSigned(bits);
        }
        const precision = reader.readBits(4) + 1;
        if (precision === 16) {
            throw new Error("bad lpc precision");
        }
        const
            shift = reader.readSigned(5),
            coefficients: number[] = [];
        for (let i = 0; i < order; i += 1) {
            coefficients.push(reader.readSigned(precision));
        }
        const residual = decodeResidual({ reader, blockSize, order });
        for (let i = order; i < blockSize; i += 1) {
            let sum = 0;
            for (let j = 0; j < order; j += 1) {
                sum += coefficients[j] * samples[i - 1 - j];
            }
            const predicted = shift >= 0
                ? Math.floor(sum / 2 ** shift)
                : sum * 2 ** -shift;
            samples[i] = predicted + residual[i - order];
        }
        return samples;
    },
    /** Decode one subframe. */
    decodeSubframe = (options: {
        reader: ReturnType<typeof createBitReader>;
        blockSize: number;
        bits: number;
    }): Float64Array => {
        const
            { reader, blockSize } = options,
            zero = reader.readBits(1);
        if (zero !== 0) {
            throw new Error("bad subframe header");
        }
        const
            type = reader.readBits(6),
            wastedFlag = reader.readBits(1);
        let wasted = 0;
        if (wastedFlag === 1) {
            wasted = reader.readUnary() + 1;
        }
        const bits = options.bits - wasted;
        if (bits <= 0) {
            throw new Error("bad wasted bits");
        }
        let samples: Float64Array;
        if (type === 0) {
            samples = new Float64Array(blockSize);
            samples.fill(reader.readSigned(bits));
        } else if (type === 1) {
            samples = new Float64Array(blockSize);
            for (let i = 0; i < blockSize; i += 1) {
                samples[i] = reader.readSigned(bits);
            }
        } else if ((type & 0x38) === 0x08) {
            const order = type & 0x07;
            if (order > 4) {
                throw new Error("bad fixed order");
            }
            samples = decodeFixed({ reader, blockSize, order, bits });
        } else if ((type & 0x20) !== 0) {
            const order = (type & 0x1f) + 1;
            samples = decodeLpc({ reader, blockSize, order, bits });
        } else {
            throw new Error("bad subframe type");
        }
        if (wasted > 0) {
            const scale = 2 ** wasted;
            for (let i = 0; i < blockSize; i += 1) {
                samples[i] *= scale;
            }
        }
        return samples;
    },
    /** Decode one frame body into per channel sample arrays. */
    decodeFrame = (options: {
        reader: ReturnType<typeof createBitReader>;
        stream: FlacStreamInfo;
    }): { channels: Float64Array[]; bits: number } => {
        const
            { reader, stream } = options,
            blocking = reader.readBits(1);
        void blocking;
        const
            blockSizeCode = reader.readBits(4),
            sampleRateCode = reader.readBits(4),
            channelAssignment = reader.readBits(4),
            sampleSizeCode = reader.readBits(3),
            reserved = reader.readBits(1);
        if (reserved !== 0) {
            throw new Error("bad frame reserved bit");
        }
        reader.readUtf8();
        let blockSize = 0;
        if (blockSizeCode === 1) {
            blockSize = 192;
        } else if (blockSizeCode >= 2 && blockSizeCode <= 5) {
            blockSize = 576 * 2 ** (blockSizeCode - 2);
        } else if (blockSizeCode === 6) {
            blockSize = reader.readBits(8) + 1;
        } else if (blockSizeCode === 7) {
            blockSize = reader.readBits(16) + 1;
        } else if (blockSizeCode >= 8) {
            blockSize = 256 * 2 ** (blockSizeCode - 8);
        } else {
            throw new Error("bad block size code");
        }
        if (sampleRateCode === 12) {
            reader.readBits(8);
        } else if (sampleRateCode === 13 || sampleRateCode === 14) {
            reader.readBits(16);
        }
        reader.readBits(8);
        const bits = sampleSizeCode === 0
            ? stream.bitsPerSample
            : SAMPLE_SIZES[sampleSizeCode];
        if (bits === 0) {
            throw new Error("bad sample size");
        }
        const independent = channelAssignment <= 7
            ? channelAssignment + 1
            : 2;
        const channels: Float64Array[] = [];
        for (let c = 0; c < independent; c += 1) {
            channels.push(decodeSubframe({ reader, blockSize, bits }));
        }
        reader.readBits(16);
        if (channelAssignment === 8) {
            const
                left = channels[0],
                side = channels[1],
                right = new Float64Array(blockSize);
            for (let i = 0; i < blockSize; i += 1) {
                right[i] = left[i] - side[i];
            }
            return { channels: [left, right], bits };
        }
        if (channelAssignment === 9) {
            const
                side = channels[0],
                right = channels[1],
                left = new Float64Array(blockSize);
            for (let i = 0; i < blockSize; i += 1) {
                left[i] = right[i] + side[i];
            }
            return { channels: [left, right], bits };
        }
        if (channelAssignment === 10) {
            const
                midSide = channels[0],
                side = channels[1],
                left = new Float64Array(blockSize),
                right = new Float64Array(blockSize);
            for (let i = 0; i < blockSize; i += 1) {
                const mid = midSide[i] * 2 + (side[i] & 1);
                left[i] = Math.floor((mid + side[i]) / 2);
                right[i] = Math.floor((mid - side[i]) / 2);
            }
            return { channels: [left, right], bits };
        }
        return { channels, bits };
    },
    /** Parse the stream information block. */
    parseStreamInfo = (bytes: Uint8Array): FlacStreamInfo => {
        const reader = createBitReader(bytes);
        reader.readBits(16);
        reader.readBits(16);
        reader.readBits(24);
        reader.readBits(24);
        const
            sampleRate = reader.readBits(20),
            channels = reader.readBits(3) + 1,
            bitsPerSample = reader.readBits(5) + 1,
            totalSamples = reader.readBits(36);
        return { sampleRate, channels, bitsPerSample, totalSamples };
    },
    /** Decode a FLAC buffer into mono f32 audio. */
    decodeFlac = (options: ReadFlacOptions): DecodedAudio => {
        const
            { bytes } = options,
            base = skipId3(bytes);
        if (
            fourCc({ bytes, offset: base }) !== "fLaC"
        ) {
            throw new Error("not a FLAC file");
        }
        let
            position = base + 4,
            stream: FlacStreamInfo | null = null;
        for (;;) {
            if (position + 4 > bytes.length) {
                break;
            }
            const
                header = bytes[position],
                last = (header & 0x80) !== 0,
                type = header & 0x7f,
                length = (bytes[position + 1] << 16) |
                    (bytes[position + 2] << 8) |
                    bytes[position + 3];
            position += 4;
            if (type === STREAMINFO && length >= 34) {
                stream = parseStreamInfo(
                    bytes.subarray(position, position + length),
                );
            }
            position += length;
            if (last) {
                break;
            }
        }
        if (!stream) {
            throw new Error("missing FLAC stream info");
        }
        const
            length = bytes.length * 8,
            blocks: Float64Array[][] = Array.from(
                { length: stream.channels },
                () => [],
            );
        let
            reader = createBitReader(bytes, position),
            bits = stream.bitsPerSample;
        while (reader.position + 16 <= length) {
            reader.align();
            const start = reader.position;
            if (reader.readBits(14) !== SYNC) {
                reader.position = start + 8;
                continue;
            }
            if (reader.readBits(1) !== 0) {
                reader.position = start + 8;
                continue;
            }
            try {
                const frame = decodeFrame({ reader, stream });
                bits = frame.bits;
                for (let c = 0; c < blocks.length; c += 1) {
                    blocks[c].push(frame.channels[c] ?? frame.channels[0]);
                }
            } catch {
                reader.position = start + 8;
            }
        }
        const
            totals = blocks.map((list) => {
                let total = 0;
                for (const block of list) {
                    total += block.length;
                }
                return total;
            }),
            count = Math.min(...totals),
            channelSamples = blocks.map((list) => {
                const out = new Float32Array(count);
                let offset = 0;
                for (const block of list) {
                    if (offset >= count) {
                        break;
                    }
                    const take = Math.min(block.length, count - offset);
                    out.set(block.subarray(0, take), offset);
                    offset += take;
                }
                return out;
            }),
            scale = 2 ** (bits - 1),
            samples = new Float32Array(count * blocks.length);
        for (let i = 0; i < count; i += 1) {
            for (let c = 0; c < blocks.length; c += 1) {
                samples[i * blocks.length + c] =
                    channelSamples[c][i] / scale;
            }
        }
        return {
            pcm: downmix({ input: samples, channels: stream.channels }),
            samplingRate: stream.sampleRate,
        };
    },
    /** Read a four character code. */
    fourCc = (options: { bytes: Uint8Array; offset: number }): string => {
        const { bytes, offset } = options;
        return String.fromCharCode(
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        );
    };

export { decodeFlac };
