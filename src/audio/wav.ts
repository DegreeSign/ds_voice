// Minimal RIFF WAVE reader.
import { downmix } from "./resample";
import type { DecodedAudio, ReadWavOptions } from "../tensor/types";
const
    /** Read a four character code. */
    fourCc = (options: { bytes: Uint8Array; offset: number }): string => {
        const { bytes, offset } = options;
        return String.fromCharCode(
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        );
    },
    /** Read one little-endian unsigned integer. */
    u32 = (options: { bytes: Uint8Array; offset: number }): number => {
        const { bytes, offset } = options;
        return (
            bytes[offset] |
            (bytes[offset + 1] << 8) |
            (bytes[offset + 2] << 16) |
            (bytes[offset + 3] << 24)
        ) >>> 0;
    },
    /** Read one little-endian unsigned short. */
    u16 = (options: { bytes: Uint8Array; offset: number }): number => {
        const { bytes, offset } = options;
        return bytes[offset] | (bytes[offset + 1] << 8);
    },
    /** Read one sample in the declared format. */
    readSample = (options: {
        view: DataView;
        at: number;
        format: number;
        bitsPerSample: number;
    }): number => {
        const { view, at, format, bitsPerSample } = options;
        if (format === 3) {
            return view.getFloat32(at, true);
        }
        if (bitsPerSample === 16) {
            return view.getInt16(at, true) / 32768;
        }
        if (bitsPerSample === 24) {
            const
                low = view.getUint16(at, true),
                high = view.getUint8(at + 2);
            let value = low | (high << 16);
            if (value & 0x800000) {
                value -= 0x1000000;
            }
            return value / 8388608;
        }
        if (bitsPerSample === 32) {
            return view.getInt32(at, true) / 2147483648;
        }
        if (bitsPerSample === 8) {
            return (view.getUint8(at) - 128) / 128;
        }
        throw new Error(`unsupported bits ${bitsPerSample}`);
    },
    /** Parse a WAV buffer into mono f32 audio. */
    decodeWav = (options: ReadWavOptions): DecodedAudio => {
        const { bytes } = options;
        if (bytes.byteLength < 44 || fourCc({ bytes, offset: 0 }) !== "RIFF") {
            throw new Error("not a RIFF file");
        }
        if (fourCc({ bytes, offset: 8 }) !== "WAVE") {
            throw new Error("not a WAVE file");
        }
        let
            offset = 12,
            format = 0,
            channels = 0,
            sampleRate = 0,
            bitsPerSample = 0,
            dataOffset = 0,
            dataLength = 0;
        while (offset + 8 <= bytes.byteLength) {
            const
                id = fourCc({ bytes, offset }),
                size = u32({ bytes, offset: offset + 4 }),
                body = offset + 8;
            if (id === "fmt ") {
                format = u16({ bytes, offset: body });
                channels = u16({ bytes, offset: body + 2 });
                sampleRate = u32({ bytes, offset: body + 4 });
                bitsPerSample = u16({ bytes, offset: body + 14 });
            } else if (id === "data") {
                dataOffset = body;
                dataLength = Math.min(size, bytes.byteLength - body);
            }
            offset = body + size + (size % 2);
        }
        if (channels === 0 || sampleRate === 0 || dataLength === 0) {
            throw new Error("incomplete WAVE file");
        }
        const
            bytesPerSample = bitsPerSample / 8,
            frames = Math.floor(dataLength / (bytesPerSample * channels)),
            interleaved = new Float32Array(frames * channels),
            view = new DataView(
                bytes.buffer,
                bytes.byteOffset,
                bytes.byteLength,
            );
        for (let i = 0; i < frames * channels; i += 1) {
            interleaved[i] = readSample({
                view,
                at: dataOffset + i * bytesPerSample,
                format,
                bitsPerSample,
            });
        }
        return {
            pcm: downmix({ input: interleaved, channels }),
            samplingRate: sampleRate,
        };
    };

export { decodeWav };
