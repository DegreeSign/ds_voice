// Audio file and blob decoding.
import { decodeFlac } from "./flac";
import { decodeMp3 } from "./mp3";
import { decodeWav } from "./wav";
import { resampleAudio } from "./resample";
import { Platform } from "../tensor/types";
import type {
    DecodeBytesOptions,
    DecodeSourceOptions,
    DecodedAudio,
} from "../tensor/types";
const
    /** Target sample rate for every decode path. */
    TARGET_RATE = 16000,
    /** Move mono audio onto the target sample rate. */
    toTargetRate = (audio: DecodedAudio): DecodedAudio => {
        if (audio.samplingRate === TARGET_RATE) {
            return audio;
        }
        return {
            pcm: resampleAudio({
                input: audio.pcm,
                inputRate: audio.samplingRate,
                outputRate: TARGET_RATE,
            }),
            samplingRate: TARGET_RATE,
        };
    },
    /** Decode with the Web Audio API. */
    decodeWebAudio = async (bytes: Uint8Array): Promise<DecodedAudio> => {
        if (typeof OfflineAudioContext === "undefined") {
            throw new Error("Web Audio decoding unavailable");
        }
        const
            context = new OfflineAudioContext(1, 1, TARGET_RATE),
            buffer = await context.decodeAudioData(bytes.slice().buffer),
            frames = buffer.length,
            mixed = new Float32Array(frames);
        for (let c = 0; c < buffer.numberOfChannels; c += 1) {
            const data = buffer.getChannelData(c);
            for (let i = 0; i < frames; i += 1) {
                mixed[i] += data[i];
            }
        }
        if (buffer.numberOfChannels > 1) {
            for (let i = 0; i < frames; i += 1) {
                mixed[i] /= buffer.numberOfChannels;
            }
        }
        return {
            pcm: mixed,
            samplingRate: buffer.sampleRate,
        };
    },
    /** Decode a Node buffer by magic bytes. */
    decodeNodeBytes = async (bytes: Uint8Array): Promise<DecodedAudio> => {
        const
            head = String.fromCharCode(
                bytes[0] ?? 0,
                bytes[1] ?? 0,
                bytes[2] ?? 0,
                bytes[3] ?? 0,
            );
        if (head === "RIFF") {
            return decodeWav({ bytes });
        }
        if (head.startsWith("fLaC")) {
            return decodeFlac({ bytes });
        }
        const id3 = head.startsWith("ID3");
        const sync = (bytes[0] ?? 0) === 0xff && ((bytes[1] ?? 0) & 0xe0) === 0xe0;
        if (id3 || sync) {
            return await decodeMp3({ bytes });
        }
        throw new Error("unsupported audio format");
    },
    /** Decode raw bytes for the current platform. */
    decodeBytes = async (options: DecodeBytesOptions): Promise<DecodedAudio> => {
        const audio = options.platform === Platform.Node
            ? await decodeNodeBytes(options.bytes)
            : await decodeWebAudio(options.bytes);
        return toTargetRate(audio);
    },
    /** Decode a URL, blob or filesystem path. */
    decodeSource = async (
        options: DecodeSourceOptions,
    ): Promise<DecodedAudio> => {
        if (typeof options.audioFilePath !== "string") {
            const bytes = new Uint8Array(
                await options.audioFilePath.arrayBuffer(),
            );
            return await decodeBytes({ bytes, platform: options.platform });
        }
        if (options.platform === Platform.Node) {
            const { readAudioFile } = await import("./nodeFile");
            return await decodeBytes({
                bytes: await readAudioFile(options.audioFilePath),
                platform: options.platform,
            });
        }
        const response = await fetch(options.audioFilePath);
        if (!response.ok) {
            throw new Error(`fetch failed ${response.status}`);
        }
        return await decodeBytes({
            bytes: new Uint8Array(await response.arrayBuffer()),
            platform: options.platform,
        });
    };

export { decodeBytes, decodeSource };
