// MP3 decoding through the inline minimp3 wasm module.
import { MINIMP3_WASM } from "./minimp3Wasm";
import { downmix } from "./resample";
import type {
    DecodedAudio,
    Minimp3Exports,
    ReadMp3Options,
} from "../tensor/types";
const
    /** Seconds decoded per wasm call. */
    DECODE_STEP_SECONDS = 10,
    /** Guard against a decoder that never reports the end. */
    MAX_STEPS = 100_000,
    /** Lazily compiled wasm module. */
    loadModule = (() => {
        let module: Promise<WebAssembly.Module> | null = null;
        return (): Promise<WebAssembly.Module> => {
            if (!module) {
                const
                    binary = atob(MINIMP3_WASM),
                    bytes = new Uint8Array(binary.length);
                for (let i = 0; i < binary.length; i += 1) {
                    bytes[i] = binary.charCodeAt(i);
                }
                module = WebAssembly.compile(bytes);
            }
            return module;
        };
    })(),
    /** Decode a MP3 buffer into mono f32 audio. */
    decodeMp3 = async (options: ReadMp3Options): Promise<DecodedAudio> => {
        const
            { bytes } = options,
            compiled = await loadModule(),
            instance = await WebAssembly.instantiate(compiled, {}),
            wasm = instance.exports as unknown as Minimp3Exports,
            chunks: Int16Array[] = [];
        let
            sampleRate = 0,
            channels = 0;
        wasm.decoder_init();
        wasm.decoder_set_mp3_data_size(bytes.byteLength);
        const target = new Uint8Array(
            wasm.memory.buffer,
            wasm.decoder_mp3_data_offset(),
            wasm.decoder_mp3_data_size(),
        );
        target.set(bytes);
        wasm.decoder_seek(0);
        for (let step = 0; step < MAX_STEPS; step += 1) {
            wasm.decoder_decode(DECODE_STEP_SECONDS);
            const
                frames = wasm.decode_results_num_samples(),
                channelCount = wasm.decode_results_num_channels();
            if (frames <= 0 || channelCount <= 0) {
                break;
            }
            if (sampleRate === 0) {
                sampleRate = wasm.decode_results_sampling_rate();
                channels = channelCount;
            }
            const
                size = wasm.decoder_pcm_data_size(),
                count = Math.min(size / 2, frames * channelCount),
                view = new Int16Array(
                    wasm.memory.buffer,
                    wasm.decoder_pcm_data_offset(),
                    count,
                );
            chunks.push(new Int16Array(view));
        }
        if (sampleRate === 0 || channels === 0) {
            throw new Error("mp3 produced no samples");
        }
        let total = 0;
        for (const chunk of chunks) {
            total += chunk.length;
        }
        const interleaved = new Float32Array(total);
        let offset = 0;
        for (const chunk of chunks) {
            for (let i = 0; i < chunk.length; i += 1) {
                interleaved[offset + i] = chunk[i] / 32768;
            }
            offset += chunk.length;
        }
        return {
            pcm: downmix({ input: interleaved, channels }),
            samplingRate: sampleRate,
        };
    };

export { decodeMp3 };
