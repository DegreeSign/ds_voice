// Microphone reference capture and MOSS audio codec encoding.
import { captureMicrophone } from "../audio/input/mic";
import { resampleAudio } from "../audio/input/resample";
import { loadCloneFiles, readCloneWeights } from "./download";
import {
    encodeMossAudio,
    loadMossEncoder,
    MAX_REFERENCE_SAMPLES,
    MIN_REFERENCE_SAMPLES,
} from "./engines/mossEncoder";
import type { MossEncoder } from "./engines/mossEncoder";
import type { CloneVoiceOptions, VoiceClone } from "./types";

let
    /** Parsed clone codec kept for later calls. */
    residentEncoder: MossEncoder | undefined;

const
    /** Native rate of the MOSS audio tokenizer. */
    SAMPLE_RATE = 48000,
    /** Working rate of the browser microphone. */
    MIC_RATE = 16000,
    /** Default microphone length in milliseconds. */
    DEFAULT_DURATION_MS = 3000,
    /** Smallest microphone length in milliseconds. */
    MIN_DURATION_MS = 250,
    /** Codebooks in one frame. */
    CODES_PER_FRAME = 16,
    /** Wait one microphone interval. */
    wait = async (durationMs: number): Promise<void> => {
        await new Promise((resolve) => {
            setTimeout(resolve, durationMs);
        });
    },
    /** Record one mono reference from the microphone. */
    recordMicrophone = async (durationMs: number): Promise<Float32Array> => {
        if (
            typeof navigator === "undefined"
            || !navigator.mediaDevices?.getUserMedia
        ) {
            throw new Error(`Microphone unavailable`);
        }
        const
            chunks: Float32Array[] = [],
            length = Math.max(MIN_DURATION_MS, durationMs);
        let failure: Error | undefined;
        const handle = captureMicrophone({
            onChunk: (chunk) => {
                chunks.push(chunk);
            },
            onError: (reason) => {
                failure = reason;
            },
        });
        try {
            await handle.ready;
            await wait(length);
        } finally {
            await handle.stop();
        }
        if (failure) {
            throw failure;
        }
        const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
        if (total === 0) {
            throw new Error(`Microphone recorded no audio`);
        }
        const pcm = new Float32Array(total);
        let offset = 0;
        for (const chunk of chunks) {
            pcm.set(chunk, offset);
            offset += chunk.length;
        }
        return pcm;
    },
    /** Load or reuse the clone codec. */
    loadEncoder = async (options: CloneVoiceOptions): Promise<MossEncoder> => {
        if (residentEncoder) {
            return residentEncoder;
        }
        await loadCloneFiles({
            weightsBaseUrl: options.weightsBaseUrl,
            onProgress: options.onProgress,
        });
        const bytes = await readCloneWeights();
        residentEncoder = loadMossEncoder(bytes);
        return residentEncoder;
    };

/** Record a reference and return its MOSS codes. */
const cloneVoice = async (
    options: CloneVoiceOptions = {},
): Promise<VoiceClone> => {
    const
        captured = options.pcm ?? await recordMicrophone(
            options.durationMs ?? DEFAULT_DURATION_MS,
        ),
        rate = options.pcm ? options.sampleRate ?? SAMPLE_RATE : MIC_RATE,
        reference = rate === SAMPLE_RATE
            ? captured
            : resampleAudio({
                input: captured,
                inputRate: rate,
                outputRate: SAMPLE_RATE,
            }),
        trimmed = reference.length > MAX_REFERENCE_SAMPLES
            ? reference.subarray(0, MAX_REFERENCE_SAMPLES)
            : reference;
    if (trimmed.length < MIN_REFERENCE_SAMPLES) {
        throw new Error(`reference audio is too short`);
    }
    const
        encoder = await loadEncoder(options),
        codes = encodeMossAudio({ encoder, pcm: trimmed });
    return {
        codes,
        frames: codes.length / CODES_PER_FRAME,
        sampleRate: SAMPLE_RATE,
    };
};

export { cloneVoice };
