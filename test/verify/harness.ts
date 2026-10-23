// Browser test harness for @degreesign/voice.
import {
    dtypeSize,
    f16ToF32,
    parseSafetensorsJson,
    product,
    readF16,
} from "../../src/tensor/binary";
import {
    addArrays,
    addInPlace,
    argmax,
    attention,
    crossAttentionOne,
    geluInPlace,
    layerNorm,
    linear,
    reluInPlace,
    siluInPlace,
    softmaxRow,
} from "../../src/tensor/kernels";
import { nextPow2, reflect } from "../../src/tensor/math";
import {
    buildFftPlan,
    computeMel,
    melEnergies,
    powerSpectrum,
} from "../../src/speech/mel";
import { dedupeBoundary, joinText, planChunks } from "../../src/speech/segment";
import {
    CHECKPOINT_IDS,
    DEFAULT_MODEL,
    DEFAULT_QUALITY,
    HF_BASE,
    MODEL_IDS,
    MODEL_SPECS,
    checkpointFiles,
    checkpointKey,
    resolveSpec,
    weightFiles,
} from "../../src/models/specs";
import { ModelFamily, ModelId } from "../../src/tensor/types";
import { parseGguf } from "../../src/formats/gguf";
import {
    assertWhisperShape,
    parseSafetensors,
} from "../../src/formats/safetensors";
import {
    readSenseVoiceConfig,
    readSenseVoiceFrontend,
} from "../../src/models/sensevoice/config";
import { decodeCtc } from "../../src/models/sensevoice/ctc";
import {
    encodeSenseVoice,
    senseVoiceLogits,
} from "../../src/models/sensevoice/encoder";
import { computeSenseVoiceFeatures } from "../../src/models/sensevoice/fbank";
import { createSenseVoiceVocabulary } from "../../src/models/sensevoice/tokenizer";
import { createGgufWeights } from "../../src/models/sensevoice/weights";
import {
    loadSenseVoice,
    transcribeSenseVoice,
} from "../../src/models/sensevoice/model";
import {
    TOKENS,
    assertSuppressMatch,
    loadConfigs,
    toConstants,
    toMelConfig,
    toMelFilters,
    toModelConfig,
} from "../../src/models/whisper/config";
import {
    createDecoderState,
    decoderStep,
    detectLanguage,
} from "../../src/models/whisper/decoder";
import { computeDtw } from "../../src/models/whisper/dtw";
import { encode } from "../../src/models/whisper/encoder";
import { generate } from "../../src/models/whisper/generate";
import { loadTokenizer } from "../../src/models/whisper/tokenizer";
import { loadWhisper, transcribeWhisper } from "../../src/models/whisper/model";
import { buildSegments, buildWords } from "../../src/models/whisper/words";
import {
    CHUNK_BYTES,
    chunkedExists,
    chunkedReady,
    chunkedSize,
    completeSizes,
    readChunks,
    removeChunks,
    writeChunkStream,
} from "../../src/store/chunk";
import { createBrowserStore } from "../../src/store/browser";
import {
    deleteModel,
    isCheckpointCached,
    prepareModel,
} from "../../src/store/download";
import {
    DEFAULT_DISK_SPACE_MB,
    ensureSpace,
    forgetModel,
    readUsage,
    touchModel,
} from "../../src/store/usage";
import { getStore } from "../../src/store/index";
import { detectPlatform, hasMicrophone } from "../../src/speech/env";
import { decodeBytes, decodeSource } from "../../src/audio/file";
import { decodeFlac } from "../../src/audio/flac";
import { decodeMp3 } from "../../src/audio/mp3";
import { decodeWav } from "../../src/audio/wav";
import { downmix, resampleAudio } from "../../src/audio/resample";
import {
    Language,
    Platform,
    ProgressPhase,
    Quality,
    ThreadCount,
    TimestampMode,
} from "../../src/tensor/types";
import { createVoice, getVoice } from "../../src/speech/voice";
import {
    createVoice as createSenseVoiceVoice,
} from "../../src/speech/senseVoice";
import {
    createVoice as createWhisperVoice,
} from "../../src/speech/whisper";
import { createSession } from "../../src/speech/session";
import { captureMicrophone } from "../../src/audio/mic";
import {
    getKernel,
    kernelThreads,
    releaseTensor,
    setThreadCount,
} from "../../src/tensor/wasm";
import {
    createPool as createBrowserPool,
    createPoolAsync as createBrowserPoolAsync,
    createSharedMemory as createBrowserSharedMemory,
} from "../../src/tensor/poolBrowser";
import {
    createPool as createStubPool,
    createSharedMemory as createStubSharedMemory,
} from "../../src/tensor/poolStub";
import {
    createWhisperWeights,
    mapName,
} from "../../src/models/whisper/weights";
import {
    GGUF_CACHE_BYTES,
    GGUF_CACHE_CAP_BYTES,
    fullCacheBytes,
    ggmlTypeName,
    readGgufTensor,
} from "../../src/formats/ggufTensor";
import { TINY_FLAC, TINY_MP3 } from "./fixtures";
import { writeWav16 } from "./wav";
import * as entry from "../../src/index";
import {
    createDictation,
    DictationErrorSource,
    DictationState,
    isDictationSupported,
} from "../../src/dictation";
import { serveDictation } from "../../src/dictation/worker";
import {
    DictationMessageType,
    DictationRequestType,
} from "../../src/dictation/types";
import type {
    DictationCapture,
    DictationMessage,
    DictationRequest,
} from "../../src/dictation/types";
import type {
    Constants,
    DownloadProgress,
    GgufFile,
    GgufValue,
    LoadedModel,
    LoadedSenseVoice,
    MelConfig,
    MicrophoneCapture,
    MicrophoneCaptureHandle,
    Model,
    ModelConfig,
    ModelPartStore,
    ModelStore,
    RawGenerationConfig,
    RawModelConfig,
    RawPreprocessorConfig,
    SenseVoiceConfig,
    SenseVoiceWeights,
    TensorStore,
    Tokenizer,
    Transcript,
    Voice,
} from "../../src/tensor/types";

/** One recorded test outcome. */
interface TestOutcome {
    name: string;
    ok: boolean;
    error?: string;
}

/** Result posted back by the pool worker test bundle. */
interface PoolWorkerResult {
    ok: boolean;
    threads?: number;
    value?: number;
    shared?: boolean;
    error?: string;
}

/** Growable little endian byte writer. */
interface ByteWriter {
    u8(value: number): void;
    u16(value: number): void;
    u32(value: number): void;
    i32(value: number): void;
    u64(value: number): void;
    f32(value: number): void;
    f64(value: number): void;
    text(value: string): void;
    raw(value: Uint8Array): void;
    bytes(): Uint8Array;
    size(): number;
}

/** One synthetic tensor with an optional explicit value list. */
interface TensorSpec {
    name: string;
    shape: number[];
    values?: number[];
}

/** A tiny in memory model store for the tests. */
interface FakeStore extends ModelStore {
    texts: Map<string, string>;
    bytes: Map<string, Uint8Array>;
}

const
    /** Collected outcomes, always with a numeric fail count. */
    outcomes: TestOutcome[] = [],
    /** Run one test, recording its error and continuing. */
    test = async (
        name: string,
        run: () => void | Promise<void>,
    ): Promise<void> => {
        try {
            await run();
            outcomes.push({ name, ok: true });
        } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            outcomes.push({ name, ok: false, error: message });
        }
    },
    /** Assert a truthy condition. */
    assert = (condition: unknown, message: string): void => {
        if (!condition) {
            throw new Error(message);
        }
    },
    /** Assert strict equality. */
    assertEqual = (actual: unknown, expected: unknown, label: string): void => {
        if (actual !== expected) {
            throw new Error(`${label}: expected ${String(expected)} got ${String(actual)}`);
        }
    },
    /** Assert two numbers match within a small tolerance. */
    assertClose = (actual: number, expected: number, label: string): void => {
        if (!(Math.abs(actual - expected) < 1e-4)) {
            throw new Error(`${label}: expected ${expected} got ${actual}`);
        }
    },
    /** Assert a float array matches an expected list. */
    assertList = (
        actual: ArrayLike<number>,
        expected: number[],
        label: string,
    ): void => {
        assertEqual(actual.length, expected.length, `${label} length`);
        for (let i = 0; i < expected.length; i += 1) {
            assertClose(actual[i], expected[i], `${label}[${i}]`);
        }
    },
    /** Assert a callback throws. */
    assertThrows = (run: () => void, label: string): void => {
        let threw = false;
        try {
            run();
        } catch {
            threw = true;
        }
        assert(threw, `${label}: expected throw`);
    },
    /** Assert an async callback rejects. */
    assertRejects = async (
        run: () => Promise<unknown>,
        label: string,
    ): Promise<void> => {
        let threw = false;
        try {
            await run();
        } catch {
            threw = true;
        }
        assert(threw, `${label}: expected rejection`);
    };

const
    /** ggml / gguf scalar type codes. */
    T_UINT32 = 4,
    T_FLOAT32 = 6,
    T_BOOL = 7,
    T_STRING = 8,
    T_ARRAY = 9,
    T_UINT64 = 10,
    /** Growable little endian byte writer. */
    byteWriter = (): ByteWriter => {
        const chunks: number[] = [];
        const push = (values: ArrayLike<number>): void => {
            for (let i = 0; i < values.length; i += 1) {
                chunks.push(values[i] & 0xff);
            }
        };
        const u64 = (value: number): void => {
            const big = BigInt(Math.trunc(value));
            for (let i = 0; i < 8; i += 1) {
                chunks.push(Number((big >> BigInt(8 * i)) & BigInt(0xff)));
            }
        };
        return {
            u8: (value) => {
                chunks.push(value & 0xff);
            },
            u16: (value) => {
                chunks.push(value & 0xff, (value >>> 8) & 0xff);
            },
            u32: (value) => {
                chunks.push(
                    value & 0xff,
                    (value >>> 8) & 0xff,
                    (value >>> 16) & 0xff,
                    (value >>> 24) & 0xff,
                );
            },
            i32: (value) => {
                const unsigned = value < 0 ? value + 0x100000000 : value;
                chunks.push(
                    unsigned & 0xff,
                    (unsigned >>> 8) & 0xff,
                    (unsigned >>> 16) & 0xff,
                    (unsigned >>> 24) & 0xff,
                );
            },
            u64,
            f32: (value) => {
                const buffer = new ArrayBuffer(4);
                new DataView(buffer).setFloat32(0, value, true);
                push(new Uint8Array(buffer));
            },
            f64: (value) => {
                const buffer = new ArrayBuffer(8);
                new DataView(buffer).setFloat64(0, value, true);
                push(new Uint8Array(buffer));
            },
            text: (value) => {
                const bytes = new TextEncoder().encode(value);
                u64(bytes.length);
                push(bytes);
            },
            raw: (value) => {
                push(value);
            },
            bytes: () => Uint8Array.from(chunks),
            size: () => chunks.length,
        };
    },
    /** Encode floats as little endian f32 bytes. */
    f32Data = (values: number[] | Float32Array): Uint8Array => {
        const
            out = new Uint8Array(values.length * 4),
            view = new DataView(out.buffer);
        for (let i = 0; i < values.length; i += 1) {
            view.setFloat32(i * 4, values[i], true);
        }
        return out;
    },
    /** Round a number to the nearest ieee half. */
    f32ToF16 = (value: number): number => {
        const
            buffer = new ArrayBuffer(4),
            view = new DataView(buffer);
        view.setFloat32(0, value, true);
        const
            bits = view.getUint32(0, true),
            sign = (bits >> 16) & 0x8000,
            exponent = (bits >> 23) & 0xff,
            mantissa = bits & 0x7fffff;
        if (exponent === 0xff) {
            return sign | 0x7c00 | (mantissa ? 0x200 : 0);
        }
        const half = exponent - 127 + 15;
        if (half <= 0) {
            return sign;
        }
        if (half >= 31) {
            return sign | 0x7c00;
        }
        return sign | (half << 10) | (mantissa >> 13);
    },
    /** Encode floats as little endian f16 bytes. */
    f16Data = (values: number[]): Uint8Array => {
        const
            out = new Uint8Array(values.length * 2),
            view = new DataView(out.buffer);
        for (let i = 0; i < values.length; i += 1) {
            view.setUint16(i * 2, f32ToF16(values[i]), true);
        }
        return out;
    },
    /** Build one Q4_K super block of 256 values. */
    q4KBlock = (options: {
        d: number;
        dmin: number;
        scales: number[];
        mins: number[];
        quants: number;
    }): Uint8Array => {
        const
            out = new Uint8Array(144),
            view = new DataView(out.buffer),
            { scales, mins } = options;
        view.setUint16(0, f32ToF16(options.d), true);
        view.setUint16(2, f32ToF16(options.dmin), true);
        for (let j = 0; j < 4; j += 1) {
            out[4 + j] = (scales[j] & 63) | ((scales[j + 4] >> 4) << 6);
            out[8 + j] = (mins[j] & 63) | ((mins[j + 4] >> 4) << 6);
            out[12 + j] = (scales[j + 4] & 15) | ((mins[j + 4] & 15) << 4);
        }
        out.fill(options.quants, 16);
        return out;
    },
    /** Build one Q6_K super block of 256 values. */
    q6KBlock = (options: {
        d: number;
        scales: number[];
        ql: number;
        qh: number;
    }): Uint8Array => {
        const
            out = new Uint8Array(210),
            view = new DataView(out.buffer);
        out.fill(options.ql, 0, 128);
        out.fill(options.qh, 128, 192);
        for (let i = 0; i < 16; i += 1) {
            out[192 + i] = options.scales[i] & 255;
        }
        view.setUint16(208, f32ToF16(options.d), true);
        return out;
    },
    /** Encode numbers as little endian i64 bytes. */
    i64Data = (values: number[]): Uint8Array => {
        const
            out = new Uint8Array(values.length * 8),
            view = new DataView(out.buffer);
        for (let i = 0; i < values.length; i += 1) {
            view.setBigInt64(i * 8, BigInt(values[i]), true);
        }
        return out;
    },
    /** Write one gguf metadata value. */
    writeGgufValue = (
        writer: ByteWriter,
        type: number,
        value: unknown,
    ): void => {
        switch (type) {
            case T_UINT32:
                writer.u32(value as number);
                break;
            case T_FLOAT32:
                writer.f32(value as number);
                break;
            case T_BOOL:
                writer.u8(value ? 1 : 0);
                break;
            case T_STRING:
                writer.text(value as string);
                break;
            case T_UINT64:
                writer.u64(value as number);
                break;
            case T_ARRAY: {
                const array = value as { element: number; items: unknown[] };
                writer.u32(array.element);
                writer.u64(array.items.length);
                for (const item of array.items) {
                    writeGgufValue(writer, array.element, item);
                }
                break;
            }
            default:
                throw new Error(`bad gguf type ${type}`);
        }
    },
    /** Build a minimal valid gguf file. */
    ggufBytes = (options: {
        metadata?: Array<[string, number, unknown]>;
        tensors: Array<{
            name: string;
            dims: number[];
            type: number;
            data: Uint8Array;
        }>;
    }): Uint8Array => {
        const
            metadata = options.metadata ?? [],
            { tensors } = options,
            writer = byteWriter();
        writer.raw(new TextEncoder().encode("GGUF"));
        writer.u32(3);
        writer.u64(tensors.length);
        writer.u64(metadata.length);
        for (const [key, type, value] of metadata) {
            writer.text(key);
            writer.u32(type);
            writeGgufValue(writer, type, value);
        }
        let running = 0;
        const offsets = tensors.map((tensor) => {
            const offset = running;
            running += tensor.data.length;
            return offset;
        });
        for (let i = 0; i < tensors.length; i += 1) {
            const tensor = tensors[i];
            writer.text(tensor.name);
            writer.u32(tensor.dims.length);
            for (const dim of tensor.dims) {
                writer.u64(dim);
            }
            writer.u32(tensor.type);
            writer.u64(offsets[i]);
        }
        const
            headerLength = writer.size(),
            padding = (32 - (headerLength % 32)) % 32;
        for (let i = 0; i < padding; i += 1) {
            writer.u8(0);
        }
        for (const tensor of tensors) {
            writer.raw(tensor.data);
        }
        return writer.bytes();
    },
    /** Build a minimal valid safetensors file. */
    safetensorsBytes = (options: {
        tensors: Array<{
            name: string;
            dtype: string;
            shape: number[];
            data: Uint8Array;
        }>;
        metadata?: Record<string, string>;
    }): Uint8Array => {
        const
            header: Record<string, unknown> = {},
            data: Uint8Array[] = [];
        let offset = 0;
        if (options.metadata) {
            header.__metadata__ = options.metadata;
        }
        for (const tensor of options.tensors) {
            header[tensor.name] = {
                dtype: tensor.dtype,
                shape: tensor.shape,
                data_offsets: [offset, offset + tensor.data.length],
            };
            data.push(tensor.data);
            offset += tensor.data.length;
        }
        let json = JSON.stringify(header);
        while (json.length % 4 !== 0) {
            json += " ";
        }
        const
            headerBytes = new TextEncoder().encode(json),
            out = new Uint8Array(8 + headerBytes.length + offset);
        new DataView(out.buffer).setBigUint64(0, BigInt(headerBytes.length), true);
        out.set(headerBytes, 8);
        let at = 8 + headerBytes.length;
        for (const bytes of data) {
            out.set(bytes, at);
            at += bytes.length;
        }
        return out;
    },
    /** A tiny in memory model store. */
    fakeStore = (input?: {
        texts?: Record<string, string>;
        bytes?: Record<string, Uint8Array>;
    }): FakeStore => {
        const
            texts = new Map<string, string>(Object.entries(input?.texts ?? {})),
            bytes = new Map<string, Uint8Array>(Object.entries(input?.bytes ?? {})),
            store: FakeStore = {
                texts,
                bytes,
                exists: async (key) => texts.has(key) || bytes.has(key),
                readText: async (key) => texts.get(key) ?? null,
                readBytes: async (key) => bytes.get(key) ?? null,
                writeText: async ({ key, text }) => {
                    texts.set(key, text);
                },
                writeStream: async ({ key, stream, onProgress }) => {
                    const reader = stream.getReader();
                    const parts: Uint8Array[] = [];
                    let total = 0;
                    for (; ;) {
                        const result = await reader.read();
                        if (result.done) {
                            break;
                        }
                        parts.push(result.value);
                        total += result.value.length;
                        onProgress?.(result.value.length);
                    }
                    const merged = new Uint8Array(total);
                    let offset = 0;
                    for (const part of parts) {
                        merged.set(part, offset);
                        offset += part.length;
                    }
                    bytes.set(key, merged);
                },
                remove: async (key) => {
                    texts.delete(key);
                    bytes.delete(key);
                },
            };
        return store;
    },
    /** Encode bytes as base64. */
    toBase64 = (bytes: Uint8Array): string => {
        let text = "";
        for (let i = 0; i < bytes.length; i += 1) {
            text += String.fromCharCode(bytes[i]);
        }
        return btoa(text);
    },
    /** Decode base64 to bytes. */
    fromBase64 = (text: string): Uint8Array => {
        const
            binary = atob(text),
            out = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i += 1) {
            out[i] = binary.charCodeAt(i);
        }
        return out;
    },
    /** Wrap bytes in a one chunk readable stream. */
    streamFromBytes = (bytes: Uint8Array): ReadableStream<Uint8Array> =>
        new ReadableStream<Uint8Array>({
            start(controller) {
                controller.enqueue(bytes);
                controller.close();
            },
        });

const
    /** Small raw whisper model config. */
    whisperRaw = (overrides?: Partial<RawModelConfig>): RawModelConfig => ({
        d_model: 4,
        encoder_layers: 1,
        decoder_layers: 1,
        encoder_attention_heads: 1,
        decoder_attention_heads: 1,
        encoder_ffn_dim: 4,
        decoder_ffn_dim: 4,
        vocab_size: 6,
        num_mel_bins: 2,
        max_source_positions: 8,
        max_target_positions: 8,
        max_length: 10,
        activation_function: "gelu",
        scale_embedding: false,
        decoder_start_token_id: 1,
        bos_token_id: 1,
        eos_token_id: 2,
        pad_token_id: 0,
        suppress_tokens: [3],
        begin_suppress_tokens: [3],
        forced_decoder_ids: [[1, 2]],
        ...overrides,
    }),
    /** Small raw whisper generation config. */
    whisperGeneration = (
        overrides?: Partial<RawGenerationConfig>,
    ): RawGenerationConfig => ({
        alignment_heads: [[0, 0]],
        begin_suppress_tokens: [3],
        bos_token_id: 1,
        decoder_start_token_id: 1,
        eos_token_id: 2,
        forced_decoder_ids: [],
        is_multilingual: true,
        lang_to_id: { "<|en|>": 4, "<|de|>": 5 },
        max_initial_timestamp_index: 1,
        max_length: 10,
        no_timestamps_token_id: 6,
        pad_token_id: 0,
        prev_sot_token_id: 7,
        suppress_tokens: [3],
        task_to_id: { translate: 8, transcribe: 9 },
        ...overrides,
    }),
    /** Small raw whisper preprocessor config. */
    whisperPreprocessor = (
        overrides?: Partial<RawPreprocessorConfig>,
    ): RawPreprocessorConfig => ({
        chunk_length: 30,
        feature_size: 2,
        hop_length: 4,
        mel_filters: [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1],
        ],
        n_fft: 8,
        n_samples: 64,
        nb_max_frames: 32,
        padding_side: "right",
        padding_value: 0,
        sampling_rate: 16000,
        ...overrides,
    }),
    /** All whisper graph tensors and their shapes. */
    whisperShapes = (config: ModelConfig): Record<string, number[]> => {
        const
            d = config.dModel,
            e = config.encoderLayers,
            l = config.decoderLayers,
            ef = config.encoderFfnDim,
            df = config.decoderFfnDim,
            v = config.vocabSize,
            m = config.numMelBins,
            shapes: Record<string, number[]> = {
                "model.encoder.conv1.weight": [d, m, 3],
                "model.encoder.conv1.bias": [d],
                "model.encoder.conv2.weight": [d, d, 3],
                "model.encoder.conv2.bias": [d],
                "model.encoder.embed_positions.weight": [config.maxSourcePositions, d],
                "model.encoder.layer_norm.weight": [d],
                "model.encoder.layer_norm.bias": [d],
                "model.decoder.embed_tokens.weight": [v, d],
                "model.decoder.embed_positions.weight": [config.maxTargetPositions, d],
                "model.decoder.layer_norm.weight": [d],
                "model.decoder.layer_norm.bias": [d],
            };
        for (let layer = 0; layer < e; layer += 1) {
            const prefix = `model.encoder.layers.${layer}`;
            Object.assign(shapes, {
                [`${prefix}.self_attn_layer_norm.weight`]: [d],
                [`${prefix}.self_attn_layer_norm.bias`]: [d],
                [`${prefix}.self_attn.q_proj.weight`]: [d, d],
                [`${prefix}.self_attn.q_proj.bias`]: [d],
                [`${prefix}.self_attn.k_proj.weight`]: [d, d],
                [`${prefix}.self_attn.v_proj.weight`]: [d, d],
                [`${prefix}.self_attn.v_proj.bias`]: [d],
                [`${prefix}.self_attn.out_proj.weight`]: [d, d],
                [`${prefix}.self_attn.out_proj.bias`]: [d],
                [`${prefix}.final_layer_norm.weight`]: [d],
                [`${prefix}.final_layer_norm.bias`]: [d],
                [`${prefix}.fc1.weight`]: [ef, d],
                [`${prefix}.fc1.bias`]: [ef],
                [`${prefix}.fc2.weight`]: [d, ef],
                [`${prefix}.fc2.bias`]: [d],
            });
        }
        for (let layer = 0; layer < l; layer += 1) {
            const prefix = `model.decoder.layers.${layer}`;
            Object.assign(shapes, {
                [`${prefix}.self_attn_layer_norm.weight`]: [d],
                [`${prefix}.self_attn_layer_norm.bias`]: [d],
                [`${prefix}.self_attn.q_proj.weight`]: [d, d],
                [`${prefix}.self_attn.q_proj.bias`]: [d],
                [`${prefix}.self_attn.k_proj.weight`]: [d, d],
                [`${prefix}.self_attn.v_proj.weight`]: [d, d],
                [`${prefix}.self_attn.v_proj.bias`]: [d],
                [`${prefix}.self_attn.out_proj.weight`]: [d, d],
                [`${prefix}.self_attn.out_proj.bias`]: [d],
                [`${prefix}.encoder_attn_layer_norm.weight`]: [d],
                [`${prefix}.encoder_attn_layer_norm.bias`]: [d],
                [`${prefix}.encoder_attn.q_proj.weight`]: [d, d],
                [`${prefix}.encoder_attn.q_proj.bias`]: [d],
                [`${prefix}.encoder_attn.k_proj.weight`]: [d, d],
                [`${prefix}.encoder_attn.v_proj.weight`]: [d, d],
                [`${prefix}.encoder_attn.v_proj.bias`]: [d],
                [`${prefix}.encoder_attn.out_proj.weight`]: [d, d],
                [`${prefix}.encoder_attn.out_proj.bias`]: [d],
                [`${prefix}.final_layer_norm.weight`]: [d],
                [`${prefix}.final_layer_norm.bias`]: [d],
                [`${prefix}.fc1.weight`]: [df, d],
                [`${prefix}.fc1.bias`]: [df],
                [`${prefix}.fc2.weight`]: [d, df],
                [`${prefix}.fc2.bias`]: [d],
            });
        }
        return shapes;
    },
    /** A read only float tensor store over zero weights. */
    tensorStoreFromShapes = (
        shapes: Record<string, number[]>,
    ): TensorStore => {
        const map = new Map<string, Float32Array>();
        for (const [name, shape] of Object.entries(shapes)) {
            map.set(name, new Float32Array(product(shape)));
        }
        return {
            count: map.size,
            names: () => Array.from(map.keys()),
            has: (name) => map.has(name),
            info: (name) => {
                const value = map.get(name);
                if (!value) {
                    throw new Error(`missing tensor ${name}`);
                }
                return {
                    name,
                    dtype: "F32",
                    shape: shapes[name],
                    dataOffset: 0,
                    numel: value.length,
                };
            },
            get: (name) => {
                const value = map.get(name);
                if (!value) {
                    throw new Error(`missing tensor ${name}`);
                }
                return value;
            },
        };
    },
    /** A tokenizer backed by an id to text map. */
    stubTokenizer = (texts: Record<number, string>): Tokenizer => ({
        decode: (ids) => ids.map((id) => texts[id] ?? "").join(""),
        tokenText: (id) => texts[id] ?? "",
        vocabSize: Object.keys(texts).length,
    }),
    /** Constants small enough for synthetic decoding. */
    testConstants = (): Constants => ({
        eosTokenId: 2,
        bosTokenId: 1,
        padTokenId: 0,
        decoderStartTokenId: 1,
        isMultilingual: true,
        startOfTranscript: 3,
        languageIds: new Map([["en", 4], ["de", 5]]),
        idToLanguage: new Map([["4", "en"], ["5", "de"]]),
        translateToken: 6,
        transcribeToken: 7,
        startOfLm: 8,
        startOfPrev: 9,
        noSpeechToken: 10,
        noTimestampsToken: 11,
        timestampBegin: 12,
        timestampStep: 0.02,
        suppressTokens: [],
        beginSuppressTokens: [],
        alignmentHeads: [[0, 0]],
        maxInitialTimestampIndex: 1,
    }),
    /** A small mel config for the graph tests. */
    testMel = (): MelConfig => ({
        samplingRate: 16000,
        nFft: 8,
        hopLength: 4,
        featureSize: 2,
        chunkLength: 30,
        nSamples: 64,
        maxFrames: 32,
    }),
    /** A complete synthetic whisper model. */
    graphModel = (): LoadedModel => {
        const config = toModelConfig(whisperRaw({
            vocab_size: 16,
            max_target_positions: 16,
            max_length: 12,
        }));
        return {
            kind: ModelFamily.Whisper,
            name: ModelId.WhisperBase,
            config,
            constants: testConstants(),
            mel: testMel(),
            melFilters: new Float32Array(config.numMelBins * 5),
            tokenizer: stubTokenizer({ 0: "a", 1: "b", 2: "c", 3: "d" }),
            tensors: tensorStoreFromShapes(whisperShapes(config)),
        };
    };

const
    /** Token types of the synthetic SenseVoice vocabulary. */
    senseVoiceTypes = [3, 3, 3, 1, 1, 1, 1, 1],
    /** Tokens of the synthetic SenseVoice vocabulary. */
    senseVoiceTokens = [
        "<blank>",
        "<|zh|>",
        "<|en|>",
        "\u2581hello",
        "\u2581world",
        "!",
        "\u2581foo",
        "\u2581bar",
    ],
    /** A small SenseVoice config matching the synthetic tokens. */
    senseVoiceConfig = (): SenseVoiceConfig => ({
        dInput: 2,
        dModel: 4,
        heads: 1,
        headDim: 4,
        ffnDim: 4,
        kernelSize: 1,
        mainBlocks: 2,
        tpBlocks: 1,
        vocabSize: senseVoiceTokens.length,
        eps: 1e-12,
        languageIds: new Map([
            ["auto", 0],
            ["zh", 1],
            ["en", 2],
            ["yue", 3],
            ["ja", 4],
            ["ko", 5],
            ["nospeech", 6],
        ]),
        eventEmotion: [7, 8],
        withItn: 9,
        withoutItn: 10,
    }),
    /** Every SenseVoice tensor and its pytorch shape. */
    senseVoiceSpecs = (config: SenseVoiceConfig): TensorSpec[] => {
        const
            d = config.dModel,
            di = config.dInput,
            f = config.ffnDim,
            k = config.kernelSize,
            v = config.vocabSize,
            ones = (length: number): number[] =>
                new Array<number>(length).fill(1),
            specs: TensorSpec[] = [
                { name: "frontend.cmvn.shift", shape: [di], values: [1, 2] },
                { name: "frontend.cmvn.scale", shape: [di], values: [0.5, 2] },
                { name: "enc.embed.weight", shape: [v + 8, di] },
                { name: "enc.after_norm.weight", shape: [d], values: ones(d) },
                { name: "enc.after_norm.bias", shape: [d] },
                { name: "enc.tp_norm.weight", shape: [d], values: ones(d) },
                { name: "enc.tp_norm.bias", shape: [d] },
                { name: "ctc.head.weight", shape: [v, d] },
                { name: "ctc.head.bias", shape: [v] },
            ],
            pushBlock = (base: string, inDim: number, normDim: number): void => {
                specs.push(
                    {
                        name: `${base}.norm_attn.weight`,
                        shape: [normDim],
                        values: ones(normDim),
                    },
                    { name: `${base}.norm_attn.bias`, shape: [normDim] },
                    { name: `${base}.attn.qkv.weight`, shape: [3 * d, inDim] },
                    { name: `${base}.attn.qkv.bias`, shape: [3 * d] },
                    { name: `${base}.attn.fsmn.weight`, shape: [d * k] },
                    { name: `${base}.attn.out.weight`, shape: [d, d] },
                    { name: `${base}.attn.out.bias`, shape: [d] },
                    {
                        name: `${base}.norm_ffn.weight`,
                        shape: [d],
                        values: ones(d),
                    },
                    { name: `${base}.norm_ffn.bias`, shape: [d] },
                    { name: `${base}.ffn.fc1.weight`, shape: [f, d] },
                    { name: `${base}.ffn.fc1.bias`, shape: [f] },
                    { name: `${base}.ffn.fc2.weight`, shape: [d, f] },
                    { name: `${base}.ffn.fc2.bias`, shape: [d] },
                );
            };
        pushBlock("enc.encoders0.0", di, di);
        for (let i = 0; i < config.mainBlocks - 1; i += 1) {
            pushBlock(`enc.encoders.${i}`, d, d);
        }
        for (let i = 0; i < config.tpBlocks; i += 1) {
            pushBlock(`enc.tp_encoders.${i}`, d, d);
        }
        return specs;
    },
    /** Build the full SenseVoice gguf used by the model tests. */
    senseVoiceGguf = (config: SenseVoiceConfig): Uint8Array => {
        const
            specs = senseVoiceSpecs(config),
            tensors = specs.map((spec) => ({
                name: spec.name,
                dims: [...spec.shape].reverse(),
                type: 0,
                data: f32Data(
                    spec.values ?? new Array<number>(product(spec.shape)).fill(0),
                ),
            })),
            metadata: Array<[string, number, unknown]> = [
                ["stt.sensevoice.encoder.d_model", T_UINT32, config.dModel],
                ["stt.sensevoice.encoder.n_heads", T_UINT32, config.heads],
                ["stt.sensevoice.encoder.d_input", T_UINT32, config.dInput],
                ["stt.sensevoice.encoder.d_ff", T_UINT32, config.ffnDim],
                ["stt.sensevoice.encoder.kernel_size", T_UINT32, config.kernelSize],
                ["stt.sensevoice.encoder.n_blocks", T_UINT32, config.mainBlocks],
                ["stt.sensevoice.encoder.tp_blocks", T_UINT32, config.tpBlocks],
                ["stt.sensevoice.special.lang_auto", T_UINT32, 0],
                ["stt.sensevoice.special.lang_zh", T_UINT32, 1],
                ["stt.sensevoice.special.lang_en", T_UINT32, 2],
                ["stt.sensevoice.special.lang_yue", T_UINT32, 3],
                ["stt.sensevoice.special.lang_ja", T_UINT32, 4],
                ["stt.sensevoice.special.lang_ko", T_UINT32, 5],
                ["stt.sensevoice.special.lang_nospeech", T_UINT32, 6],
                ["stt.sensevoice.special.event_speech", T_UINT32, 7],
                ["stt.sensevoice.special.emotion_neutral", T_UINT32, 8],
                ["stt.sensevoice.special.withitn", T_UINT32, 9],
                ["stt.sensevoice.special.woitn", T_UINT32, 10],
                ["stt.frontend.sample_rate", T_UINT32, 16000],
                ["stt.frontend.num_mels", T_UINT32, config.dInput],
                ["stt.frontend.n_fft", T_UINT32, 4],
                ["stt.frontend.win_length", T_UINT32, 4],
                ["stt.frontend.hop_length", T_UINT32, 2],
                ["stt.frontend.lfr_m", T_UINT32, 1],
                ["stt.frontend.lfr_n", T_UINT32, 1],
                ["stt.frontend.upscale_samples", T_BOOL, false],
                [
                    "tokenizer.ggml.tokens",
                    T_ARRAY,
                    { element: T_STRING, items: senseVoiceTokens },
                ],
                [
                    "tokenizer.ggml.token_type",
                    T_ARRAY,
                    { element: T_UINT32, items: senseVoiceTypes },
                ],
            ];
        return ggufBytes({ metadata, tensors });
    },
    /** Shapes of the SenseVoice tensors. */
    senseVoiceShapes = (config: SenseVoiceConfig): Record<string, number[]> => {
        const shapes: Record<string, number[]> = {};
        for (const spec of senseVoiceSpecs(config)) {
            shapes[spec.name] = spec.shape;
        }
        return shapes;
    },
    /** A SenseVoice weight store over zero tensors. */
    senseVoiceWeights = (
        shapes: Record<string, number[]>,
        overrides?: Record<string, number[]>,
    ): SenseVoiceWeights => {
        const map = new Map<string, Float32Array>();
        for (const [name, shape] of Object.entries(shapes)) {
            const values = overrides?.[name]
                ?? new Array<number>(product(shape)).fill(0);
            map.set(name, Float32Array.from(values));
        }
        return {
            names: () => Array.from(map.keys()),
            has: (name) => map.has(name),
            shape: (name) => shapes[name] ?? [],
            f32: (name) => {
                const value = map.get(name);
                if (!value) {
                    throw new Error(`missing tensor ${name}`);
                }
                return value;
            },
        };
    },
    /** A gguf file literal for the config and tokenizer tests. */
    ggufFileLiteral = (
        metadata: Array<[string, GgufValue]>,
    ): GgufFile => ({
        version: 3,
        metadata: new Map<string, GgufValue>(metadata),
        tensors: new Map(),
        dataOffset: 0,
        buffer: new Uint8Array(0),
    });

/** One fake dictation worker driven by the test. */
class DictationWorkerStub {
    onmessage: ((event: MessageEvent<DictationMessage>) => void) | null = null;
    onerror: ((event: ErrorEvent) => void) | null = null;
    messages: DictationRequest[] = [];
    handler: ((request: DictationRequest) => void) | undefined;

    postMessage(request: DictationRequest): void {
        this.messages.push(request);
        this.handler?.(request);
    }

    terminate(): void {
        return;
    }

    reply(message: DictationMessage): void {
        this.onmessage?.({
            data: message,
        } as MessageEvent<DictationMessage>);
    }
}

/** One fake microphone capture. */
interface FakeCapture {
    options: MicrophoneCapture;
    handle: MicrophoneCaptureHandle;
    counts: {
        paused: number;
        resumed: number;
        stopped: number;
    };
    create: DictationCapture;
}

const
    /** Worker shared by the dictation tests. */
    dictationWorker = new DictationWorkerStub(),
    dictationWorkerInstance = dictationWorker as unknown as Worker,
    /** Build one fake capture with call counters. */
    fakeCapture = (): FakeCapture => {
        const
            counts = { paused: 0, resumed: 0, stopped: 0 },
            options: MicrophoneCapture = {
                onChunk: () => undefined,
                onError: () => undefined,
            },
            handle: MicrophoneCaptureHandle = {
                ready: Promise.resolve(),
                stop: async (): Promise<void> => {
                    counts.stopped += 1;
                },
                pause: async (): Promise<void> => {
                    counts.paused += 1;
                },
                resume: async (): Promise<void> => {
                    counts.resumed += 1;
                },
            };
        return {
            options,
            handle,
            counts,
            create: (input) => {
                options.onChunk = input.onChunk;
                options.onError = input.onError;
                options.onFrequency = input.onFrequency;
                return handle;
            },
        };
    };

/** Every test, each deferred behind a function. */
const definitions: Array<{
    name: string;
    run: () => void | Promise<void>;
}> = [
        {
            name: "tensor/binary dtypeSize",
            run: () => {
                assertEqual(dtypeSize("F32"), 4, "F32");
                assertEqual(dtypeSize("I32"), 4, "I32");
                assertEqual(dtypeSize("F16"), 2, "F16");
                assertEqual(dtypeSize("BF16"), 2, "BF16");
                assertEqual(dtypeSize("I16"), 2, "I16");
                assertEqual(dtypeSize("I64"), 8, "I64");
                assertEqual(dtypeSize("U8"), 1, "U8");
                assertEqual(dtypeSize("I8"), 1, "I8");
                assertEqual(dtypeSize("BOOL"), 1, "BOOL");
                assertThrows(() => dtypeSize("X"), "unknown dtype");
            },
        },
        {
            name: "tensor/binary product",
            run: () => {
                assertEqual(product([2, 3, 4]), 24, "product");
                assertEqual(product([]), 1, "empty product");
                assertEqual(product([5]), 5, "single");
            },
        },
        {
            name: "tensor/binary f16ToF32",
            run: () => {
                assertEqual(f16ToF32(0x0000), 0, "zero");
                assertEqual(f16ToF32(0x3c00), 1, "one");
                assertEqual(f16ToF32(0xbc00), -1, "minus one");
                assertEqual(f16ToF32(0x7c00), Infinity, "+inf");
                assertEqual(f16ToF32(0xfc00), -Infinity, "-inf");
                assert(Number.isNaN(f16ToF32(0x7c01)), "nan");
                assertClose(f16ToF32(0x0001), 2 ** -14 / 1024, "subnormal");
                assertEqual(f16ToF32(0x4000), 2, "two");
            },
        },
        {
            name: "tensor/binary readF16",
            run: () => {
                const bytes = f16Data([1, 0.5, -2]);
                const values = readF16({ bytes, offset: 0, count: 3 });
                assertList(values, [1, 0.5, -2], "readF16");
                const offset = readF16({ bytes, offset: 2, count: 2 });
                assertList(offset, [0.5, -2], "readF16 offset");
                const out = new Float32Array(3);
                assert(
                    readF16({ bytes, offset: 0, count: 3, out }) === out,
                    "readF16 out",
                );
                assertList(out, [1, 0.5, -2], "readF16 out values");
            },
        },
        {
            name: "tensor/binary parseSafetensorsJson",
            run: () => {
                const bytes = safetensorsBytes({
                    tensors: [{
                        name: "t",
                        dtype: "F32",
                        shape: [2],
                        data: f32Data([1, 2]),
                    }],
                    metadata: { format: "pt" },
                });
                const parsed = parseSafetensorsJson(bytes);
                assertEqual(parsed.entries.length, 1, "entry count");
                assertEqual(parsed.entries[0][0], "t", "entry name");
                assertEqual(parsed.metadata.format, "pt", "metadata");
                assertEqual(parsed.dataStart % 4, 0, "aligned");
            },
        },
        {
            name: "tensor/kernels activations",
            run: () => {
                const gelu = new Float32Array([0, 1, -1]);
                geluInPlace({ values: gelu, length: 3 });
                assertClose(gelu[0], 0, "gelu zero");
                assertClose(gelu[1], 0.8413447, "gelu one");
                assertClose(gelu[2], -0.1586553, "gelu minus one");
                const relu = new Float32Array([-1, 2, -3, 4]);
                reluInPlace({ values: relu, offset: 1, length: 2 });
                assertList(relu, [-1, 2, 0, 4], "relu range");
                const silu = new Float32Array([0, 1]);
                siluInPlace({ values: silu, length: 2 });
                assertClose(silu[0], 0, "silu zero");
                assertClose(silu[1], 0.7310586, "silu one");
            },
        },
        {
            name: "tensor/kernels elementwise",
            run: () => {
                const a = new Float32Array([1, 2, 3]);
                const b = new Float32Array([4, 5, 6]);
                assertList(addArrays({ a, b }), [5, 7, 9], "add");
                assertList(addArrays({ a, b, scaleB: 2 }), [9, 12, 15], "scaled add");
                addInPlace({ a, b, scaleB: 0.5 });
                assertList(a, [3, 4.5, 6], "addInPlace");
            },
        },
        {
            name: "tensor/kernels linear",
            run: () => {
                const input = new Float32Array([1, 2]);
                const weight = new Float32Array([1, 0, 0, 1]);
                assertList(
                    linear({ input, weight, rows: 1, inDim: 2, outDim: 2 }),
                    [1, 2],
                    "identity",
                );
                assertList(
                    linear({
                        input,
                        weight,
                        bias: new Float32Array([10, 20]),
                        rows: 1,
                        inDim: 2,
                        outDim: 2,
                    }),
                    [11, 22],
                    "bias",
                );
            },
        },
        {
            name: "tensor/kernels layerNorm",
            run: () => {
                const input = new Float32Array([1, 2, 3]);
                const meanOut = new Float32Array(1);
                const rstdOut = new Float32Array(1);
                const out = layerNorm({
                    input,
                    weight: new Float32Array([1, 1, 1]),
                    bias: new Float32Array([0, 0, 0]),
                    rows: 1,
                    dims: 3,
                    eps: 0,
                    meanOut,
                    rstdOut,
                });
                assertClose(meanOut[0], 2, "mean");
                assertClose(rstdOut[0], 1 / Math.sqrt(2 / 3), "rstd");
                assertList(out, [-1.2247449, 0, 1.2247449], "normed");
            },
        },
        {
            name: "tensor/kernels softmaxRow",
            run: () => {
                const values = new Float32Array([1, 2, 3]);
                softmaxRow({ values, offset: 0, length: 3 });
                const denom = Math.exp(0) + Math.exp(1) + Math.exp(2);
                assertList(
                    values,
                    [Math.exp(0) / denom, Math.exp(1) / denom, Math.exp(2) / denom],
                    "softmax",
                );
            },
        },
        {
            name: "tensor/kernels attention",
            run: () => {
                const q = new Float32Array([1, 0, 0, 1]);
                const k = new Float32Array([1, 0, 0, 1, 1, 1]);
                const v = new Float32Array([1, 2, 3, 4, 5, 6]);
                const out = attention({
                    q,
                    k,
                    v,
                    seqQ: 2,
                    seqK: 3,
                    headDim: 2,
                    heads: 1,
                });
                assertEqual(out.length, 4, "attention output");
                const causal = attention({
                    q,
                    k,
                    v,
                    seqQ: 2,
                    seqK: 3,
                    headDim: 2,
                    heads: 1,
                    causal: true,
                });
                assertEqual(causal.length, 4, "causal output");
            },
        },
        {
            name: "tensor/kernels crossAttentionOne",
            run: () => {
                const weightsOut = new Float32Array(2);
                const out = crossAttentionOne({
                    q: new Float32Array([1, 0]),
                    k: new Float32Array([1, 0, 0, 1]),
                    v: new Float32Array([1, 0, 0, 1]),
                    heads: 1,
                    headDim: 2,
                    seqK: 2,
                    captureHeads: [0],
                    weightsOut,
                });
                assertEqual(out.length, 2, "cross output");
                assertClose(weightsOut[0] + weightsOut[1], 1, "weights sum");
            },
        },
        {
            name: "tensor/kernels argmax",
            run: () => {
                const values = new Float32Array([1, 9, 3, 9]);
                assertEqual(argmax({ values }), 1, "argmax");
                assertEqual(argmax({ values, start: 2 }), 3, "argmax start");
                assertEqual(argmax({ values, start: 0, end: 2 }), 1, "argmax end");
            },
        },
        {
            name: "tensor/wasm arena reuse",
            run: () => {
                const
                    weight = new Float32Array(256 * 4),
                    input = new Float32Array(4);
                weight.fill(0.5);
                input.fill(2);
                const run = (): Float32Array => linear({
                    input,
                    weight,
                    rows: 1,
                    inDim: 4,
                    outDim: 256,
                });
                assertClose(run()[0], 4, "arena linear");
                releaseTensor(weight);
                const base = getKernel()?.memory.buffer.byteLength ?? 0;
                for (let round = 0; round < 8; round += 1) {
                    assertClose(run()[0], 4, "arena linear again");
                    releaseTensor(weight);
                }
                const after = getKernel()?.memory.buffer.byteLength ?? 0;
                assertEqual(after, base, "arena reused");
            },
        },
        {
            name: "tensor/math nextPow2 and reflect",
            run: () => {
                assertEqual(nextPow2(1), 1, "one");
                assertEqual(nextPow2(3), 4, "three");
                assertEqual(nextPow2(5), 8, "five");
                assertEqual(nextPow2(16), 16, "sixteen");
                assertEqual(reflect({ index: -1, length: 5 }), 1, "left");
                assertEqual(reflect({ index: 5, length: 5 }), 3, "right");
                assertEqual(reflect({ index: 7, length: 5 }), 1, "wrapped");
            },
        },
        {
            name: "speech/mel buildFftPlan",
            run: () => {
                const plan = buildFftPlan(4);
                assertEqual(plan.n, 4, "n");
                assertEqual(plan.m, 8, "m");
                assertEqual(plan.chirpRe.length, 4, "chirp");
                assertEqual(plan.bitrev.length, 8, "bitrev");
                assertEqual(plan.cosTable.length, 5, "cos table");
            },
        },
        {
            name: "speech/mel powerSpectrum",
            run: () => {
                const
                    plan = buildFftPlan(4),
                    out = new Float32Array(3);
                powerSpectrum({
                    plan,
                    frame: new Float32Array([1, 1, 1, 1]),
                    window: new Float32Array([1, 1, 1, 1]),
                    bins: 3,
                    scratchRe: new Float32Array(plan.m),
                    scratchIm: new Float32Array(plan.m),
                    convRe: new Float32Array(plan.m),
                    convIm: new Float32Array(plan.m),
                    kernelRe: plan.kernelRe,
                    kernelIm: plan.kernelIm,
                    out,
                    outOffset: 0,
                    stride: 1,
                });
                assertClose(out[0], 16, "dc power");
                assertClose(out[1], 0, "first bin");
                assertClose(out[2], 0, "second bin");
            },
        },
        {
            name: "speech/mel melEnergies",
            run: () => {
                const out = melEnergies({
                    filters: new Float32Array([1, 2]),
                    power: new Float32Array([1, 2, 3, 4]),
                    bins: 2,
                    mels: 1,
                    frames: 2,
                });
                assertList(out, [7, 10], "mel energies");
            },
        },
        {
            name: "speech/mel computeMel",
            run: () => {
                const
                    mel: MelConfig = testMel(),
                    filters = new Float32Array(mel.featureSize * 5).fill(0.1),
                    pcm = new Float32Array(64).fill(0.25),
                    result = computeMel({ pcm, mel, filters });
                assertEqual(result.frames, 16, "frames");
                assertEqual(result.melBins, 2, "mel bins");
                assertEqual(result.values.length, 32, "values length");
                const short = computeMel({
                    pcm,
                    mel,
                    filters,
                    padToChunk: false,
                });
                assertEqual(short.frames, 16, "short frames");
            },
        },
        {
            name: "speech/segment planChunks",
            run: () => {
                const chunks = planChunks({
                    pcm: new Float32Array(2000).fill(1),
                    samplingRate: 1000,
                });
                assertEqual(chunks.length, 1, "one chunk");
                assertClose(chunks[0].start, 0, "chunk start");
                assertClose(chunks[0].end, 2, "chunk end");
                const empty = planChunks({
                    pcm: new Float32Array(5),
                    samplingRate: 1000,
                });
                assertEqual(empty.length, 0, "empty");
            },
        },
        {
            name: "speech/segment dedupeBoundary",
            run: () => {
                assertEqual(
                    dedupeBoundary({ previous: "the cat sat", next: "sat on the mat" }),
                    "on the mat",
                    "dedupe",
                );
                assertEqual(
                    dedupeBoundary({ previous: "hello", next: "world" }),
                    "world",
                    "no overlap",
                );
                assertEqual(
                    dedupeBoundary({
                        previous: "你好世界，今天天气",
                        next: "今天天气晴朗。",
                    }),
                    "晴朗。",
                    "cjk overlap",
                );
                assertEqual(
                    joinText({ previous: "hello", next: "world" }),
                    "hello world",
                    "join latin",
                );
                assertEqual(
                    joinText({ previous: "你好", next: "世界" }),
                    "你好世界",
                    "join cjk",
                );
                assertEqual(
                    joinText({ previous: "hello", next: "." }),
                    "hello.",
                    "join punctuation",
                );
                assertEqual(
                    joinText({ previous: "hello ", next: "world" }),
                    "hello world",
                    "join spaced",
                );
            },
        },
        {
            name: "models/specs constants",
            run: () => {
                assertEqual(HF_BASE, "https://huggingface.co", "hf base");
                assertEqual(DEFAULT_MODEL, ModelId.WhisperBase, "default");
                assertEqual(DEFAULT_QUALITY, Quality.High, "default quality");
                assertEqual(MODEL_IDS.length, 2, "model ids");
                assertEqual(CHECKPOINT_IDS.length, 3, "checkpoints");
                assert(MODEL_SPECS[ModelId.WhisperBase] !== undefined, "spec");
                assertEqual(
                    weightFiles(MODEL_SPECS[ModelId.WhisperBase]).join(","),
                    [
                        "whisper-base-F16.gguf",
                        "whisper-base-Q8_0.gguf",
                        "whisper-base-Q4_K_M.gguf",
                    ].join(","),
                    "weight files",
                );
                const checkpoint = checkpointFiles(
                    MODEL_SPECS[ModelId.WhisperBase],
                );
                assert(checkpoint.includes("config.json"), "checkpoint text");
                assert(checkpoint.includes("whisper-base-F16.gguf"), "weight high");
                assert(checkpoint.includes("whisper-base-Q4_K_M.gguf"), "weight low");
                assertEqual(
                    checkpointKey({
                        checkpoint: ModelId.WhisperBase,
                        file: "config.json",
                    }),
                    "whisper-base/config.json",
                    "model key",
                );
                assertEqual(
                    MODEL_SPECS[ModelId.WhisperBase].family,
                    ModelFamily.Whisper,
                    "family",
                );
                assertEqual(
                    resolveSpec({
                        model: ModelId.WhisperBase,
                        language: Language.en,
                    }).repo,
                    "handy-computer/whisper-base.en-gguf",
                    "english checkpoint",
                );
                assertEqual(
                    resolveSpec({
                        model: ModelId.WhisperBase,
                        language: Language.en,
                    }).textRepo,
                    "openai/whisper-base.en",
                    "english text files",
                );
                assertEqual(
                    resolveSpec({
                        model: ModelId.WhisperBase,
                        language: Language.auto,
                    }).weightFile,
                    "whisper-base-F16.gguf",
                    "high quality weight",
                );
                assertEqual(
                    resolveSpec({
                        model: ModelId.WhisperBase,
                        quality: Quality.Medium,
                    }).weightFile,
                    "whisper-base-Q8_0.gguf",
                    "medium quality weight",
                );
                assertEqual(
                    resolveSpec({
                        model: ModelId.WhisperBase,
                        quality: Quality.Low,
                    }).weightFile,
                    "whisper-base-Q4_K_M.gguf",
                    "low quality weight",
                );
                assertEqual(
                    resolveSpec({
                        model: ModelId.WhisperBase,
                        language: Language.en,
                        quality: Quality.Low,
                    }).weightFile,
                    "whisper-base.en-Q4_K_M.gguf",
                    "english low weight",
                );
                assertEqual(
                    resolveSpec({
                        model: ModelId.WhisperBase,
                        language: Language.en,
                    }).id,
                    ModelId.WhisperBase,
                    "english public id",
                );
                assertEqual(
                    resolveSpec({
                        model: ModelId.WhisperBase,
                        language: Language.auto,
                    }).repo,
                    "handy-computer/whisper-base-gguf",
                    "auto checkpoint",
                );
                assertEqual(
                    resolveSpec({ model: ModelId.WhisperBase }).repo,
                    "handy-computer/whisper-base-gguf",
                    "default checkpoint",
                );
                assertEqual(
                    resolveSpec({
                        model: ModelId.SenseVoiceSmall,
                        language: Language.en,
                    }).repo,
                    "handy-computer/SenseVoiceSmall-gguf",
                    "sensevoice checkpoint",
                );
            },
        },
        {
            name: "tensor/types enums",
            run: () => {
                assertEqual(ModelFamily.Whisper, "whisper", "whisper family");
                assertEqual(ModelFamily.SenseVoice, "sensevoice", "sensevoice family");
                assertEqual(ModelId.SenseVoiceSmall, "sensevoice-small", "sensevoice id");
                assertEqual(ModelId.WhisperBase, "whisper-base", "whisper id");
                assertEqual(Quality.High, "high", "high quality");
                assertEqual(Quality.Medium, "medium", "medium quality");
                assertEqual(Quality.Low, "low", "low quality");
                assertEqual(ThreadCount.Two, 2, "two threads");
                assertEqual(ThreadCount.Four, 4, "four threads");
                assertEqual(ThreadCount.Eight, 8, "eight threads");
            },
        },
        {
            name: "formats/gguf parseGguf",
            run: () => {
                const bytes = ggufBytes({
                    metadata: [
                        ["general.alignment", T_UINT32, 32],
                        ["test.key", T_STRING, "hello"],
                        ["test.array", T_ARRAY, {
                            element: T_UINT32,
                            items: [1, 2, 3],
                        }],
                    ],
                    tensors: [{
                        name: "t",
                        dims: [2, 3],
                        type: 0,
                        data: f32Data([1, 2, 3, 4, 5, 6]),
                    }],
                });
                const file = parseGguf({ bytes });
                assertEqual(file.version, 3, "version");
                assertEqual(file.metadata.get("test.key"), "hello", "metadata");
                assertEqual(file.dataOffset % 32, 0, "aligned");
                assertEqual(file.tensors.get("t")?.dims.length, 2, "rank");
                assertThrows(
                    () => parseGguf({ bytes: new Uint8Array(4) }),
                    "too small",
                );
                assertThrows(
                    () => parseGguf({ bytes: new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]) }),
                    "bad magic",
                );
            },
        },
        {
            name: "formats/ggufTensor fullCacheBytes",
            run: () => {
                const
                    mega = 1024 * 1024,
                    floor = GGUF_CACHE_BYTES / mega,
                    cap = GGUF_CACHE_CAP_BYTES / mega;
                assertEqual(fullCacheBytes(0) / mega, floor, "empty floor");
                assertEqual(fullCacheBytes(-1) / mega, floor, "negative floor");
                assertEqual(fullCacheBytes(64 * mega) / mega, floor, "small floor");
                assertEqual(fullCacheBytes(300 * mega) / mega, 300, "checkpoint");
                assertEqual(fullCacheBytes(4096 * mega) / mega, cap, "cap");
            },
        },
        {
            name: "formats/safetensors parseSafetensors",
            run: () => {
                const bytes = safetensorsBytes({
                    tensors: [
                        {
                            name: "a",
                            dtype: "F32",
                            shape: [2, 2],
                            data: f32Data([1, 2, 3, 4]),
                        },
                        {
                            name: "b",
                            dtype: "F32",
                            shape: [2],
                            data: f32Data([5, 6]),
                        },
                    ],
                });
                const file = parseSafetensors({ bytes });
                assertEqual(file.store.count, 2, "count");
                assert(file.store.has("a"), "has a");
                assertList(file.store.get("a"), [1, 2, 3, 4], "tensor a");
                assertList(file.store.get("b"), [5, 6], "tensor b");
                assertEqual(file.store.info("b").shape[0], 2, "shape");
                assert(file.store.names().includes("a"), "names");
                assertThrows(() => file.store.get("missing"), "missing");
            },
        },
        {
            name: "formats/safetensors assertWhisperShape",
            run: () => {
                const store = tensorStoreFromShapes({
                    "model.decoder.embed_tokens.weight": [4, 4],
                });
                assertWhisperShape({ store, vocabSize: 4, dModel: 4 });
                assertThrows(
                    () => assertWhisperShape({ store, vocabSize: 5, dModel: 4 }),
                    "wrong vocab",
                );
                const withProj = tensorStoreFromShapes({
                    "model.decoder.embed_tokens.weight": [4, 4],
                    proj_out: [4, 4],
                });
                assertThrows(
                    () => assertWhisperShape({ store: withProj, vocabSize: 4, dModel: 4 }),
                    "proj out",
                );
            },
        },
        {
            name: "whisper/config TOKENS and toModelConfig",
            run: () => {
                assertEqual(TOKENS.endOfText, 50257, "end of text");
                assertEqual(TOKENS.startOfTranscript, 50258, "sot");
                assertEqual(TOKENS.timestampBegin, 50364, "timestamp begin");
                const config = toModelConfig(whisperRaw());
                assertEqual(config.dModel, 4, "dModel");
                assertEqual(config.headDim, 4, "headDim");
                assertEqual(config.vocabSize, 6, "vocab");
                assertEqual(config.scaleEmbedding, false, "scale embedding");
                assertThrows(
                    () => toModelConfig(whisperRaw({ activation_function: "relu" })),
                    "bad activation",
                );
                assertThrows(
                    () => toModelConfig(whisperRaw({ scale_embedding: true })),
                    "scale embedding",
                );
            },
        },
        {
            name: "whisper/config toMelConfig and toMelFilters",
            run: () => {
                const raw = whisperPreprocessor();
                const mel = toMelConfig(raw);
                assertEqual(mel.samplingRate, 16000, "rate");
                assertEqual(mel.nFft, 8, "nFft");
                assertEqual(mel.featureSize, 2, "feature size");
                assertEqual(mel.nSamples, 64, "nSamples");
                const filters = toMelFilters(raw);
                assertEqual(filters.length, 10, "filter length");
                assertClose(filters[0], 0.1, "filter first");
                assertClose(filters[9], 0.1, "filter last");
            },
        },
        {
            name: "whisper/config toConstants",
            run: () => {
                const constants = toConstants(whisperGeneration());
                assertEqual(constants.startOfTranscript, 50258, "sot");
                assertEqual(constants.isMultilingual, true, "multilingual");
                assertEqual(constants.languageIds.get("en"), 4, "language id");
                assertEqual(constants.idToLanguage.get("4"), "en", "id to language");
                assertEqual(constants.translateToken, 8, "translate");
                assertEqual(constants.transcribeToken, 9, "transcribe");
                assertEqual(constants.noTimestampsToken, 6, "no timestamps");
                assertEqual(constants.timestampBegin, 7, "timestamp begin");
                assertEqual(constants.alignmentHeads.length, 1, "alignment");
            },
        },
        {
            name: "whisper/config assertSuppressMatch",
            run: () => {
                assertSuppressMatch({ a: [3, 1], b: [1, 3] });
                assertThrows(
                    () => assertSuppressMatch({ a: [3], b: [4] }),
                    "mismatch",
                );
            },
        },
        {
            name: "whisper/config loadConfigs",
            run: () => {
                const store = fakeStore({
                    texts: {
                        [checkpointKey({ checkpoint: ModelId.WhisperBase, file: "config.json" })]:
                            JSON.stringify(whisperRaw()),
                        [checkpointKey({
                            checkpoint: ModelId.WhisperBase,
                            file: "generation_config.json",
                        })]: JSON.stringify(whisperGeneration()),
                        [checkpointKey({
                            checkpoint: ModelId.WhisperBase,
                            file: "preprocessor_config.json",
                        })]: JSON.stringify(whisperPreprocessor()),
                    },
                });
                return loadConfigs({
                    store,
                    model: ModelId.WhisperBase,
                    checkpoint: ModelId.WhisperBase,
                }).then((loaded) => {
                    assertEqual(loaded.model.d_model, 4, "model");
                    assertEqual(loaded.generation.is_multilingual, true, "generation");
                    assertEqual(loaded.preprocessor.n_fft, 8, "preprocessor");
                });
            },
        },
        {
            name: "whisper/tokenizer loadTokenizer",
            run: () => {
                const store = fakeStore({
                    texts: {
                        [checkpointKey({
                            checkpoint: ModelId.WhisperBase,
                            file: "tokenizer.json",
                        })]:
                            JSON.stringify({
                                model: { vocab: { "h": 0, "e": 1, "l": 2, "o": 3 } },
                                added_tokens: [{ id: 4, content: "<|end|>", special: true }],
                            }),
                    },
                });
                return loadTokenizer({
                    store,
                    model: ModelId.WhisperBase,
                    checkpoint: ModelId.WhisperBase,
                }).then((tokenizer) => {
                    assertEqual(
                        tokenizer.decode([0, 1, 2, 2, 3]),
                        "hello",
                        "decode",
                    );
                    assertEqual(tokenizer.tokenText(0), "h", "token text");
                    assertEqual(tokenizer.tokenText(4), "", "special text");
                    assertEqual(tokenizer.vocabSize, 5, "vocab size");
                });
            },
        },
        {
            name: "whisper/words buildWords and buildSegments",
            run: () => {
                const tokenizer = stubTokenizer({
                    0: "hello",
                    1: " world",
                    2: "!",
                });
                const words = buildWords({
                    ids: [0, 1, 2],
                    tokenTimes: new Float32Array([0, 0.5, 1, 1.5]),
                    tokenizer,
                    timeOffset: 0,
                });
                assertEqual(words.length, 2, "word count");
                assertEqual(words[0].word, "hello", "first word");
                assertClose(words[0].start, 0, "first start");
                assertClose(words[0].end, 0.5, "first end");
                assertEqual(words[1].word, "world!", "second word");
                const segments = buildSegments({ words });
                assertEqual(segments.length, 1, "segment count");
                assertEqual(segments[0].text, "hello world!", "segment text");
                const paused = buildSegments({
                    words: [
                        { word: "hi", start: 0, end: 0.2 },
                        { word: "there", start: 2, end: 2.4 },
                    ],
                });
                assertEqual(paused.length, 2, "pause split");
            },
        },
        {
            name: "whisper/encoder encode",
            run: () => {
                const model = graphModel();
                const result = encode({
                    model,
                    mel: new Float32Array(model.config.numMelBins * 4).fill(0.1),
                });
                assertEqual(result.seq, 2, "seq");
                assertEqual(
                    result.output.length,
                    result.seq * model.config.dModel,
                    "output",
                );
                assertEqual(result.perLayer.length, 0, "no layers");
                const collected = encode({
                    model,
                    mel: new Float32Array(model.config.numMelBins * 4).fill(0.1),
                    collect: true,
                });
                assertEqual(collected.perLayer.length, 1, "collected");
            },
        },
        {
            name: "whisper/decoder decoderStep",
            run: () => {
                const
                    model = graphModel(),
                    encoderOutput = new Float32Array(2 * model.config.dModel).fill(0.1),
                    state = createDecoderState({ model, encoderOutput, encoderSeq: 2 }),
                    step = decoderStep({ state, tokenId: 0 });
                assertEqual(step.logits.length, model.config.vocabSize, "logits");
                assertEqual(step.captures.length, 0, "captures");
                const captured = decoderStep({
                    state,
                    tokenId: 1,
                    capture: [{ layer: 0, head: 0 }],
                });
                assertEqual(captured.captures.length, 1, "capture count");
                assertEqual(captured.captures[0].length, 2, "capture width");
            },
        },
        {
            name: "whisper/decoder detectLanguage",
            run: () => {
                const
                    model = graphModel(),
                    encoderOutput = new Float32Array(2 * model.config.dModel).fill(0.1),
                    state = createDecoderState({ model, encoderOutput, encoderSeq: 2 }),
                    result = detectLanguage({ state, constants: model.constants });
                assert(
                    result.language === "en" || result.language === "de",
                    "language",
                );
                assert(result.token === 4 || result.token === 5, "token");
            },
        },
        {
            name: "whisper/generate generate",
            run: () => {
                const model = graphModel();
                const output = generate({
                    model,
                    encoderOutput: new Float32Array(2 * model.config.dModel).fill(0.1),
                    encoderSeq: 2,
                    prompt: [3, 4],
                    timestamps: false,
                    maxTokens: 3,
                });
                assert(output.ids.length <= 3, "token cap");
                assertEqual(output.allIds.length, 2 + output.ids.length, "all ids");
                assertEqual(typeof output.eos, "boolean", "eos flag");
                assert(output.attention.length >= 1, "attention rows");
                const timestamped = generate({
                    model,
                    encoderOutput: new Float32Array(2 * model.config.dModel).fill(0.1),
                    encoderSeq: 2,
                    prompt: [3, 4, 7],
                    timestamps: true,
                    maxTokens: 2,
                });
                assert(timestamped.ids.length <= 2, "timestamp cap");
            },
        },
        {
            name: "whisper/dtw computeDtw",
            run: () => {
                const attention: Float32Array[][] = [
                    [new Float32Array([0.1, 0.2, 0.3, 0.4])],
                    [new Float32Array([0.4, 0.3, 0.2, 0.1])],
                    [new Float32Array([0.2, 0.2, 0.2, 0.2])],
                ];
                const result = computeDtw({ attention, tokenCount: 3, frameCount: 4, heads: 1 });
                assertEqual(result.times.length, 3, "times");
                assertEqual(result.textIndices.length, result.timeIndices.length, "index pair");
                const empty = computeDtw({
                    attention: [],
                    tokenCount: 0,
                    frameCount: 4,
                    heads: 1,
                });
                assertEqual(empty.times.length, 0, "empty times");
            },
        },
        {
            name: "store/download quality selection",
            run: async () => {
                const
                    high = resolveSpec({
                        model: ModelId.WhisperBase,
                        quality: Quality.High,
                    }),
                    low = resolveSpec({
                        model: ModelId.WhisperBase,
                        quality: Quality.Low,
                    }),
                    texts: Record<string, string> = {},
                    bytes: Record<string, Uint8Array> = {};
                for (const file of low.files) {
                    const key = checkpointKey({
                        checkpoint: low.checkpoint,
                        file,
                    });
                    if (file === low.weightFile) {
                        bytes[key] = new Uint8Array([0]);
                    } else {
                        texts[key] = "{}";
                    }
                }
                const store = fakeStore({ texts, bytes });
                assertEqual(
                    await isCheckpointCached({ store, spec: low }),
                    true,
                    "low cached",
                );
                assertEqual(
                    await isCheckpointCached({ store, spec: high }),
                    false,
                    "high missing",
                );
                const prepared = await prepareModel({
                    store,
                    model: ModelId.WhisperBase,
                    quality: Quality.Low,
                });
                assertEqual(prepared.quality, Quality.Low, "prepared quality");
                assertEqual(
                    prepared.weightFile,
                    "whisper-base-Q4_K_M.gguf",
                    "prepared weight",
                );
            },
        },
        {
            name: "whisper/model loadWhisper",
            run: async () => {
                const
                    config = whisperRaw(),
                    bytes = safetensorsBytes({
                        tensors: [{
                            name: "model.decoder.embed_tokens.weight",
                            dtype: "F32",
                            shape: [config.vocab_size, config.d_model],
                            data: f32Data(
                                new Array<number>(
                                    config.vocab_size * config.d_model,
                                ).fill(0.5),
                            ),
                        }],
                    }),
                    store = fakeStore({
                        texts: {
                            [checkpointKey({
                                checkpoint: ModelId.WhisperBase,
                                file: "config.json",
                            })]: JSON.stringify(config),
                            [checkpointKey({
                                checkpoint: ModelId.WhisperBase,
                                file: "generation_config.json",
                            })]: JSON.stringify(whisperGeneration()),
                            [checkpointKey({
                                checkpoint: ModelId.WhisperBase,
                                file: "preprocessor_config.json",
                            })]: JSON.stringify(whisperPreprocessor()),
                            [checkpointKey({
                                checkpoint: ModelId.WhisperBase,
                                file: "tokenizer.json",
                            })]: JSON.stringify({ model: { vocab: { "a": 0, "b": 1 } } }),
                        },
                        bytes: {
                            [checkpointKey({
                                checkpoint: ModelId.WhisperBase,
                                file: "model.safetensors",
                            })]: bytes,
                        },
                    }),
                    loaded = await loadWhisper({
                        store,
                        model: ModelId.WhisperBase,
                        checkpoint: ModelId.WhisperBase,
                        spec: resolveSpec({ model: ModelId.WhisperBase }),
                    });
                assertEqual(loaded.kind, ModelFamily.Whisper, "kind");
                assertEqual(loaded.config.dModel, 4, "dModel");
                assertEqual(loaded.mel.featureSize, 2, "mel");
                assertEqual(loaded.constants.isMultilingual, true, "constants");
                assertEqual(loaded.tokenizer.vocabSize, 2, "tokenizer");
                assert(
                    loaded.tensors.has("model.decoder.embed_tokens.weight"),
                    "tensors",
                );
            },
        },
        {
            name: "whisper/model loadWhisper English",
            run: async () => {
                const
                    config = whisperRaw(),
                    checkpoint = MODEL_SPECS["whisper-base.en"].checkpoint,
                    bytes = safetensorsBytes({
                        tensors: [{
                            name: "model.decoder.embed_tokens.weight",
                            dtype: "F32",
                            shape: [config.vocab_size, config.d_model],
                            data: f32Data(
                                new Array<number>(
                                    config.vocab_size * config.d_model,
                                ).fill(0.5),
                            ),
                        }],
                    }),
                    store = fakeStore({
                        texts: {
                            [checkpointKey({
                                checkpoint,
                                file: "config.json",
                            })]: JSON.stringify(config),
                            [checkpointKey({
                                checkpoint,
                                file: "generation_config.json",
                            })]: JSON.stringify(whisperGeneration({
                                is_multilingual: false,
                                lang_to_id: {},
                                task_to_id: {},
                            })),
                            [checkpointKey({
                                checkpoint,
                                file: "preprocessor_config.json",
                            })]: JSON.stringify(whisperPreprocessor()),
                            [checkpointKey({
                                checkpoint,
                                file: "tokenizer.json",
                            })]: JSON.stringify({ model: { vocab: { "a": 0 } } }),
                        },
                        bytes: {
                            [checkpointKey({
                                checkpoint,
                                file: "model.safetensors",
                            })]: bytes,
                        },
                    }),
                    loaded = await loadWhisper({
                        store,
                        model: ModelId.WhisperBase,
                        checkpoint,
                        spec: resolveSpec({
                            model: ModelId.WhisperBase,
                            language: Language.en,
                        }),
                    });
                assertEqual(loaded.name, ModelId.WhisperBase, "public name");
                assertEqual(loaded.constants.isMultilingual, false, "english");
                assertEqual(loaded.tokenizer.vocabSize, 1, "tokenizer");
            },
        },
        {
            name: "whisper/model loadWhisper GGUF",
            run: async () => {
                const
                    config = whisperRaw(),
                    bytes = ggufBytes({
                        tensors: [{
                            name: "dec.token_embd.weight",
                            dims: [config.d_model, config.vocab_size],
                            type: 0,
                            data: f32Data(
                                new Array<number>(
                                    config.vocab_size * config.d_model,
                                ).fill(0.5),
                            ),
                        }],
                    }),
                    checkpoint = ModelId.WhisperBase,
                    spec = MODEL_SPECS[checkpoint],
                    store = fakeStore({
                        texts: {
                            [checkpointKey({
                                checkpoint,
                                file: "config.json",
                            })]: JSON.stringify(config),
                            [checkpointKey({
                                checkpoint,
                                file: "generation_config.json",
                            })]: JSON.stringify(whisperGeneration()),
                            [checkpointKey({
                                checkpoint,
                                file: "preprocessor_config.json",
                            })]: JSON.stringify(whisperPreprocessor()),
                            [checkpointKey({
                                checkpoint,
                                file: "tokenizer.json",
                            })]: JSON.stringify({ model: { vocab: { "a": 0 } } }),
                        },
                        bytes: {
                            [checkpointKey({
                                checkpoint,
                                file: spec.weightFile,
                            })]: bytes,
                        },
                    }),
                    loaded = await loadWhisper({
                        store,
                        model: ModelId.WhisperBase,
                        checkpoint,
                        spec,
                    });
                assertEqual(loaded.tensors.info(
                    "model.decoder.embed_tokens.weight",
                ).dtype, "F32", "gguf dtype");
                assert(
                    loaded.tensors.get(
                        "model.decoder.embed_tokens.weight",
                    ).length > 0,
                    "gguf weights",
                );
            },
        },
        {
            name: "whisper/weights GGUF mapping and dequantization",
            run: async () => {
                const
                    config = whisperRaw(),
                    bytes = ggufBytes({
                        tensors: [{
                            name: "dec.token_embd.weight",
                            dims: [config.d_model, config.vocab_size],
                            type: 0,
                            data: f32Data(
                                new Array<number>(
                                    config.vocab_size * config.d_model,
                                ).fill(0.5),
                            ),
                        }],
                    }),
                    file = parseGguf({ bytes }),
                    weights = createWhisperWeights({ file }),
                    embed = "model.decoder.embed_tokens.weight";
                assertEqual(mapName("enc.conv.0.weight"), "model.encoder.conv1.weight", "map conv");
                assertEqual(
                    mapName("enc.blocks.0.attn.q.weight"),
                    "model.encoder.layers.0.self_attn.q_proj.weight",
                    "map encoder attn",
                );
                assertEqual(
                    mapName("dec.blocks.0.cross_attn.out.bias"),
                    "model.decoder.layers.0.encoder_attn.out_proj.bias",
                    "map decoder cross",
                );
                assertEqual(mapName("frontend.window"), null, "skip frontend");
                assertEqual(weights.has(embed), true, "has embed");
                assertEqual(weights.info(embed).shape[0], config.vocab_size, "shape");
                assertList(
                    weights.get(embed).subarray(0, 2),
                    [0.5, 0.5],
                    "embed values",
                );
                assertEqual(ggmlTypeName(0), "F32", "f32 name");
                assertEqual(ggmlTypeName(8), "Q8_0", "q8 name");
                assertEqual(ggmlTypeName(12), "Q4_K", "q4 name");
                assertEqual(ggmlTypeName(14), "Q6_K", "q6 name");
                assertEqual(ggmlTypeName(2), "ggml:2", "other name");
                assertEqual(GGUF_CACHE_BYTES, 128 * 1024 * 1024, "cache budget");
                const
                    narrow = parseGguf({
                        bytes: ggufBytes({
                            tensors: [{
                                name: "bad",
                                dims: [128],
                                type: 12,
                                data: new Uint8Array(72),
                            }],
                        }),
                    }),
                    other = parseGguf({
                        bytes: ggufBytes({
                            tensors: [{
                                name: "other",
                                dims: [32],
                                type: 2,
                                data: new Uint8Array(18),
                            }],
                        }),
                    });
                assertThrows(
                    () => readGgufTensor({
                        file: narrow,
                        info: narrow.tensors.get("bad")!,
                    }),
                    "q4 row width",
                );
                assertThrows(
                    () => readGgufTensor({
                        file: other,
                        info: other.tensors.get("other")!,
                    }),
                    "unsupported ggml type",
                );
                const
                    source = file.tensors.get("dec.token_embd.weight"),
                    f32 = source
                        ? readGgufTensor({ file, info: source })
                        : new Float32Array(0);
                assertEqual(f32.length, config.vocab_size * config.d_model, "read tensor");
            },
        },
        {
            name: "whisper/model transcribeWhisper",
            run: () => {
                const model = graphModel();
                const transcript = transcribeWhisper({
                    model,
                    pcm: new Float32Array(64).fill(0.1),
                    language: Language.en,
                    timestamps: TimestampMode.none,
                    lowLatency: true,
                });
                assertEqual(typeof transcript.text, "string", "text");
                assertEqual(transcript.segments.length, 1, "segments");
                assertEqual(transcript.language, "en", "language");
                assertClose(transcript.duration, 64 / 16000, "duration");
                assertEqual(transcript.meta.model, ModelId.WhisperBase, "meta");
                assertEqual(transcript.meta.threads, 1, "threads");
            },
        },
        {
            name: "sensevoice/config readSenseVoiceConfig",
            run: () => {
                const file = ggufFileLiteral([
                    ["stt.sensevoice.encoder.d_model", 4],
                    ["stt.sensevoice.encoder.n_heads", 1],
                    ["stt.sensevoice.encoder.d_input", 2],
                    ["stt.sensevoice.encoder.d_ff", 4],
                    ["stt.sensevoice.encoder.kernel_size", 1],
                    ["stt.sensevoice.encoder.n_blocks", 2],
                    ["stt.sensevoice.encoder.tp_blocks", 1],
                    ["stt.sensevoice.special.lang_auto", 0],
                    ["stt.sensevoice.special.lang_zh", 1],
                    ["stt.sensevoice.special.lang_en", 2],
                    ["stt.sensevoice.special.lang_yue", 3],
                    ["stt.sensevoice.special.lang_ja", 4],
                    ["stt.sensevoice.special.lang_ko", 5],
                    ["stt.sensevoice.special.lang_nospeech", 6],
                    ["stt.sensevoice.special.event_speech", 7],
                    ["stt.sensevoice.special.emotion_neutral", 8],
                    ["stt.sensevoice.special.withitn", 9],
                    ["stt.sensevoice.special.woitn", 10],
                    ["tokenizer.ggml.tokens", [...senseVoiceTokens]],
                ]);
                const config = readSenseVoiceConfig({ file });
                assertEqual(config.dModel, 4, "dModel");
                assertEqual(config.headDim, 4, "headDim");
                assertEqual(config.eps, 1e-12, "eps");
                assertEqual(config.vocabSize, senseVoiceTokens.length, "vocab");
                assertEqual(config.languageIds.get("en"), 2, "language");
                assertEqual(config.eventEmotion[0], 7, "event");
                assertEqual(config.withItn, 9, "itn");
                assertThrows(
                    () => readSenseVoiceConfig({ file: ggufFileLiteral([]) }),
                    "missing keys",
                );
            },
        },
        {
            name: "sensevoice/config readSenseVoiceFrontend",
            run: () => {
                const
                    config = senseVoiceConfig(),
                    weights = senseVoiceWeights(senseVoiceShapes(config), {
                        "frontend.cmvn.shift": [1, 2],
                        "frontend.cmvn.scale": [0.5, 2],
                    }),
                    file = ggufFileLiteral([
                        ["stt.frontend.sample_rate", 16000],
                        ["stt.frontend.num_mels", 2],
                        ["stt.frontend.n_fft", 4],
                        ["stt.frontend.win_length", 4],
                        ["stt.frontend.hop_length", 2],
                        ["stt.frontend.lfr_m", 1],
                        ["stt.frontend.lfr_n", 1],
                        ["stt.frontend.upscale_samples", true],
                    ]),
                    frontend = readSenseVoiceFrontend({ file, weights });
                assertEqual(frontend.samplingRate, 16000, "rate");
                assertEqual(frontend.numMels, 2, "mels");
                assertEqual(frontend.lfrM, 1, "lfr m");
                assertEqual(frontend.upscale, 32768, "upscale");
                assertList(frontend.shift, [1, 2], "shift");
                assertList(frontend.scale, [0.5, 2], "scale");
            },
        },
        {
            name: "sensevoice/ctc decodeCtc",
            run: () => {
                const single = decodeCtc({
                    logits: new Float32Array([1, 5, 2]),
                    frames: 1,
                    vocabSize: 3,
                    blankId: 0,
                });
                assertList(single.ids, [1], "single id");
                assertList(single.frames, [0], "single frame");
                const blank = decodeCtc({
                    logits: new Float32Array([5, 1, 2]),
                    frames: 1,
                    vocabSize: 3,
                    blankId: 0,
                });
                assertList(blank.ids, [], "blank id");
                const multi = decodeCtc({
                    logits: new Float32Array([5, 1, 2, 0, 5, 1, 0, 5, 1, 0, 1, 5]),
                    frames: 4,
                    vocabSize: 3,
                    blankId: 0,
                });
                assertEqual(multi.ids.length, multi.frames.length, "shape");
                assertList(multi.ids, [1, 2], "multi ids");
                assertList(multi.frames, [1, 3], "multi frames");
            },
        },
        {
            name: "sensevoice/tokenizer createSenseVoiceVocabulary",
            run: () => {
                const file = ggufFileLiteral([
                    ["tokenizer.ggml.tokens", [...senseVoiceTokens]],
                    ["tokenizer.ggml.token_type", [...senseVoiceTypes]],
                ]);
                const vocabulary = createSenseVoiceVocabulary({ file });
                assertEqual(vocabulary.tokenizer.tokenText(0), "", "control text");
                assertEqual(vocabulary.tokenizer.tokenText(3), " hello", "metaspace");
                assertEqual(
                    vocabulary.tokenizer.decode([3, 4]),
                    "hello world",
                    "decode",
                );
                assertEqual(vocabulary.isControl(0), true, "is control");
                assertEqual(vocabulary.isControl(3), false, "not control");
                assertEqual(vocabulary.languageOf(2), "en", "language of");
                assertEqual(vocabulary.languageOf(3), null, "no language");
                assertEqual(vocabulary.tokenizer.vocabSize, senseVoiceTokens.length, "vocab");
                assertThrows(
                    () => createSenseVoiceVocabulary({ file: ggufFileLiteral([]) }),
                    "missing tokens",
                );
            },
        },
        {
            name: "sensevoice/fbank computeSenseVoiceFeatures",
            run: () => {
                const
                    config = senseVoiceConfig(),
                    frontend = {
                        samplingRate: 16000,
                        numMels: config.dInput,
                        nFft: 4,
                        winLength: 4,
                        hopLength: 2,
                        lfrM: 1,
                        lfrN: 1,
                        upscale: 1,
                        shift: new Float32Array(config.dInput),
                        scale: new Float32Array(config.dInput).fill(1),
                    },
                    result = computeSenseVoiceFeatures({
                        pcm: new Float32Array(24).fill(0.3),
                        frontend,
                    });
                assertEqual(result.frames, 11, "frames");
                assertEqual(result.values.length, result.frames * config.dInput, "values");
                const shaped = computeSenseVoiceFeatures({
                    pcm: new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
                    frontend,
                });
                assertEqual(shaped.frames, 2, "shaped frames");
                assertList(
                    shaped.values,
                    [-15.94238, -4.80129, -15.94238, -4.80129],
                    "shaped values",
                );
                const short = computeSenseVoiceFeatures({
                    pcm: new Float32Array(2),
                    frontend,
                });
                assertEqual(short.frames, 0, "short frames");
            },
        },
        {
            name: "sensevoice/weights createGgufWeights",
            run: () => {
                const q8 = new Uint8Array(34);
                q8[0] = 0x00;
                q8[1] = 0x3c;
                for (let i = 0; i < 32; i += 1) {
                    q8[2 + i] = i + 1;
                }
                const bytes = ggufBytes({
                    tensors: [
                        {
                            name: "f32t",
                            dims: [2, 2],
                            type: 0,
                            data: f32Data([1, 2, 3, 4]),
                        },
                        {
                            name: "f16t",
                            dims: [2],
                            type: 1,
                            data: f16Data([1, 0.5]),
                        },
                        { name: "q8t", dims: [32], type: 8, data: q8 },
                        {
                            name: "q4t",
                            dims: [256],
                            type: 12,
                            data: q4KBlock({
                                d: 1,
                                dmin: 0.5,
                                scales: [1, 2, 3, 4, 1, 2, 3, 4],
                                mins: [0, 2, 0, 0, 0, 0, 0, 0],
                                quants: 0x21,
                            }),
                        },
                        {
                            name: "q6t",
                            dims: [256],
                            type: 14,
                            data: q6KBlock({
                                d: 1,
                                scales: [
                                    1, 2, 3, 4, 5, 6, 7, 8,
                                    9, 10, 11, 12, 13, 14, 15, 16,
                                ],
                                ql: 0x21,
                                qh: 0xe4,
                            }),
                        },
                    ],
                });
                const weights = createGgufWeights({ file: parseGguf({ bytes }) });
                assertList(weights.f32("f32t"), [1, 2, 3, 4], "f32");
                assertList(weights.f32("f16t"), [1, 0.5], "f16");
                const dequant = weights.f32("q8t");
                assertEqual(dequant.length, 32, "q8 length");
                assertClose(dequant[0], 1, "q8 first");
                assertClose(dequant[31], 32, "q8 last");
                const q4 = weights.f32("q4t");
                assertEqual(q4.length, 256, "q4 length");
                assertClose(q4[0], 1, "q4 first");
                assertClose(q4[32], 3, "q4 second");
                assertClose(q4[64], 3, "q4 third");
                assertClose(q4[96], 8, "q4 fourth");
                assertClose(q4[160], 4, "q4 fifth");
                assertClose(q4[224], 8, "q4 last");
                const q6 = weights.f32("q6t");
                assertEqual(q6.length, 256, "q6 length");
                assertClose(q6[0], -31, "q6 first");
                assertClose(q6[32], -45, "q6 second");
                assertClose(q6[64], 10, "q6 third");
                assertClose(q6[96], 126, "q6 fourth");
                assertClose(q6[128], -279, "q6 fifth");
                assertClose(q6[224], 270, "q6 last");
                assertEqual(weights.shape("f32t")[0], 2, "shape");
                assert(weights.names().includes("f32t"), "names");
                assertThrows(() => weights.f32("missing"), "missing");
                const uncached = createGgufWeights({
                    file: parseGguf({ bytes }),
                    cacheBytes: 0,
                });
                assertList(uncached.f32("f32t"), [1, 2, 3, 4], "no cache");
                const evicting = createGgufWeights({
                    file: parseGguf({ bytes }),
                    cacheBytes: 16,
                });
                assertClose(evicting.f32("q4t")[0], 1, "tiny cache first");
                assertList(evicting.f32("f32t"), [1, 2, 3, 4], "tiny cache f32");
                assertClose(evicting.f32("q4t")[0], 1, "tiny cache reload");
            },
        },
        {
            name: "sensevoice/encoder encodeSenseVoice and senseVoiceLogits",
            run: () => {
                const
                    config = senseVoiceConfig(),
                    weights = senseVoiceWeights(senseVoiceShapes(config)),
                    encoded = encodeSenseVoice({
                        config,
                        weights,
                        features: new Float32Array(2 * 2).fill(0.2),
                        frames: 2,
                        prefix: [0, 7, 8, 9],
                    });
                assertEqual(encoded.seq, 6, "seq");
                assertEqual(encoded.hidden.length, encoded.seq * config.dModel, "hidden");
                const logits = senseVoiceLogits({
                    config,
                    weights,
                    hidden: encoded.hidden,
                    seq: encoded.seq,
                });
                assertEqual(logits.length, encoded.seq * config.vocabSize, "logits");
            },
        },
        {
            name: "sensevoice/model loadSenseVoice",
            run: async () => {
                const
                    config = senseVoiceConfig(),
                    store = fakeStore({
                        bytes: {
                            [checkpointKey({
                                checkpoint: ModelId.SenseVoiceSmall,
                                file: MODEL_SPECS[ModelId.SenseVoiceSmall].weightFile,
                            })]: senseVoiceGguf(config),
                        },
                    }),
                    loaded: LoadedSenseVoice = await loadSenseVoice({
                        store,
                        model: ModelId.SenseVoiceSmall,
                        checkpoint: ModelId.SenseVoiceSmall,
                        spec: MODEL_SPECS[ModelId.SenseVoiceSmall],
                    });
                assertEqual(loaded.kind, ModelFamily.SenseVoice, "kind");
                assertEqual(loaded.config.dModel, 4, "dModel");
                assertEqual(loaded.frontend.numMels, 2, "frontend");
                assertEqual(loaded.vocabulary.isControl(0), true, "vocabulary");
                assert(loaded.weights.has("ctc.head.weight"), "weights");
            },
        },
        {
            name: "sensevoice/model transcribeSenseVoice",
            run: async () => {
                const
                    config = senseVoiceConfig(),
                    store = fakeStore({
                        bytes: {
                            [checkpointKey({
                                checkpoint: ModelId.SenseVoiceSmall,
                                file: MODEL_SPECS[ModelId.SenseVoiceSmall].weightFile,
                            })]: senseVoiceGguf(config),
                        },
                    }),
                    model = await loadSenseVoice({
                        store,
                        model: ModelId.SenseVoiceSmall,
                        checkpoint: ModelId.SenseVoiceSmall,
                        spec: MODEL_SPECS[ModelId.SenseVoiceSmall],
                    }),
                    transcript = transcribeSenseVoice({
                        model,
                        pcm: new Float32Array(24).fill(0.3),
                        language: Language.en,
                        itn: true,
                    });
                assertEqual(typeof transcript.text, "string", "text");
                assertEqual(transcript.segments.length, 1, "segments");
                assertEqual(transcript.language, "en", "language");
                assertClose(transcript.duration, 24 / 16000, "duration");
            },
        },
        {
            name: "store/chunk round trip",
            run: async () => {
                const
                    map = new Map<string, string>(),
                    read = async (key: string): Promise<string | undefined> => map.get(key),
                    payload = Uint8Array.from([1, 2, 3, 4, 5, 6, 7, 8]),
                    progress: number[] = [];
                assertEqual(
                    await chunkedExists({ key: "a", read }),
                    false,
                    "absent",
                );
                assertEqual(
                    await chunkedReady({ key: "a", read }),
                    false,
                    "not ready",
                );
                await writeChunkStream({
                    key: "a",
                    stream: streamFromBytes(payload),
                    encode: toBase64,
                    write: async (key, text) => {
                        map.set(key, text);
                    },
                    remove: async (key) => {
                        map.delete(key);
                    },
                    onProgress: (loaded) => progress.push(loaded),
                });
                assertEqual(await chunkedExists({ key: "a", read }), true, "present");
                assertEqual(await chunkedReady({ key: "a", read }), true, "ready");
                map.set("b", "raw");
                assertEqual(await chunkedExists({ key: "b", read }), true, "raw present");
                assertEqual(await chunkedReady({ key: "b", read }), false, "raw not ready");
                assertEqual(await chunkedSize({ key: "a", read }), 8, "chunk size");
                assertEqual(progress.length, 1, "progress");
                const bytes = await readChunks({ key: "a", read, decode: fromBase64 });
                assertList(bytes ?? new Uint8Array(0), [1, 2, 3, 4, 5, 6, 7, 8], "bytes");
                assertEqual(
                    await readChunks({ key: "missing", read, decode: fromBase64 }),
                    null,
                    "missing chunks",
                );
                await removeChunks({
                    key: "a",
                    read,
                    remove: async (key) => {
                        map.delete(key);
                    },
                });
                assertEqual(await chunkedExists({ key: "a", read }), false, "removed");
                assertEqual(await chunkedReady({ key: "a", read }), false, "not ready after");
                // orphan chunks of an interrupted download
                map.set("c.part0", "one");
                map.set("c.part1", "two");
                map.set("c.part2", "three");
                await removeChunks({
                    key: "c",
                    read,
                    remove: async (key) => {
                        map.delete(key);
                    },
                });
                assertEqual(map.has("c.part0"), false, "orphan0 removed");
                assertEqual(map.has("c.part1"), false, "orphan1 removed");
                assertEqual(map.has("c.part2"), false, "orphan2 removed");
            },
        },
        {
            name: "store/browser createBrowserStore",
            run: async () => {
                const store = createBrowserStore();
                for (const key of [
                    "exists",
                    "readText",
                    "readBytes",
                    "hasBinary",
                    "size",
                    "writeText",
                    "writeStream",
                    "remove",
                    "removeRaw",
                ] as const) {
                    assertEqual(typeof store[key], "function", `method ${key}`);
                }
                assert(store.parts !== undefined, "parts");
                await store.parts?.write({
                    key: "harness/parts",
                    index: 0,
                    bytes: Uint8Array.from([1, 2]),
                });
                await store.parts?.writeSizes({
                    key: "harness/parts",
                    sizes: [2],
                });
                assertList(
                    (await store.parts?.read({
                        key: "harness/parts",
                        index: 0,
                    })) ?? new Uint8Array(0),
                    [1, 2],
                    "part read",
                );
                assertList(
                    (await store.parts?.readSizes("harness/parts")) ?? [],
                    [2],
                    "part sizes",
                );
                assertList(
                    (await store.readBytes("harness/parts")) ??
                        new Uint8Array(0),
                    [1, 2],
                    "part file",
                );
                await store.remove("harness/parts");
                assertEqual(
                    await store.parts?.readSizes("harness/parts"),
                    null,
                    "part removed",
                );
                await store.writeText({ key: "harness/probe", text: "value" });
                assertEqual(await store.readText("harness/probe"), "value", "round trip");
                assertEqual(await store.size?.("harness/probe"), 5, "probe size");
                assertEqual(
                    await store.hasBinary?.("harness/probe"),
                    false,
                    "text not binary",
                );
                await store.writeStream({
                    key: "harness/bin",
                    stream: streamFromBytes(Uint8Array.from([9, 8, 7])),
                });
                assertEqual(
                    await store.hasBinary?.("harness/bin"),
                    true,
                    "binary ready",
                );
                assertList(
                    (await store.readBytes("harness/bin")) ?? new Uint8Array(0),
                    [9, 8, 7],
                    "binary read",
                );
                await store.remove("harness/bin");
                assertEqual(
                    await store.hasBinary?.("harness/bin"),
                    false,
                    "binary removed",
                );
                await store.remove("harness/probe");
                assertEqual(await store.exists("harness/probe"), false, "removed");
            },
        },
        {
            name: "store/download prepareModel and deleteModel",
            run: async () => {
                const
                    spec = MODEL_SPECS[ModelId.WhisperBase],
                    english = MODEL_SPECS["whisper-base.en"],
                    texts: Record<string, string> = {},
                    bytes: Record<string, Uint8Array> = {};
                for (const file of spec.files) {
                    const key = checkpointKey({ checkpoint: spec.checkpoint, file });
                    if (file === spec.weightFile) {
                        bytes[key] = new Uint8Array([0]);
                    } else {
                        texts[key] = "{}";
                    }
                }
                for (const file of english.files) {
                    const key = checkpointKey({
                        checkpoint: english.checkpoint,
                        file,
                    });
                    if (file === english.weightFile) {
                        bytes[key] = new Uint8Array([0]);
                    } else {
                        texts[key] = "{}";
                    }
                }
                const store = fakeStore({ texts, bytes });
                const prepared = await prepareModel({
                    store,
                    model: ModelId.WhisperBase,
                });
                assertEqual(prepared.id, ModelId.WhisperBase, "spec id");
                const englishPrepared = await prepareModel({
                    store,
                    model: ModelId.WhisperBase,
                    language: Language.en,
                });
                assertEqual(
                    englishPrepared.checkpoint,
                    "whisper-base.en",
                    "english checkpoint",
                );
                const
                    defaultSpec = MODEL_SPECS[DEFAULT_MODEL],
                    defaultTexts: Record<string, string> = {},
                    defaultBytes: Record<string, Uint8Array> = {};
                for (const file of defaultSpec.files) {
                    const key = checkpointKey({ checkpoint: defaultSpec.checkpoint, file });
                    if (file === defaultSpec.weightFile) {
                        defaultBytes[key] = new Uint8Array([0]);
                    } else {
                        defaultTexts[key] = "{}";
                    }
                }
                const fallback = await prepareModel({
                    store: fakeStore({ texts: defaultTexts, bytes: defaultBytes }),
                });
                assertEqual(fallback.id, DEFAULT_MODEL, "default spec");
                await deleteModel({ store, model: ModelId.WhisperBase });
                assertEqual(
                    await store.exists(checkpointKey({
                        checkpoint: spec.checkpoint,
                        file: spec.weightFile,
                    })),
                    false,
                    "deleted multilingual",
                );
                assertEqual(
                    await store.exists(checkpointKey({
                        checkpoint: english.checkpoint,
                        file: english.weightFile,
                    })),
                    false,
                    "deleted english",
                );
            },
        },
        {
            name: "store/download refreshes a stale weight",
            run: async () => {
                const
                    spec = MODEL_SPECS["whisper-base.en"],
                    weightKey = checkpointKey({
                        checkpoint: spec.checkpoint,
                        file: spec.weightFile,
                    }),
                    texts = new Map<string, string>(),
                    binaries = new Map<string, Uint8Array>();
                let stale = true;
                const store: ModelStore = {
                    exists: async (key) =>
                        texts.has(key) || binaries.has(key) ||
                        (key === weightKey && stale),
                    readText: async (key) => texts.get(key) ?? null,
                    readBytes: async (key) =>
                        key === weightKey && stale
                            ? null
                            : binaries.get(key) ?? null,
                    hasBinary: async (key) => key === weightKey && !stale,
                    writeText: async ({ key, text }) => {
                        texts.set(key, text);
                    },
                    writeStream: async ({ key, stream, onProgress }) => {
                        const reader = stream.getReader();
                        const parts: Uint8Array[] = [];
                        let total = 0;
                        for (;;) {
                            const result = await reader.read();
                            if (result.done) {
                                break;
                            }
                            parts.push(result.value);
                            total += result.value.length;
                            onProgress?.(result.value.length);
                        }
                        const merged = new Uint8Array(total);
                        let offset = 0;
                        for (const part of parts) {
                            merged.set(part, offset);
                            offset += part.length;
                        }
                        binaries.set(key, merged);
                        if (key === weightKey) {
                            stale = false;
                        }
                    },
                    remove: async (key) => {
                        texts.delete(key);
                        binaries.delete(key);
                        if (key === weightKey) {
                            stale = false;
                        }
                    },
                };
                assertEqual(await store.exists(weightKey), true, "stale present");
                assertEqual(
                    await store.hasBinary?.(weightKey),
                    false,
                    "stale unreadable",
                );
                const
                    original = globalThis.fetch,
                    savedBytes = spec.weights[Quality.High].bytes,
                    requested: string[] = [];
                spec.weights[Quality.High].bytes = 4;
                globalThis.fetch = (async (input: RequestInfo | URL) => {
                    const url = String(input);
                    requested.push(url);
                    const body = url.endsWith(spec.weightFile)
                        ? new Uint8Array([1, 2, 3, 4])
                        : "{}";
                    return new Response(body, { status: 200 });
                }) as typeof fetch;
                try {
                    await prepareModel({
                        store,
                        model: ModelId.WhisperBase,
                        language: Language.en,
                    });
                } finally {
                    globalThis.fetch = original;
                    spec.weights[Quality.High].bytes = savedBytes;
                }
                assert(requested.length > 0, "fetched");
                assertEqual(
                    await store.hasBinary?.(weightKey),
                    true,
                    "binary ready",
                );
                assertList(
                    (await store.readBytes(weightKey)) ?? new Uint8Array(0),
                    [1, 2, 3, 4],
                    "refreshed bytes",
                );
            },
        },
        {
            name: "store/download verifies size without onProgress",
            run: async () => {
                const
                    spec = MODEL_SPECS[ModelId.WhisperBase],
                    weightKey = checkpointKey({
                        checkpoint: spec.checkpoint,
                        file: spec.weightFile,
                    }),
                    texts = new Map<string, string>(),
                    binaries = new Map<string, Uint8Array>(),
                    store: ModelStore = {
                        exists: async (key) =>
                            texts.has(key) || binaries.has(key),
                        readText: async (key) => texts.get(key) ?? null,
                        readBytes: async (key) => binaries.get(key) ?? null,
                        hasBinary: async (key) => binaries.has(key),
                        writeText: async ({ key, text }) => {
                            texts.set(key, text);
                        },
                        writeStream: async ({ key, stream, onProgress }) => {
                            const reader = stream.getReader();
                            let total = 0;
                            for (; ;) {
                                const result = await reader.read();
                                if (result.done) {
                                    break;
                                }
                                total += result.value.length;
                                onProgress?.(result.value.length);
                            }
                            binaries.set(key, new Uint8Array(total));
                        },
                        remove: async (key) => {
                            texts.delete(key);
                            binaries.delete(key);
                        },
                    },
                    savedBytes = spec.weights[Quality.High].bytes,
                    original = globalThis.fetch;
                spec.weights[Quality.High].bytes = 4;
                globalThis.fetch = (async (input: RequestInfo | URL) => {
                    const url = String(input);
                    const weight = url.endsWith(spec.weightFile);
                    return new Response(
                        weight ? new Uint8Array([1, 2, 3, 4]) : "{}",
                        {
                            status: 200,
                            headers: {
                                "content-length": weight ? "4" : "2",
                            },
                        },
                    );
                }) as typeof fetch;
                try {
                    await prepareModel({ store, model: ModelId.WhisperBase });
                } finally {
                    globalThis.fetch = original;
                    spec.weights[Quality.High].bytes = savedBytes;
                }
                assertEqual(
                    await store.hasBinary?.(weightKey),
                    true,
                    "binary ready",
                );
                assertEqual(
                    (await store.readBytes(weightKey))?.length,
                    4,
                    "verified size",
                );
            },
        },
        {
            name: "store/download resumes ranges with aggregate progress",
            run: async () => {
                const
                    spec = MODEL_SPECS[ModelId.SenseVoiceSmall],
                    savedBytes = spec.weights[Quality.High].bytes,
                    total = CHUNK_BYTES * 2,
                    key = checkpointKey({
                        checkpoint: spec.checkpoint,
                        file: spec.weightFile,
                    }),
                    manifests = new Map<string, number[]>(),
                    chunks = new Map<string, Uint8Array>(),
                    chunkKey = (input: { key: string; index: number }) =>
                        `${input.key}:${input.index}`,
                    parts: ModelPartStore = {
                        readSizes: async (inner) =>
                            manifests.get(inner) ?? null,
                        writeSizes: async ({ key: inner, sizes }) => {
                            manifests.set(inner, sizes.slice());
                        },
                        read: async (input) =>
                            chunks.get(chunkKey(input)) ?? null,
                        write: async (input) => {
                            chunks.set(chunkKey(input), input.bytes.slice());
                        },
                        remove: async (input) => {
                            chunks.delete(chunkKey(input));
                        },
                    },
                    store: ModelStore = {
                        exists: async (inner) =>
                            manifests.has(inner) ||
                            chunks.has(`${inner}:0`),
                        readText: async () => null,
                        readBytes: async () => null,
                        hasBinary: async (inner) => {
                            const sizes = manifests.get(inner);
                            return sizes !== undefined &&
                                completeSizes(sizes);
                        },
                        size: async (inner) => {
                            const sizes = manifests.get(inner) ?? [];
                            let size = 0;
                            for (const item of sizes) {
                                size += item;
                            }
                            return size;
                        },
                        writeText: async () => undefined,
                        writeStream: async () => {
                            throw new Error("stream fallback used");
                        },
                        remove: async (inner) => {
                            manifests.delete(inner); // @ts-ignore
                            for (const chunk of [...chunks.keys()]) {
                                if (chunk.startsWith(`${inner}:`)) {
                                    chunks.delete(chunk);
                                }
                            }
                        },
                        parts,
                    },
                    urls: string[] = [],
                    ranges: string[] = [],
                    reports: DownloadProgress[] = [],
                    original = globalThis.fetch;
                await parts.write({
                    key,
                    index: 0,
                    bytes: new Uint8Array(CHUNK_BYTES),
                });
                await parts.writeSizes({
                    key,
                    sizes: [CHUNK_BYTES, 0],
                });
                spec.weights[Quality.High].bytes = total;
                let failOnce = true;
                globalThis.fetch = (async (
                    input: RequestInfo | URL,
                    init?: RequestInit,
                ) => {
                    urls.push(String(input));
                    const headers = (init?.headers ?? {}) as Record<
                        string,
                        string
                    >;
                    const range = headers.Range ?? "";
                    ranges.push(range);
                    const match = /^bytes=(\d+)-(\d+)$/.exec(range);
                    if (!match) {
                        throw new Error(`no range ${range}`);
                    }
                    const
                        start = Number(match[1]),
                        end = Number(match[2]),
                        length = end - start + 1;
                    if (start === CHUNK_BYTES && failOnce) {
                        failOnce = false;
                        return new Response(new Uint8Array(2), {
                            status: 206,
                        });
                    }
                    return new Response(new Uint8Array(length), {
                        status: 206,
                        headers: {
                            "content-range":
                                `bytes ${start}-${end}/${total}`,
                        },
                    });
                }) as typeof fetch;
                try {
                    assertEqual(
                        await isCheckpointCached({ store, spec }),
                        false,
                        "partial not cached",
                    );
                    await prepareModel({
                        store,
                        model: ModelId.SenseVoiceSmall,
                        weightsBaseUrl: "https://mirror.example/",
                        onProgress: (progress) => reports.push(progress),
                    });
                } finally {
                    globalThis.fetch = original;
                    spec.weights[Quality.High].bytes = savedBytes;
                }
                assertEqual(
                    await isCheckpointCached({ store, spec }),
                    true,
                    "complete cached",
                );
                assert(
                    ranges.includes(`bytes=${CHUNK_BYTES}-${total - 1}`),
                    "missing chunk fetched",
                );
                assert(
                    !ranges.includes(`bytes=0-${CHUNK_BYTES - 1}`),
                    "complete chunk kept",
                );
                const sizes = manifests.get(key) ?? [];
                assertList(sizes, [CHUNK_BYTES, CHUNK_BYTES], "sizes");
                assertEqual(chunks.get(`${key}:1`)?.length, CHUNK_BYTES, "chunk");
                for (const url of urls) {
                    assert(
                        url.startsWith("https://mirror.example/"),
                        `mirror ${url}`,
                    );
                }
                assert(reports.length > 0, "progress missing");
                let previous = -1;
                for (const report of reports) {
                    assert(report.loaded >= previous, "progress regressed");
                    previous = report.loaded;
                }
                const last = reports[reports.length - 1];
                assertEqual(last.phase, ProgressPhase.verify, "final phase");
                assertClose(last.ratio, 1, "final ratio");
                assertEqual(last.total, total, "final total");
            },
        },
        {
            name: "voice createVoice",
            run: async () => {
                const
                    config = senseVoiceConfig(),
                    store = fakeStore({
                        bytes: {
                            [checkpointKey({
                                checkpoint: ModelId.SenseVoiceSmall,
                                file: MODEL_SPECS[ModelId.SenseVoiceSmall].weightFile,
                            })]: senseVoiceGguf(config),
                        },
                    }),
                    voice = createVoice({ store }),
                    prepared = await voice.prepareModel({
                        model: ModelId.SenseVoiceSmall,
                    }),
                    loaded = await voice.loadModel({
                        model: ModelId.SenseVoiceSmall,
                    }),
                    transcript = voice.transcribe({
                        model: loaded,
                        pcm: new Float32Array(24).fill(0.3),
                    });
                assertEqual(prepared.id, ModelId.SenseVoiceSmall, "prepared");
                assertEqual(transcript.meta.model, ModelId.SenseVoiceSmall, "meta");
                assertEqual(
                    typeof voice.ensureModel,
                    "function",
                    "ensureModel",
                );
                assertEqual(
                    typeof getVoice().ensureModel,
                    "function",
                    "shared voice",
                );
                const whisper = voice.transcribe({
                    model: graphModel(),
                    pcm: new Float32Array(64).fill(0.1),
                    language: Language.en,
                    lowLatency: true,
                });
                assertEqual(whisper.meta.model, ModelId.WhisperBase, "whisper branch");
                await voice.deleteModel({ model: ModelId.SenseVoiceSmall });
                assertEqual(
                    await store.exists(checkpointKey({
                        checkpoint: ModelId.SenseVoiceSmall,
                        file: MODEL_SPECS[ModelId.SenseVoiceSmall].weightFile,
                    })),
                    false,
                    "deleted",
                );
            },
        },
        {
            name: "speech family bundles limit models",
            run: async () => {
                const
                    config = senseVoiceConfig(),
                    store = fakeStore({
                        bytes: {
                            [checkpointKey({
                                checkpoint: ModelId.SenseVoiceSmall,
                                file: MODEL_SPECS[ModelId.SenseVoiceSmall].weightFile,
                            })]: senseVoiceGguf(config),
                        },
                    }),
                    options = {
                        model: ModelId.SenseVoiceSmall,
                        weightsBaseUrl: "https://mirror.example",
                        weightCacheMB: 1,
                        threads: 1,
                    },
                    voice = createSenseVoiceVoice({ store }),
                    cached = await voice.isModelCached({ model: options.model });
                assertEqual(cached, true, "family cached");
                const loaded = await voice.loadModel(options);
                assertEqual(loaded.kind, ModelFamily.SenseVoice, "family kind");
                assertEqual(kernelThreads(), 1, "single thread");
                setThreadCount(1);
                const transcript = voice.transcribe({
                    model: loaded,
                    pcm: new Float32Array(24).fill(0.3),
                });
                assertEqual(transcript.meta.threads, 1, "meta threads");
                await assertRejects(
                    async () => await createWhisperVoice({ store }).loadModel({
                        model: ModelId.SenseVoiceSmall,
                    }),
                    "missing family",
                );
            },
        },
        {
            name: "tensor/pool browser pool",
            run: async () => {
                const
                    shared = createBrowserSharedMemory(),
                    memory = shared ?? new WebAssembly.Memory({ initial: 1 });
                assertEqual(
                    createBrowserPool({ memory, threads: 2 }),
                    null,
                    "sync pool unavailable",
                );
                if (shared === null) {
                    assertEqual(
                        await createBrowserPoolAsync({ memory, threads: 2 }),
                        null,
                        "no isolation",
                    );
                } else {
                    assertEqual(
                        await createBrowserPoolAsync({ memory, threads: 2 }),
                        null,
                        "main thread cannot block",
                    );
                    const result = await new Promise<PoolWorkerResult>(
                        (resolve, reject) => {
                            const worker = new Worker("poolWorker.js");
                            worker.onmessage = (event) => {
                                worker.terminate();
                                resolve(event.data as PoolWorkerResult);
                            };
                            worker.onerror = (event) => {
                                worker.terminate();
                                reject(new Error(event.message));
                            };
                            worker.postMessage(null);
                        },
                    );
                    assertEqual(result.ok, true, result.error ?? "worker");
                    assertEqual(
                        result.threads,
                        2,
                        `pool threads shared=${result.shared}`,
                    );
                    assertClose(result.value ?? 0, 1, "pool matmul");
                }
                assertEqual(
                    createStubSharedMemory(),
                    null,
                    "stub shared memory",
                );
                assertEqual(
                    createStubPool({
                        memory: new WebAssembly.Memory({ initial: 1 }),
                    }),
                    null,
                    "stub pool",
                );
            },
        },
        {
            name: "speech/env detect",
            run: () => {
                assertEqual(detectPlatform(), Platform.Browser, "platform");
                assertEqual(
                    typeof hasMicrophone(),
                    "boolean",
                    "microphone guard",
                );
            },
        },
        {
            name: "audio/resample downmix and resample",
            run: () => {
                assertList(
                    downmix({ input: new Float32Array([1, 3, 5, 7]), channels: 2 }),
                    [2, 6],
                    "downmix",
                );
                assertList(
                    resampleAudio({
                        input: new Float32Array([0.5]),
                        inputRate: 16000,
                        outputRate: 16000,
                    }),
                    [0.5],
                    "same rate",
                );
                assertEqual(
                    resampleAudio({
                        input: new Float32Array([0, 1, 0, -1]),
                        inputRate: 16000,
                        outputRate: 8000,
                    }).length,
                    2,
                    "half rate",
                );
            },
        },
        {
            name: "audio/wav decodeWav",
            run: () => {
                const
                    samples = new Float32Array(16000);
                for (let i = 0; i < samples.length; i += 1) {
                    samples[i] = Math.sin((2 * Math.PI * 300 * i) / 16000) * 0.5;
                }
                const decoded = decodeWav({
                    bytes: writeWav16({ samples, sampleRate: 16000 }),
                });
                assertEqual(decoded.samplingRate, 16000, "rate");
                assertEqual(decoded.pcm.length, 16000, "length");
                assertClose(decoded.pcm[100], samples[100], "sample");
            },
        },
        {
            name: "audio/flac decodeFlac",
            run: () => {
                const decoded = decodeFlac({ bytes: fromBase64(TINY_FLAC) });
                assertEqual(decoded.samplingRate, 16000, "rate");
                assert(decoded.pcm.length >= 3900, "length");
                let peak = 0; // @ts-ignore
                for (const value of decoded.pcm)
                    peak = Math.max(peak, Math.abs(value));
                assert(peak > 0.1, "peak");
            },
        },
        {
            name: "audio/mp3 decodeMp3",
            run: async () => {
                const decoded = await decodeMp3({ bytes: fromBase64(TINY_MP3) });
                assertEqual(decoded.samplingRate, 16000, "rate");
                assert(decoded.pcm.length > 3500, "length");
            },
        },
        {
            name: "audio/file decodeBytes and decodeSource",
            run: async () => {
                const wav = writeWav16({
                    samples: new Float32Array(16000),
                    sampleRate: 16000,
                });
                const viaBytes = await decodeBytes({
                    bytes: wav,
                    platform: Platform.Browser,
                });
                assertEqual(viaBytes.samplingRate, 16000, "bytes rate");
                assertEqual(viaBytes.pcm.length, 16000, "bytes length");
                const viaBlob = await decodeSource({
                    audioFilePath: new Blob([new Uint8Array(wav)]),
                    platform: Platform.Browser,
                });
                assertEqual(viaBlob.pcm.length, 16000, "blob length");
            },
        },
        {
            name: "audio/mic captureMicrophone reports errors",
            run: async () => {
                const
                    original = navigator.mediaDevices,
                    denied = new Error("denied");
                let captured: Error | null = null;
                Object.defineProperty(navigator, "mediaDevices", {
                    configurable: true,
                    value: {
                        getUserMedia: async (): Promise<never> => {
                            throw denied;
                        },
                    },
                });
                const capture = captureMicrophone({
                    onChunk: () => undefined,
                    onError: (error) => {
                        captured = error;
                    },
                });
                await new Promise((resolve) => setTimeout(resolve, 0));
                await capture.ready.catch(() => undefined);
                await capture.stop();
                Object.defineProperty(navigator, "mediaDevices", {
                    configurable: true,
                    value: original,
                });
                assert(captured !== null, "onError called");
                assertEqual((captured as any as Error).message, "denied", "message");
            },
        },
        {
            name: "audio/mic captures the fake device",
            run: async () => {
                let
                    chunks = 0,
                    samples = 0,
                    captured: Error | null = null;
                const capture = captureMicrophone({
                    onChunk: (data) => {
                        chunks += 1;
                        samples += data.length;
                    },
                    onError: (error) => {
                        captured = error;
                    },
                });
                void capture.ready.catch(() => undefined);
                const deadline = Date.now() + 10000;
                while (
                    samples <= 1000 && captured === null && Date.now() < deadline
                ) {
                    await new Promise((resolve) => setTimeout(resolve, 100));
                }
                await capture.stop();
                const failure = captured as Error | null;
                assertEqual(failure, null, `mic error ${failure?.message ?? ""}`);
                assert(chunks > 0, "no microphone chunks");
                assert(samples > 1000, "too few microphone samples");
            },
        },
        {
            name: "session streams the fake microphone",
            run: async () => {
                const
                    model = graphModel(),
                    fake = {
                        ensureModel: async (): Promise<Model> => model,
                        transcribe: (): Transcript => ({
                            text: "mic live",
                            language: Language.en,
                            duration: 1,
                            segments: [{
                                start: 0,
                                end: 1,
                                text: "mic live",
                                words: [],
                            }],
                            meta: { model: "sensevoice-small", rtf: 0, threads: 1 },
                        }),
                    } as unknown as Voice,
                    session = createSession({
                        options: { updateIntervalMs: 200 },
                        voice: fake,
                    });
                const deadline = Date.now() + 10000;
                while (
                    session.fullText().text.length === 0 &&
                    session.error === null &&
                    Date.now() < deadline
                ) {
                    await new Promise((resolve) => setTimeout(resolve, 100));
                }
                const transcript = await session.stop();
                assertEqual(session.error, null, "session mic error");
                assert(transcript.text.length > 0, "empty mic transcript");
                assert(transcript.duration > 0.2, "short mic duration");
            },
        },
        {
            name: "session accepts pushed audio",
            run: async () => {
                const
                    model = graphModel(),
                    fake = {
                        ensureModel: async (): Promise<Model> => model,
                        transcribe: (): Transcript => ({
                            text: "push live",
                            language: Language.en,
                            duration: 1,
                            segments: [{
                                start: 0,
                                end: 1,
                                text: "push live",
                                words: [],
                            }],
                            meta: { model: "sensevoice-small", rtf: 0, threads: 1 },
                        }),
                    } as unknown as Voice,
                    session = createSession({
                        options: {
                            microphone: false,
                            updateIntervalMs: 100,
                        },
                        voice: fake,
                    });
                session.push(new Float32Array(16000).fill(0.2));
                const deadline = Date.now() + 10000;
                while (
                    session.fullText().text.length === 0 &&
                    session.error === null &&
                    Date.now() < deadline
                ) {
                    await new Promise((resolve) => setTimeout(resolve, 50));
                }
                const transcript = await session.stop();
                assertEqual(transcript.text, "push live", "push text");
                assertEqual(session.error, null, "push error");
            },
        },
        {
            name: "session cancels pushed audio",
            run: async () => {
                const
                    model = graphModel(),
                    fake = {
                        ensureModel: async (): Promise<Model> => model,
                        transcribe: (): Transcript => ({
                            text: "push live",
                            language: Language.en,
                            duration: 1,
                            segments: [{
                                start: 0,
                                end: 1,
                                text: "push live",
                                words: [],
                            }],
                            meta: { model: "sensevoice-small", rtf: 0, threads: 1 },
                        }),
                    } as unknown as Voice,
                    session = createSession({
                        options: { microphone: false },
                        voice: fake,
                    });
                session.push(new Float32Array(16000).fill(0.2));
                session.cancel();
                const transcript = await session.done;
                assertEqual(transcript.text, "", "cancel text");
                assertEqual(session.fullText().text, "", "cancel live text");
                assertEqual(session.error, null, "cancel error");
            },
        },
        {
            name: "store/usage budget and pruning",
            run: async () => {
                const
                    spec = MODEL_SPECS[ModelId.WhisperBase],
                    english = MODEL_SPECS["whisper-base.en"],
                    store = fakeStore({
                        bytes: {
                            [checkpointKey({
                                checkpoint: spec.checkpoint,
                                file: spec.weightFile,
                            })]: new Uint8Array([1, 2, 3]),
                            [checkpointKey({
                                checkpoint: english.checkpoint,
                                file: english.weightFile,
                            })]: new Uint8Array([4, 5]),
                        },
                    });
                await touchModel({ store, model: ModelId.WhisperBase });
                const before = await readUsage({ store });
                assertEqual(
                    before.models[ModelId.WhisperBase]?.bytes,
                    5,
                    "usage bytes",
                );
                await ensureSpace({
                    store,
                    model: ModelId.SenseVoiceSmall,
                    allowedDiskSpaceMB: 1,
                });
                assertEqual(
                    await store.exists(checkpointKey({
                        checkpoint: spec.checkpoint,
                        file: spec.weightFile,
                    })),
                    false,
                    "pruned",
                );
                assertEqual(
                    await store.exists(checkpointKey({
                        checkpoint: english.checkpoint,
                        file: english.weightFile,
                    })),
                    false,
                    "pruned english",
                );
                await forgetModel({ store, model: ModelId.SenseVoiceSmall });
                const after = await readUsage({ store });
                assertEqual(
                    after.models[ModelId.SenseVoiceSmall],
                    undefined,
                    "forgotten",
                );
                assertEqual(DEFAULT_DISK_SPACE_MB, 512, "default cap");
            },
        },
        {
            name: "store/index getStore",
            run: async () => {
                const store = await getStore();
                assertEqual(typeof store.readText, "function", "readText");
                assertEqual(typeof store.size, "function", "size");
            },
        },
        {
            name: "index entry exports",
            run: () => {
                for (const key of [
                    "transcribe",
                    "preloadVoice",
                    "loadModel",
                    "isModelCached",
                ] as const) {
                    assertEqual(typeof entry[key], "function", `export ${key}`);
                }
                assertEqual(
                    entry.ModelId.SenseVoiceSmall,
                    "sensevoice-small",
                    "id",
                );
                assertEqual(entry.ProgressPhase.download, "download", "phase");
                assertEqual(entry.ProgressPhase.decode, "decode", "decode phase");
                assertEqual(entry.Quality.High, "high", "quality high");
                assertEqual(entry.Quality.Medium, "medium", "quality medium");
                assertEqual(entry.Quality.Low, "low", "quality low");
                assertEqual(entry.ThreadCount.Two, 2, "threads two");
                assertEqual(entry.ThreadCount.Four, 4, "threads four");
                assertEqual(entry.ThreadCount.Eight, 8, "threads eight");
                assertEqual(entry.Language.en, "en", "language");
                assertEqual(entry.Language.auto, "auto", "auto language");
                assertEqual(entry.TimestampMode.none, "none", "timestamp none");
                assertEqual(entry.TimestampMode.segment, "segment", "timestamp segment");
                assertEqual(entry.TimestampMode.word, "word", "timestamp word");
            },
        },
        {
            name: "session drives a fake voice",
            run: async () => {
                const
                    model = graphModel(),
                    fake = {
                        ensureModel: async (): Promise<Model> => model,
                        transcribe: (): Transcript => ({
                            text: "hello world",
                            language: Language.en,
                            duration: 1,
                            segments: [{
                                start: 0,
                                end: 1,
                                text: "hello world",
                                words: [],
                            }],
                            meta: { model: "sensevoice-small", rtf: 0, threads: 1 },
                        }),
                    } as unknown as Voice,
                    session = createSession({
                        options: { pcm: new Float32Array(32000).fill(0.2) },
                        voice: fake,
                    }),
                    snapshot = session.fullText();
                assertEqual(typeof snapshot.msElapsed, "number", "msElapsed");
                const transcript = await session.done;
                assertEqual(transcript.text, "hello world", "session text");
                assertEqual(session.error, null, "session error");
            },
        },
        {
            name: "session times decode without model download",
            run: async () => {
                const
                    model = graphModel(),
                    stall = (ms: number): void => {
                        const until = performance.now() + ms;
                        while (performance.now() < until) {
                            // burn decode time
                        }
                    },
                    fake = {
                        ensureModel: async (): Promise<Model> => {
                            await new Promise((resolve) => {
                                setTimeout(resolve, 400);
                            });
                            return model;
                        },
                        transcribe: (): Transcript => {
                            stall(80);
                            return {
                                text: "timed",
                                language: Language.en,
                                duration: 1,
                                segments: [{
                                    start: 0,
                                    end: 1,
                                    text: "timed",
                                    words: [],
                                }],
                                meta: {
                                    model: "sensevoice-small",
                                    rtf: 0.5,
                                    threads: 1,
                                },
                            };
                        },
                    } as unknown as Voice,
                    session = createSession({
                        options: { pcm: new Float32Array(16000).fill(0.2) },
                        voice: fake,
                    });
                assertEqual(
                    session.fullText().msElapsed,
                    0,
                    "initial elapsed",
                );
                await new Promise((resolve) => {
                    setTimeout(resolve, 100);
                });
                assertEqual(
                    session.fullText().msElapsed,
                    0,
                    "download elapsed",
                );
                const transcript = await session.done;
                const live = session.fullText();
                assert(transcript.meta.rtf > 0.02, "decode time missing");
                assert(transcript.meta.rtf < 0.3, "download counted in rtf");
                assert(live.msElapsed > 20, "live decode time missing");
                assert(live.msElapsed < 250, "live download counted");
            },
        },
        {
            name: "dictation lifecycle",
            run: async () => {
                const
                    states: DictationState[] = [],
                    texts: Array<[string, boolean]> = [],
                    capture = fakeCapture();
                let session = 0;
                dictationWorker.handler = (request) => {
                    if (request.type === DictationRequestType.start) {
                        session = request.id;
                        dictationWorker.reply({
                            type: DictationMessageType.progress,
                            ratio: 0.5,
                        });
                        dictationWorker.reply({
                            type: DictationMessageType.loaded,
                        });
                        return;
                    }
                    if (request.type === DictationRequestType.stop) {
                        dictationWorker.reply({
                            type: DictationMessageType.done,
                            id: request.id,
                            text: ` hello `,
                        });
                    }
                };
                const dictation = createDictation({
                    workerFactory: () => dictationWorkerInstance,
                    capture: capture.create,
                    language: Language.en,
                    onState: (state) => states.push(state),
                    onText: (text, final) => texts.push([text, final]),
                });
                assertEqual(
                    typeof isDictationSupported(),
                    `boolean`,
                    `support flag`,
                );
                await dictation.start();
                assertEqual(
                    dictation.state(),
                    DictationState.recording,
                    `recording`,
                );
                assert(
                    states.includes(DictationState.requesting),
                    `requesting state`,
                );
                capture.options.onChunk(new Float32Array(2000));
                dictationWorker.reply({
                    type: DictationMessageType.partial,
                    id: session,
                    text: `hel`,
                });
                assertEqual(texts[0]?.[0], `hel`, `partial text`);
                await dictation.finish();
                assertEqual(
                    dictation.state(),
                    DictationState.idle,
                    `idle after finish`,
                );
                assertEqual(
                    texts[texts.length - 1]?.[0],
                    `hello`,
                    `final text`,
                );
                assertEqual(
                    texts[texts.length - 1]?.[1],
                    true,
                    `final flag`,
                );
                const audio = dictationWorker.messages.filter(
                    (request) =>
                        request.type === DictationRequestType.audio,
                );
                assertEqual(audio.length, 1, `audio posted`);
                dictation.destroy();
            },
        },
        {
            name: "dictation pause, resume and cancel",
            run: async () => {
                const
                    texts: Array<[string, boolean]> = [],
                    capture = fakeCapture();
                dictationWorker.handler = (request) => {
                    if (request.type === DictationRequestType.start) {
                        dictationWorker.reply({
                            type: DictationMessageType.loaded,
                        });
                    }
                };
                const dictation = createDictation({
                    workerFactory: () => dictationWorkerInstance,
                    capture: capture.create,
                    onText: (text, final) => texts.push([text, final]),
                });
                await dictation.start();
                dictation.pause();
                assertEqual(
                    dictation.state(),
                    DictationState.paused,
                    `paused`,
                );
                assertEqual(capture.counts.paused, 1, `capture paused`);
                dictation.resume();
                assertEqual(
                    dictation.state(),
                    DictationState.recording,
                    `resumed`,
                );
                assertEqual(capture.counts.resumed, 1, `capture resumed`);
                dictation.cancel();
                assertEqual(
                    dictation.state(),
                    DictationState.idle,
                    `cancelled`,
                );
                assertEqual(capture.counts.stopped, 1, `capture stopped`);
                assertEqual(
                    texts[texts.length - 1]?.[0],
                    ``,
                    `empty final`,
                );
                assertEqual(
                    texts[texts.length - 1]?.[1],
                    true,
                    `final flag`,
                );
                dictation.destroy();
            },
        },
        {
            name: "dictation one active controller",
            run: async () => {
                const
                    textA: Array<[string, boolean]> = [],
                    captureA = fakeCapture(),
                    captureB = fakeCapture();
                dictationWorker.handler = (request) => {
                    if (request.type === DictationRequestType.start) {
                        dictationWorker.reply({
                            type: DictationMessageType.loaded,
                        });
                    }
                };
                const
                    first = createDictation({
                        workerFactory: () => dictationWorkerInstance,
                        capture: captureA.create,
                        onText: (text, final) => textA.push([text, final]),
                    }),
                    second = createDictation({
                        workerFactory: () => dictationWorkerInstance,
                        capture: captureB.create,
                    });
                await first.start();
                await second.start();
                assertEqual(
                    first.state(),
                    DictationState.idle,
                    `first idle`,
                );
                assertEqual(
                    second.state(),
                    DictationState.recording,
                    `second recording`,
                );
                assertEqual(
                    textA[textA.length - 1]?.[0],
                    ``,
                    `first cleared`,
                );
                second.cancel();
                first.destroy();
                second.destroy();
            },
        },
        {
            name: "dictation classifies worker errors",
            run: async () => {
                const errors: DictationErrorSource[] = [];
                dictationWorker.handler = (request) => {
                    if (request.type === DictationRequestType.start) {
                        dictationWorker.reply({
                            type: DictationMessageType.error,
                            id: request.id,
                            message: `model unavailable`,
                        });
                    }
                };
                const dictation = createDictation({
                    workerFactory: () => dictationWorkerInstance,
                    capture: fakeCapture().create,
                    onError: (error) => errors.push(error.source),
                });
                await dictation.start();
                assertEqual(
                    errors[0],
                    DictationErrorSource.model,
                    `model error`,
                );
                assertEqual(
                    dictation.state(),
                    DictationState.idle,
                    `idle after error`,
                );
                dictation.destroy();
            },
        },
        {
            name: "dictation worker dispatches requests",
            run: async () => {
                const
                    previous = window.onmessage,
                    replies: DictationMessage[] = [],
                    listener = (event: MessageEvent<DictationMessage>) => {
                        const data = event.data;
                        if (
                            data
                            && typeof data === `object`
                            && `type` in data
                        ) {
                            replies.push(data);
                        }
                    };
                window.addEventListener(`message`, listener);
                try {
                    serveDictation();
                    const handler = window.onmessage as
                        ((event: MessageEvent<DictationRequest>) => void) | null;
                    assert(handler !== null, `worker handler missing`);
                    handler?.({
                        data: { type: DictationRequestType.stop, id: 42 },
                    } as MessageEvent<DictationRequest>);
                    handler?.({
                        data: { type: DictationRequestType.cancel, id: 42 },
                    } as MessageEvent<DictationRequest>);
                    handler?.({
                        data: {
                            type: DictationRequestType.audio,
                            id: 42,
                            pcm: new Float32Array(4),
                        },
                    } as MessageEvent<DictationRequest>);
                    await new Promise((resolve) => setTimeout(resolve, 50));
                } finally {
                    window.removeEventListener(`message`, listener);
                    window.onmessage = previous;
                }
                assertEqual(replies.length, 1, `one worker reply`);
                assertEqual(
                    replies[0]?.type,
                    DictationMessageType.done,
                    `done reply`,
                );
                assertEqual(replies[0]?.id, 42, `reply id`);
                assertEqual(replies[0]?.text, ``, `empty reply`);
            },
        },
    ];

/** Run every test and publish the report. */
const main = async (): Promise<void> => {
    for (const definition of definitions) {
        await test(definition.name, definition.run);
    }
    const
        pass = outcomes.filter((outcome) => outcome.ok).length,
        fail = outcomes.length - pass,
        report = { pass, fail, tests: outcomes },
        json = JSON.stringify(report).replace("\"fail\":", "\"fail\": "),
        results = document.getElementById("results");
    if (results) {
        results.textContent = json;
    }
    console.log("TESTS_RESULT", json);
};

void main();
