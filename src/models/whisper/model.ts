// Whisper model loading.
import {
    TOKENS,
    assertSuppressMatch,
    loadConfigs,
    toConstants,
    toMelConfig,
    toMelFilters,
    toModelConfig,
} from "./config";
import { loadTokenizer } from "./tokenizer";
import { computeMel } from "../../speech/mel";
import { encode } from "./encoder";
import { createDecoderState, detectLanguage } from "./decoder";
import { generate } from "./generate";
import { computeDtw } from "./dtw";
import { buildSegments, buildWords } from "./words";
import { dedupeBoundary, joinText, planChunks } from "../../speech/segment";
import { assertWhisperShape, parseSafetensors } from "../../formats/safetensors";
import { parseGguf } from "../../formats/gguf";
import { createWhisperWeights } from "./weights";
import { kernelThreads } from "../../tensor/wasm";
import { checkpointKey } from "../specs";
import { Language, ModelFamily, TimestampMode } from "../../tensor/types";
import type {
    CapturePair,
    LoadedModel,
    Segment,
    StoreModelOptions,
    TensorStore,
    Transcript,
    WhisperWindowOptions,
    WindowResult,
} from "../../tensor/types";
const
    /** Load a Whisper model from a store. */
    loadWhisper = async (
        options: StoreModelOptions,
    ): Promise<LoadedModel> => {
        const
            source = {
                store: options.store,
                model: options.model,
                checkpoint: options.checkpoint,
            },
            { model, generation, preprocessor } = await loadConfigs(source),
            { spec } = options,
            candidates = spec.weightFile === "model.safetensors"
                ? [spec.weightFile]
                : [spec.weightFile, "model.safetensors"];
        let tensors: TensorStore | null = null;
        let quantized = false;
        for (const file of candidates) {
            const bytes = await options.store.readBytes(
                checkpointKey({
                    checkpoint: options.checkpoint,
                    file,
                }),
            );
            if (bytes === null) {
                continue;
            }
            if (file.endsWith(".gguf")) {
                tensors = createWhisperWeights({
                    file: parseGguf({ bytes }),
                    cacheBytes: options.weightCacheMB === undefined
                        ? undefined
                        : options.weightCacheMB * 1024 * 1024,
                });
                quantized = true;
            } else {
                tensors = parseSafetensors({ bytes }).store;
            }
            break;
        }
        if (tensors === null) {
            throw new Error("missing whisper weights");
        }
        const
            tokenizer = await loadTokenizer(source),
            constants = toConstants(generation);
        assertSuppressMatch({
            a: model.suppress_tokens,
            b: generation.suppress_tokens,
        });
        assertWhisperShape({
            store: tensors,
            vocabSize: model.vocab_size,
            dModel: model.d_model,
            allowQuantized: quantized,
        });
        return {
            kind: ModelFamily.Whisper,
            name: options.model,
            config: toModelConfig(model),
            constants,
            mel: toMelConfig(preprocessor),
            melFilters: toMelFilters(preprocessor),
            tokenizer,
            tensors,
        };
    },
    /** Transcribe one audio window of at most 30 seconds. */
    transcribeWindow = (options: WhisperWindowOptions): WindowResult => {
        const
            { model, pcm } = options,
            constant = model.constants,
            mel = computeMel({
                pcm,
                mel: model.mel,
                filters: model.melFilters,
                padToChunk: options.lowLatency !== true,
            }),
            encoded = encode({ model, mel: mel.values });
        let
            languageId: number | null = null,
            language: Language = Language.en;
        if (!constant.isMultilingual) {
            language = Language.en;
        } else if (!options.language || options.language === Language.auto) {
            const
                probe = createDecoderState({
                    model,
                    encoderOutput: encoded.output,
                    encoderSeq: encoded.seq,
                }),
                detected = detectLanguage({ state: probe, constants: constant });
            languageId = detected.token;
            language = detected.language;
        } else {
            const id = constant.languageIds.get(options.language);
            if (id === undefined) {
                throw new Error(`unknown language ${options.language}`);
            }
            languageId = id;
            language = options.language;
        }
        const withTimestamps = options.timestamps !== TimestampMode.none;
        let prompt: number[];
        if (!constant.isMultilingual) {
            // english models lack tokens
            prompt = withTimestamps
                ? [constant.startOfTranscript]
                : [constant.startOfTranscript, constant.noTimestampsToken];
        } else {
            const head = [
                constant.startOfTranscript,
                languageId as number,
                constant.transcribeToken,
            ];
            prompt = withTimestamps ? head : [...head, constant.noTimestampsToken];
        }
        const
            capture: CapturePair[] = withTimestamps &&
                options.timestamps === TimestampMode.word
                ? constant.alignmentHeads.map(([layer, head]) => ({ layer, head }))
                : [],
            result = generate({
                model,
                encoderOutput: encoded.output,
                encoderSeq: encoded.seq,
                prompt,
                timestamps: withTimestamps,
                capture,
                guardRepeat: options.lowLatency === true,
            }),
            generated = stripEnd(result.ids),
            plain = generated.filter((id) => id < constant.timestampBegin),
            text = model.tokenizer.decode(plain).trim();
        if (options.timestamps === TimestampMode.word) {
            const
                times = computeDtw({
                    attention: result.attention,
                    tokenCount: result.attention.length,
                    frameCount: encoded.seq,
                    heads: constant.alignmentHeads.length,
                }),
                tokenTimes = new Float32Array(generated.length);
            for (let j = 0; j < generated.length; j += 1) {
                // align dtw jump rows
                const row = Math.min(prompt.length + j, times.times.length - 1);
                tokenTimes[j] = times.times[row];
            }
            const
                words = buildWords({
                    ids: generated,
                    tokenTimes,
                    tokenizer: model.tokenizer,
                    timeOffset: options.timeOffset,
                }),
                segments = buildSegments({ words });
            return { ids: result.ids, text, words, segments, language };
        }
        if (options.timestamps === TimestampMode.segment) {
            const segments = segmentsFromTokens({
                ids: generated,
                tokenizer: model.tokenizer,
                timeOffset: options.timeOffset,
                constants: constant,
            });
            return { ids: result.ids, text, words: [], segments, language };
        }
        return {
            ids: result.ids,
            text,
            words: [],
            segments: [
                {
                    start: options.timeOffset,
                    end: options.timeOffset + pcm.length / model.mel.samplingRate,
                    text,
                    words: [],
                },
            ],
            language,
        };
    },
    /** Drop a trailing end of text token. */
    stripEnd = (ids: number[]): number[] => {
        const out = ids.slice();
        while (out.length > 0 && out[out.length - 1] === TOKENS.endOfText) {
            out.pop();
        }
        return out;
    },
    /** Build segments from timestamp token pairs. */
    segmentsFromTokens = (options: {
        ids: number[];
        tokenizer: LoadedModel["tokenizer"];
        timeOffset: number;
        constants: LoadedModel["constants"];
    }): Segment[] => {
        const
            { ids, tokenizer, timeOffset, constants } = options,
            segments: Segment[] = [];
        let
            current: number[] = [],
            start: number | null = null;
        const flush = (end: number): void => {
            const text = tokenizer.decode(current).trim();
            if (text.length > 0 && start !== null) {
                segments.push({ start, end, text, words: [] });
            }
            current = [];
            start = null;
        };
        for (const id of ids) {
            if (id >= constants.timestampBegin) {
                const time = (id - constants.timestampBegin) *
                    constants.timestampStep +
                    timeOffset;
                if (start === null) {
                    start = time;
                } else {
                    flush(time);
                }
            } else {
                current.push(id);
            }
        }
        if (start !== null && current.length > 0) {
            flush(start);
        }
        return segments;
    },
    /** Transcribe arbitrary length audio with a Whisper model. */
    transcribeWhisper = (options: {
        model: LoadedModel;
        pcm: Float32Array;
        language?: Language;
        timestamps: TimestampMode;
        lowLatency?: boolean;
    }): Transcript => {
        const
            { model, pcm } = options,
            timestamps = options.timestamps,
            started = performance.now(),
            duration = pcm.length / model.mel.samplingRate,
            singleWindow = model.mel.chunkLength;
        let
            segments: Segment[] = [],
            language = options.language ?? Language.auto,
            text: string;
        if (duration <= singleWindow) {
            const result = transcribeWindow({
                model,
                pcm,
                language: options.language,
                timestamps,
                timeOffset: 0,
                lowLatency: options.lowLatency,
            });
            segments = result.segments;
            language = result.language;
            text = result.text;
        } else {
            const
                chunks = planChunks({
                    pcm,
                    samplingRate: model.mel.samplingRate,
                }),
                texts: string[] = [];
            for (const chunk of chunks) {
                const
                    startSample = Math.floor(chunk.start * model.mel.samplingRate),
                    endSample = Math.floor(chunk.end * model.mel.samplingRate),
                    slice = pcm.subarray(startSample, endSample),
                    result = transcribeWindow({
                        model,
                        pcm: slice,
                        language: options.language,
                        timestamps,
                        timeOffset: chunk.start,
                        lowLatency: options.lowLatency,
                    });
                if (result.text.length === 0) {
                    continue;
                }
                const deduped = dedupeBoundary({
                    previous: texts[texts.length - 1] ?? "",
                    next: result.text,
                });
                if (deduped.length > 0) {
                    texts.push(deduped);
                }
                segments = segments.concat(result.segments);
                language = result.language;
            }
            text = "";
            for (const part of texts) {
                text = joinText({ previous: text, next: part });
            }
            text = text.trim();
        }
        const elapsed = (performance.now() - started) / 1000;
        return {
            text,
            language,
            duration,
            segments,
            meta: {
                model: model.name,
                rtf: duration > 0 ? elapsed / duration : 0,
                threads: kernelThreads(),
            },
        };
    };

export { loadWhisper, transcribeWhisper };
