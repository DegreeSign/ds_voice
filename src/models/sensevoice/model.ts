// SenseVoice model loading.
import { parseGguf } from "../../formats/gguf";
import { dedupeBoundary, hasSignal, joinText, planChunks } from "../../speech/segment";
import { padWindow } from "../../audio/pad";
import { decodeCtc } from "./ctc";
import { readSenseVoiceConfig, readSenseVoiceFrontend } from "./config";
import { computeSenseVoiceFeatures } from "./fbank";
import { encodeSenseVoice, senseVoiceLogits } from "./encoder";
import { createSenseVoiceVocabulary } from "./tokenizer";
import { createGgufWeights } from "./weights";
import { checkpointKey } from "../specs";
import { kernelThreads } from "../../tensor/wasm";
import { Language, ModelFamily } from "../../tensor/types";
import type {
    LoadedSenseVoice,
    LoadSenseVoiceOptions,
    Transcript,
} from "../../tensor/types";
const
    /** Blank token id, shared with the unknown token. */
    BLANK_ID = 0,
    /** Model training window in seconds. */
    MAX_WINDOW_SECONDS = 30,
    /** Load a SenseVoice model from a store. */
    loadSenseVoice = async (
        options: LoadSenseVoiceOptions,
    ): Promise<LoadedSenseVoice> => {
        const bytes = await options.store.readBytes(
            checkpointKey({
                checkpoint: options.checkpoint,
                file: options.spec.weightFile,
            }),
        );
        if (bytes === null) {
            throw new Error("missing sensevoice weights");
        }
        const
            file = parseGguf({ bytes }),
            weights = createGgufWeights({
                file,
                cacheBytes: options.weightCacheMB === undefined
                    ? undefined
                    : options.weightCacheMB * 1024 * 1024,
            }),
            vocabulary = createSenseVoiceVocabulary({ file });
        return {
            kind: ModelFamily.SenseVoice,
            name: options.model,
            config: readSenseVoiceConfig({ file }),
            frontend: readSenseVoiceFrontend({ file, weights }),
            tokenizer: vocabulary.tokenizer,
            vocabulary,
            weights,
        };
    },
    /** Transcribe one window of at most the training length. */
    transcribeWindow = (options: {
        model: LoadedSenseVoice;
        pcm: Float32Array;
        language?: Language;
        itn?: boolean;
    }): { text: string; language: Language } => {
        const
            { model, pcm } = options,
            window = padWindow({
                pcm,
                samplingRate: model.frontend.samplingRate,
            }),
            features = computeSenseVoiceFeatures({
                pcm: window.pcm,
                frontend: model.frontend,
            });
        if (features.frames <= 0) {
            return { text: "", language: options.language ?? Language.auto };
        }
        const
            requested = options.language ?? Language.auto,
            languageIds = model.config.languageIds,
            lid = languageIds.get(requested) ??
                languageIds.get(Language.auto) ?? 0,
            [event, emotion] = model.config.eventEmotion,
            textnorm = options.itn === false
                ? model.config.withoutItn
                : model.config.withItn,
            encoded = encodeSenseVoice({
                config: model.config,
                weights: model.weights,
                features: features.values,
                frames: features.frames,
                prefix: [lid, event, emotion, textnorm],
            }),
            logits = senseVoiceLogits({
                config: model.config,
                weights: model.weights,
                hidden: encoded.hidden,
                seq: encoded.seq,
            }),
            decoded = decodeCtc({
                logits,
                frames: encoded.seq,
                vocabSize: model.config.vocabSize,
                blankId: BLANK_ID,
            }),
            textIds = decoded.ids.filter(
                (id) => !model.vocabulary.isControl(id),
            ),
            text = model.tokenizer.decode(textIds);
        let language: Language = requested;
        if (requested === Language.auto) {
            language = decoded.ids.length > 0
                ? model.vocabulary.languageOf(decoded.ids[0]) ?? Language.en
                : Language.en;
        }
        return { text, language };
    },
    /** Transcribe one buffer with a SenseVoice model. */
    transcribeSenseVoice = (options: {
        model: LoadedSenseVoice;
        pcm: Float32Array;
        language?: Language;
        itn?: boolean;
    }): Transcript => {
        const
            { model, pcm } = options,
            rate = model.frontend.samplingRate,
            duration = pcm.length / rate;
        let
            text: string,
            language = options.language ?? Language.auto;
        if (!hasSignal({ pcm, samplingRate: rate })) {
            return {
                text: "",
                language,
                duration,
                segments: [{
                    start: 0,
                    end: duration,
                    text: "",
                    words: [],
                }],
                meta: {
                    model: model.name,
                    rtf: 0,
                    threads: kernelThreads(),
                },
            };
        }
        const started = performance.now();
        if (duration <= MAX_WINDOW_SECONDS) {
            const window = transcribeWindow({
                model,
                pcm,
                language: options.language,
                itn: options.itn,
            });
            text = window.text;
            language = window.language;
        } else {
            const
                chunks = planChunks({ pcm, samplingRate: rate }),
                texts: string[] = [];
            let previousEnd = 0;
            for (const chunk of chunks) {
                const
                    slice = pcm.subarray(
                        Math.floor(chunk.start * rate),
                        Math.floor(chunk.end * rate),
                    ),
                    window = transcribeWindow({
                        model,
                        pcm: slice,
                        language: options.language,
                        itn: options.itn,
                    }),
                    // overlapping chunks share boundary
                    overlapping = chunk.start < previousEnd - 1e-6;
                if (window.text.length > 0) {
                    const joined = overlapping
                        ? dedupeBoundary({
                            previous: texts[texts.length - 1] ?? "",
                            next: window.text,
                        })
                        : window.text;
                    if (joined.length > 0) {
                        texts.push(joined);
                    }
                    language = window.language;
                }
                previousEnd = chunk.end;
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
            segments: [{ start: 0, end: duration, text, words: [] }],
            meta: {
                model: model.name,
                rtf: duration > 0 ? elapsed / duration : 0,
                threads: kernelThreads(),
            },
        };
    };

export { loadSenseVoice, transcribeSenseVoice };
