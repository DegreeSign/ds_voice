// Greedy Whisper generation.
// Ported from transformers 4.44.0.
import { argmax } from "../../tensor/kernels";
import { createDecoderState, decoderStep } from "./decoder";
import type {
    GenerateOptions,
    GenerateOutput,
    LoadedModel,
} from "../../tensor/types";
const
    /** Apply the suppression lists and the timestamp rules. */
    processLogits = (options: {
        logits: Float32Array;
        allIds: number[];
        beginIndex: number;
        constants: LoadedModel["constants"];
        timestamps: boolean;
    }): void => {
        const { logits, allIds, beginIndex, constants, timestamps } = options;
        if (allIds.length === beginIndex) {
            for (const id of constants.beginSuppressTokens) {
                logits[id] = -Infinity;
            }
        }
        for (const id of constants.suppressTokens) {
            logits[id] = -Infinity;
        }
        if (timestamps) {
            timestampProcessor({
                logits,
                allIds,
                beginIndex,
                constants,
            });
        }
    },
    /** Port of WhisperTimeStampLogitsProcessor. */
    timestampProcessor = (options: {
        logits: Float32Array;
        allIds: number[];
        beginIndex: number;
        constants: LoadedModel["constants"];
    }): void => {
        const
            { logits, allIds, beginIndex, constants } = options,
            timestampBegin = constants.timestampBegin,
            eos = constants.eosTokenId;
        logits[constants.noTimestampsToken] = -Infinity;
        const
            seq = allIds.slice(beginIndex),
            lastWasTimestamp =
                seq.length >= 1 && seq[seq.length - 1] >= timestampBegin,
            penultimateWasTimestamp =
                seq.length < 2 || seq[seq.length - 2] >= timestampBegin;
        if (lastWasTimestamp) {
            if (penultimateWasTimestamp) {
                for (let id = timestampBegin; id < logits.length; id += 1) {
                    logits[id] = -Infinity;
                }
            } else {
                for (let id = 0; id < eos; id += 1) {
                    logits[id] = -Infinity;
                }
            }
        }
        let lastTimestamp = -1;
        for (const id of seq) {
            if (id >= timestampBegin) {
                lastTimestamp = id;
            }
        }
        if (lastTimestamp >= 0) {
            let timestampLast: number;
            if (lastWasTimestamp && !penultimateWasTimestamp) {
                timestampLast = lastTimestamp;
            } else {
                timestampLast = lastTimestamp + 1;
            }
            for (let id = timestampBegin; id < timestampLast; id += 1) {
                logits[id] = -Infinity;
            }
        }
        if (allIds.length === beginIndex) {
            for (let id = 0; id < timestampBegin; id += 1) {
                logits[id] = -Infinity;
            }
            const lastAllowed =
                timestampBegin + constants.maxInitialTimestampIndex;
            for (let id = lastAllowed + 1; id < logits.length; id += 1) {
                logits[id] = -Infinity;
            }
        }
        const logprobs = logSoftmax(logits);
        let timestampLogprob = -Infinity;
        for (let id = timestampBegin; id < logits.length; id += 1) {
            timestampLogprob = logAdd({ a: timestampLogprob, b: logprobs[id] });
        }
        let maxText = -Infinity;
        for (let id = 0; id < timestampBegin; id += 1) {
            if (logprobs[id] > maxText) {
                maxText = logprobs[id];
            }
        }
        if (timestampLogprob > maxText) {
            for (let id = 0; id < timestampBegin; id += 1) {
                if (logits[id] !== -Infinity) {
                    logits[id] = -Infinity;
                }
            }
        }
    },
    /** Numerically stable log add. */
    logAdd = (options: { a: number; b: number }): number => {
        const { a, b } = options;
        if (a === -Infinity) {
            return b;
        }
        if (b === -Infinity) {
            return a;
        }
        const max = a > b ? a : b;
        return max + Math.log(Math.exp(a - max) + Math.exp(b - max));
    },
    /** Log softmax over the full logits. */
    logSoftmax = (values: Float32Array): Float32Array => {
        let max = -Infinity;
        for (let i = 0; i < values.length; i += 1) {
            if (values[i] > max) {
                max = values[i];
            }
        }
        let sum = 0;
        for (let i = 0; i < values.length; i += 1) {
            sum += Math.exp(values[i] - max);
        }
        const
            logSum = max + Math.log(sum),
            out = new Float32Array(values.length);
        for (let i = 0; i < values.length; i += 1) {
            out[i] = values[i] - logSum;
        }
        return out;
    },
    /** Detect a repeating tail, returning its period. */
    repeatPeriod = (ids: number[]): number => {
        const
            n = ids.length,
            maxPeriod = Math.floor(n / 3);
        for (let p = 1; p <= maxPeriod; p += 1) {
            let match = true;
            for (let i = 0; i < 2 * p; i += 1) {
                if (ids[n - 1 - i] !== ids[n - 1 - i - p]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                return p;
            }
        }
        return 0;
    },
    /** Run a full greedy decode. */
    generate = (options: GenerateOptions): GenerateOutput => {
        const
            { model, prompt } = options,
            state = createDecoderState({
                model,
                encoderOutput: options.encoderOutput,
                encoderSeq: options.encoderSeq,
            }),
            capture = options.capture ?? [],
            attention: Float32Array[][] = [],
            allIds: number[] = prompt.slice();
        for (let i = 0; i < prompt.length - 1; i += 1) {
            const result = decoderStep({
                state,
                tokenId: prompt[i],
                capture,
            });
            attention.push(result.captures);
        }
        const
            maxTokens =
                options.maxTokens ?? model.config.maxLength - prompt.length,
            ids: number[] = [];
        let
            eos = false,
            lastToken = prompt[prompt.length - 1];
        for (let step = 0; step < maxTokens; step += 1) {
            const result = decoderStep({
                state,
                tokenId: lastToken,
                capture,
            });
            attention.push(result.captures);
            const logits = result.logits.slice();
            processLogits({
                logits,
                allIds,
                beginIndex: prompt.length,
                constants: model.constants,
                timestamps: options.timestamps,
            });
            const best = argmax({ values: logits });
            if (best === model.constants.eosTokenId) {
                eos = true;
                allIds.push(best);
                ids.push(best);
                break;
            }
            allIds.push(best);
            ids.push(best);
            lastToken = best;
            if (options.guardRepeat === true && ids.length >= 9) {
                const period = repeatPeriod(ids);
                if (period > 0) {
                    ids.length -= 2 * period;
                    allIds.length = prompt.length + ids.length;
                    attention.length -= 2 * period;
                    break;
                }
            }
        }
        return { ids, allIds, attention, eos };
    };

export { generate };
