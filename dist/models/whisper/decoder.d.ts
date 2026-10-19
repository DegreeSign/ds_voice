import { Language } from "../../tensor/types";
import type { CapturePair, DecoderState, LoadedModel, StepResult } from "../../tensor/types";
declare const 
/** Create decoder state and precompute cross keys and values. */
createDecoderState: (options: {
    model: LoadedModel;
    encoderOutput: Float32Array;
    encoderSeq: number;
}) => DecoderState, 
/** Run one decoder step for a single token. */
decoderStep: (options: {
    state: DecoderState;
    tokenId: number;
    capture?: CapturePair[];
}) => StepResult, 
/** Detect the language token for one encoder output. */
detectLanguage: (options: {
    state: DecoderState;
    constants: LoadedModel["constants"];
}) => {
    language: Language;
    token: number;
};
export { createDecoderState, decoderStep, detectLanguage };
