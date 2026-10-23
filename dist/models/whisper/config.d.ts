import type { CheckpointStoreOptions, Constants, MelConfig, ModelConfig, RawGenerationConfig, RawModelConfig, RawPreprocessorConfig } from "../../tensor/types";
declare const 
/** Structural token ids verified against the checkpoint. */
TOKENS: {
    readonly endOfText: 50257;
    readonly startOfTranscript: 50258;
    readonly translate: 50358;
    readonly transcribe: 50359;
    readonly startOfLm: 50360;
    readonly startOfPrev: 50361;
    readonly noSpeech: 50362;
    readonly noTimestamps: 50363;
    readonly timestampBegin: 50364;
    readonly timestampEnd: 51864;
}, 
/** Load the three configuration files for a model. */
loadConfigs: (options: CheckpointStoreOptions) => Promise<{
    model: RawModelConfig;
    generation: RawGenerationConfig;
    preprocessor: RawPreprocessorConfig;
}>, 
/** Normalize the raw model config. */
toModelConfig: (raw: RawModelConfig) => ModelConfig, 
/** Normalize the raw preprocessor config. */
toMelConfig: (raw: RawPreprocessorConfig) => MelConfig, 
/** Flatten the stored mel filter bank to a row major f32 array. */
toMelFilters: (raw: RawPreprocessorConfig) => Float32Array, 
/** Build the token and generation constants. */
toConstants: (raw: RawGenerationConfig) => Constants, 
/** Assert the two suppression lists match across the config files. */
assertSuppressMatch: (options: {
    a: number[];
    b: number[];
}) => void;
export { TOKENS, loadConfigs, toModelConfig, toMelConfig, toMelFilters, toConstants, assertSuppressMatch, };
