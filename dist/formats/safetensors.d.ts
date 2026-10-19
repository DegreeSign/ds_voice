import type { ParseSafetensorsOptions, SafetensorsFile, TensorStore } from "../tensor/types";
declare const 
/** Parse a safetensors buffer into a tensor store. */
parseSafetensors: (options: ParseSafetensorsOptions) => SafetensorsFile, 
/** Assert the Whisper base tensor invariants. */
assertWhisperShape: (options: {
    store: TensorStore;
    vocabSize: number;
    dModel: number;
}) => void;
export { parseSafetensors, assertWhisperShape };
