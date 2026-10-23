// Dictation controller entry.
import { createDictation, isDictationSupported } from "./controller";
import {
    DictationErrorSource,
    DictationPermission,
    DictationState,
} from "./types";
import { Language, ModelId, Quality, ThreadCount, TimestampMode } from "../tensor/types";

export {
    createDictation,
    isDictationSupported,
    DictationState,
    DictationPermission,
    DictationErrorSource,
    Language,
    ModelId,
    Quality,
    ThreadCount,
    TimestampMode,
};
export type {
    DictationController,
    DictationDetail,
    DictationError,
    DictationOptions,
} from "./types";
