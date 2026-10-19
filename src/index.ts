// Package entry, shared by Node and the browser.
import { getVoice } from "./speech/voice";
import { createSession } from "./speech/session";
import { Language, ModelId, TimestampMode } from "./tensor/types";
import type {
    PreloadVoiceOptions,
    TranscribeOptions,
    TranscribeSession,
} from "./tensor/types";
const
    /** Start a streaming transcription. */
    transcribe = (options: TranscribeOptions = {}): TranscribeSession => {
        return createSession({ options, voice: getVoice() });
    },
    /** Download and load a model ahead of time. */
    preloadVoice = async (
        options: PreloadVoiceOptions = {},
    ): Promise<void> => {
        await getVoice().ensureModel({
            model: options.model,
            language: options.language,
            allowedDiskSpaceMB: options.allowedDiskSpaceMB,
            onProgress: options.onProgress,
        });
    };

export { transcribe, preloadVoice, Language, ModelId, TimestampMode };
export type {
    DiskBudgetOptions,
    DownloadProgress,
    ModelOptionsBase,
    PreloadVoiceOptions,
    ProgressHandler,
    Segment,
    TextSnapshot,
    TranscribeBase,
    TranscribeOptions,
    TranscribeSession,
    Transcript,
    TranscriptMeta,
    Word,
} from "./tensor/types";
