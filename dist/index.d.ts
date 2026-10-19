import { Language, ModelId, TimestampMode } from "./tensor/types";
import type { PreloadVoiceOptions, TranscribeOptions, TranscribeSession } from "./tensor/types";
declare const 
/** Start a streaming transcription. */
transcribe: (options?: TranscribeOptions) => TranscribeSession, 
/** Download and load a model ahead of time. */
preloadVoice: (options?: PreloadVoiceOptions) => Promise<void>;
export { transcribe, preloadVoice, Language, ModelId, TimestampMode };
export type { DiskBudgetOptions, DownloadProgress, ModelOptionsBase, PreloadVoiceOptions, ProgressHandler, Segment, TextSnapshot, TranscribeBase, TranscribeOptions, TranscribeSession, Transcript, TranscriptMeta, Word, } from "./tensor/types";
