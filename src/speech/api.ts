// Public API bound to one voice.
import { createSession } from "./session";
import type {
    ModelOptionsBase,
    PreloadVoiceOptions,
    TranscribeOptions,
    Voice,
    VoiceApi,
} from "../tensor/types";
const
    /** Build the public functions over one voice. */
    createApi = (voice: Voice): VoiceApi => ({
        /** Download and load a model ahead of time. */
        preloadVoice: async (
            options: PreloadVoiceOptions = {},
        ): Promise<void> => {
            await voice.ensureModel({
                model: options.model,
                language: options.language,
                allowedDiskSpaceMB: options.allowedDiskSpaceMB,
                weightsBaseUrl: options.weightsBaseUrl,
                weightCacheMB: options.weightCacheMB,
                threads: options.threads,
                onProgress: options.onProgress,
            });
        },
        /** Download and load a model ahead of time. */
        loadModel: async (options: ModelOptionsBase = {}) =>
            await voice.loadModel(options),
        /** True when every file of a model is already stored. */
        isModelCached: async (
            options: ModelOptionsBase = {},
        ): Promise<boolean> => await voice.isModelCached(options),
        /** Start a streaming transcription. */
        transcribe: (options: TranscribeOptions = {}) =>
            createSession({ options, voice }),
    });

export { createApi };
