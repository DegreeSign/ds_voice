// Full voice bundle with both model families.
import { createVoiceEngine } from "./engine";
import { loadSenseVoice, transcribeSenseVoice } from "../models/sensevoice/model";
import { loadWhisper, transcribeWhisper } from "../models/whisper/model";
import { ModelFamily, TimestampMode } from "../tensor/types";
import type {
    ModelStore,
    Voice,
    VoiceFamilies,
} from "../tensor/types";
const
    /** Both model families. */
    families: VoiceFamilies = {
        [ModelFamily.SenseVoice]: {
            load: async (options) => await loadSenseVoice(options),
            transcribe: (options) => {
                const model = options.model;
                if (model.kind !== ModelFamily.SenseVoice) {
                    throw new Error("sensevoice model expected");
                }
                return transcribeSenseVoice({
                    model,
                    pcm: options.pcm,
                    language: options.language,
                    itn: options.itn,
                });
            },
        },
        [ModelFamily.Whisper]: {
            load: async (options) => await loadWhisper(options),
            transcribe: (options) => {
                const model = options.model;
                if (model.kind !== ModelFamily.Whisper) {
                    throw new Error("whisper model expected");
                }
                return transcribeWhisper({
                    model,
                    pcm: options.pcm,
                    language: options.language,
                    timestamps: options.timestamps ?? TimestampMode.none,
                    lowLatency: options.lowLatency,
                });
            },
        },
    },
    /** Create the voice functions for one store. */
    createVoice = (
        input: { store?: ModelStore } = {},
    ): Voice => createVoiceEngine({
        store: input.store,
        families,
    });

let shared: Voice | null = null;

/** Shared voice bound to the platform store. */
const getVoice = (): Voice => {
    if (!shared) {
        shared = createVoice();
    }
    return shared;
};

export { createVoice, getVoice };
