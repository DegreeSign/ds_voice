// Voice bundle with the SenseVoice family only.
import { createVoiceEngine } from "./engine";
import { loadSenseVoice, transcribeSenseVoice } from "../models/sensevoice/model";
import { ModelFamily } from "../tensor/types";
import type {
    ModelStore,
    Voice,
    VoiceFamilies,
} from "../tensor/types";
const
    /** SenseVoice family only. */
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
