// Platform neutral voice engine.
import { getStore } from "../store/index";
import { deleteModel, prepareModel } from "../store/download";
import {
    DEFAULT_DISK_SPACE_MB,
    ensureSpace,
    forgetModel,
    touchModel,
} from "../store/usage";
import { DEFAULT_MODEL, MODEL_SPECS, resolveSpec } from "../models/specs";
import { loadSenseVoice, transcribeSenseVoice } from "../models/sensevoice/model";
import { loadWhisper, transcribeWhisper } from "../models/whisper/model";
import { ModelFamily, TimestampMode } from "../tensor/types";
import type {
    CheckpointId,
    DeleteModelOptions,
    EnsureModelOptions,
    LoadModelOptions,
    Model,
    ModelId,
    ModelSpec,
    ModelStore,
    PrepareModelOptions,
    TranscribeAudioOptions,
    Transcript,
    Voice,
} from "../tensor/types";
const
    /** Create the voice functions for one store. */
    createVoice = (input: { store?: ModelStore } = {}): Voice => {
        let
            cached: {
                model: ModelId;
                checkpoint: CheckpointId;
                value: Model;
            } | null = null,
            queue: Promise<void> = Promise.resolve(),
            storePromise: Promise<ModelStore> | null = null;
        const
            resolveStore = async (): Promise<ModelStore> => {
                if (input.store) {
                    return input.store;
                }
                if (!storePromise) {
                    storePromise = getStore();
                }
                return await storePromise;
            },
            prepare = async (
                options: PrepareModelOptions = {},
            ): Promise<ModelSpec> => {
                const store = await resolveStore();
                return await prepareModel({
                    store,
                    model: options.model,
                    language: options.language,
                    onProgress: options.onProgress,
                });
            },
            loadFamily = async (options: {
                store: ModelStore;
                model: ModelId;
                checkpoint: CheckpointId;
            }): Promise<Model> => {
                const
                    { store, model, checkpoint } = options,
                    spec = MODEL_SPECS[checkpoint];
                if (spec.family === ModelFamily.SenseVoice) {
                    return await loadSenseVoice({ store, model, checkpoint });
                }
                return await loadWhisper({ store, model, checkpoint });
            },
            load = async (options: LoadModelOptions = {}): Promise<Model> => {
                const
                    store = await resolveStore(),
                    model: ModelId = options.model ?? DEFAULT_MODEL,
                    spec = resolveSpec({
                        model,
                        language: options.language,
                    });
                await prepareModel({
                    store,
                    model,
                    language: options.language,
                    onProgress: options.onProgress,
                });
                return await loadFamily({
                    store,
                    model,
                    checkpoint: spec.checkpoint,
                });
            },
            transcribe = (options: TranscribeAudioOptions): Transcript => {
                const
                    { model, pcm } = options,
                    timestamps = options.timestamps ?? TimestampMode.none;
                if (model.kind === ModelFamily.SenseVoice) {
                    return transcribeSenseVoice({
                        model,
                        pcm,
                        language: options.language,
                        itn: options.itn,
                    });
                }
                return transcribeWhisper({
                    model,
                    pcm,
                    language: options.language,
                    timestamps,
                    lowLatency: options.lowLatency,
                });
            },
            remove = async (options: DeleteModelOptions): Promise<void> => {
                const store = await resolveStore();
                await deleteModel({ store, model: options.model });
                await forgetModel({ store, model: options.model });
                if (cached?.model === options.model) {
                    cached = null;
                }
            },
            ensureModel = async (
                options: EnsureModelOptions = {},
            ): Promise<Model> => {
                const
                    model: ModelId = options.model ?? DEFAULT_MODEL,
                    spec = resolveSpec({
                        model,
                        language: options.language,
                    });
                if (
                    cached?.model === model &&
                    cached.checkpoint === spec.checkpoint
                ) {
                    return cached.value;
                }
                let release = (): void => undefined;
                const previous = queue;
                queue = new Promise<void>((resolve) => {
                    release = resolve;
                });
                await previous;
                try {
                    if (
                        cached?.model === model &&
                        cached.checkpoint === spec.checkpoint
                    ) {
                        return cached.value;
                    }
                    const store = await resolveStore();
                    await ensureSpace({
                        store,
                        model,
                        language: options.language,
                        allowedDiskSpaceMB: options.allowedDiskSpaceMB ??
                            DEFAULT_DISK_SPACE_MB,
                    });
                    await prepareModel({
                        store,
                        model,
                        language: options.language,
                        onProgress: options.onProgress,
                    });
                    await touchModel({ store, model });
                    const value = await loadFamily({
                        store,
                        model,
                        checkpoint: spec.checkpoint,
                    });
                    cached = {
                        model,
                        checkpoint: spec.checkpoint,
                        value,
                    };
                    return value;
                } finally {
                    release();
                }
            };

        return {
            prepareModel: prepare,
            loadModel: load,
            transcribe,
            deleteModel: remove,
            ensureModel,
        };
    };

let shared: Voice | null = null;

/** Shared voice bound to the platform store. */
const getVoice = (): Voice => {
    if (!shared) {
        shared = createVoice();
    }
    return shared;
};

export { createVoice, getVoice };
