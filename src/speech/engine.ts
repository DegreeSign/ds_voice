// Platform neutral voice engine.
import { getStore } from "../store/index";
import {
    deleteModel,
    isCheckpointCached,
    prepareModel,
} from "../store/download";
import {
    DEFAULT_DISK_SPACE_MB,
    ensureSpace,
    forgetModel,
    touchModel,
} from "../store/usage";
import { DEFAULT_MODEL, resolveSpec } from "../models/specs";
import { prepareThreads } from "../tensor/wasm";
import { ProgressPhase } from "../tensor/types";
import type {
    CheckpointId,
    CreateVoiceEngineOptions,
    DeleteModelOptions,
    EnsureModelOptions,
    LoadModelOptions,
    Model,
    ModelId,
    ModelOptionsBase,
    ModelSpec,
    ModelStore,
    PrepareModelOptions,
    Quality,
    TranscribeAudioOptions,
    Transcript,
    Voice,
} from "../tensor/types";
const
    /** Create the voice functions for one store and family set. */
    createVoiceEngine = (input: CreateVoiceEngineOptions): Voice => {
        let
            cached: {
                model: ModelId;
                checkpoint: CheckpointId;
                quality: Quality;
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
                    quality: options.quality,
                    weightsBaseUrl: options.weightsBaseUrl,
                    onProgress: options.onProgress,
                });
            },
            loadFamily = async (options: {
                store: ModelStore;
                model: ModelId;
                spec: ModelSpec;
                weightCacheMB?: number;
            }): Promise<Model> => {
                const
                    { store, model, spec, weightCacheMB } = options,
                    family = input.families[spec.family];
                if (!family) {
                    throw new Error(`model family not bundled ${spec.family}`);
                }
                return await family.load({
                    store,
                    model,
                    checkpoint: spec.checkpoint,
                    spec,
                    weightCacheMB,
                });
            },
            reportDecode = (options: {
                spec: ModelSpec;
                loaded: number;
                total: number;
                onProgress?: ModelOptionsBase["onProgress"];
            }): void => {
                options.onProgress?.({
                    model: options.spec.id,
                    file: options.spec.weightFile,
                    phase: ProgressPhase.decode,
                    loaded: options.loaded,
                    total: options.total,
                    ratio: options.total > 0
                        ? Math.min(1, options.loaded / options.total)
                        : 1,
                });
            },
            load = async (options: LoadModelOptions = {}): Promise<Model> => {
                const
                    store = await resolveStore(),
                    model: ModelId = options.model ?? DEFAULT_MODEL,
                    spec = resolveSpec({
                        model,
                        language: options.language,
                        quality: options.quality,
                    });
                let
                    finalLoaded = 0,
                    finalTotal = 0;
                await prepareModel({
                    store,
                    model,
                    language: options.language,
                    quality: options.quality,
                    weightsBaseUrl: options.weightsBaseUrl,
                    onProgress: (progress) => {
                        finalLoaded = progress.loaded;
                        finalTotal = progress.total;
                        options.onProgress?.(progress);
                    },
                });
                await prepareThreads(options.threads);
                reportDecode({
                    spec,
                    loaded: finalLoaded,
                    total: finalTotal,
                    onProgress: options.onProgress,
                });
                return await loadFamily({
                    store,
                    model,
                    spec,
                    weightCacheMB: options.weightCacheMB,
                });
            },
            transcribe = (options: TranscribeAudioOptions): Transcript => {
                const family = input.families[options.model.kind];
                if (!family) {
                    throw new Error(
                        `model family not bundled ${options.model.kind}`,
                    );
                }
                return family.transcribe(options);
            },
            remove = async (options: DeleteModelOptions): Promise<void> => {
                const store = await resolveStore();
                await deleteModel({ store, model: options.model });
                await forgetModel({ store, model: options.model });
                if (cached?.model === options.model) {
                    cached = null;
                }
            },
            isModelCached = async (
                options: ModelOptionsBase = {},
            ): Promise<boolean> => {
                const
                    store = await resolveStore(),
                    spec = resolveSpec({
                        model: options.model,
                        language: options.language,
                        quality: options.quality,
                    });
                return await isCheckpointCached({ store, spec });
            },
            ensureModel = async (
                options: EnsureModelOptions = {},
            ): Promise<Model> => {
                const
                    model: ModelId = options.model ?? DEFAULT_MODEL,
                    spec = resolveSpec({
                        model,
                        language: options.language,
                        quality: options.quality,
                    });
                if (
                    cached?.model === model &&
                    cached.checkpoint === spec.checkpoint &&
                    cached.quality === spec.quality
                ) {
                    await prepareThreads(options.threads);
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
                        cached.checkpoint === spec.checkpoint &&
                        cached.quality === spec.quality
                    ) {
                        await prepareThreads(options.threads);
                        return cached.value;
                    }
                    const store = await resolveStore();
                    await ensureSpace({
                        store,
                        model,
                        language: options.language,
                        quality: options.quality,
                        allowedDiskSpaceMB: options.allowedDiskSpaceMB ??
                            DEFAULT_DISK_SPACE_MB,
                    });
                    let
                        finalLoaded = 0,
                        finalTotal = 0;
                    await prepareModel({
                        store,
                        model,
                        language: options.language,
                        quality: options.quality,
                        weightsBaseUrl: options.weightsBaseUrl,
                        onProgress: (progress) => {
                            finalLoaded = progress.loaded;
                            finalTotal = progress.total;
                            options.onProgress?.(progress);
                        },
                    });
                    await touchModel({ store, model });
                    await prepareThreads(options.threads);
                    reportDecode({
                        spec,
                        loaded: finalLoaded,
                        total: finalTotal,
                        onProgress: options.onProgress,
                    });
                    const value = await loadFamily({
                        store,
                        model,
                        spec,
                        weightCacheMB: options.weightCacheMB,
                    });
                    cached = {
                        model,
                        checkpoint: spec.checkpoint,
                        quality: spec.quality,
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
            isModelCached,
        };
    };

export { createVoiceEngine };
