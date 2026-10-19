// Model download and caching.
import {
    CHECKPOINT_IDS,
    HF_BASE,
    MODEL_SPECS,
    checkpointKey,
    resolveSpec,
} from "../models/specs";
import type {
    DeleteModelOptions,
    ModelSpec,
    ModelStore,
    PrepareModelOptions,
    ProgressHandler,
} from "../tensor/types";
const
    /** Download one model file when absent. */
    downloadFile = async (options: {
        store: ModelStore;
        spec: ModelSpec;
        file: string;
        onProgress?: ProgressHandler;
    }): Promise<void> => {
        const
            { store, spec, file } = options,
            key = checkpointKey({ checkpoint: spec.checkpoint, file }),
            isWeight = file === spec.weightFile;
        if (isWeight && store.hasBinary) {
            if (await store.hasBinary(key)) {
                return;
            }
        } else if (await store.exists(key)) {
            return;
        }
        // a raw weight is unreadable
        const staleRaw = isWeight && await store.exists(key);
        const response = await fetch(`${HF_BASE}/${spec.repo}/resolve/main/${file}`);
        if (!response.ok || !response.body) {
            throw new Error(`fetch failed ${response.status} ${file}`);
        }
        if (!isWeight) {
            await store.writeText({ key, text: await response.text() });
            return;
        }
        const total = Number(response.headers.get("content-length") ?? "0");
        let loaded = 0;
        await store.writeStream({
            key,
            stream: response.body,
            onProgress: (size) => {
                loaded += size;
                options.onProgress?.({
                    model: spec.id,
                    file,
                    loaded,
                    total,
                    ratio: total > 0 ? loaded / total : 0,
                });
            },
        });
        if (spec.weightBytes > 0 && loaded !== spec.weightBytes) {
            await store.remove(key);
            throw new Error(`size mismatch ${file} ${loaded} != ${spec.weightBytes}`);
        }
        if (staleRaw) {
            await store.removeRaw?.(key);
        }
    },
    /** Download every file of a model into the store. */
    prepareModel = async (
        options: PrepareModelOptions & { store: ModelStore },
    ): Promise<ModelSpec> => {
        const spec = resolveSpec({
            model: options.model,
            language: options.language,
        });
        for (const file of spec.files) {
            await downloadFile({
                store: options.store,
                spec,
                file,
                onProgress: options.onProgress,
            });
        }
        return spec;
    },
    /** Remove every file of a model from the store. */
    deleteModel = async (
        options: DeleteModelOptions & { store: ModelStore },
    ): Promise<void> => {
        for (const checkpoint of CHECKPOINT_IDS) {
            const spec = MODEL_SPECS[checkpoint];
            if (spec.id !== options.model) {
                continue;
            }
            for (const file of spec.files) {
                await options.store.remove(checkpointKey({ checkpoint, file }));
            }
        }
    };

export { prepareModel, deleteModel };
