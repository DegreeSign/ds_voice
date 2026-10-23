// Cache usage and disk budget.
import { deleteModel } from "./download";
import {
    CHECKPOINT_IDS,
    MODEL_IDS,
    MODEL_SPECS,
    checkpointFiles,
    checkpointKey,
    resolveSpec,
    weightFiles,
} from "../models/specs";
import { ModelId } from "../tensor/types";
import type {
    CheckpointId,
    EnsureSpaceOptions,
    ModelSpec,
    ModelUsage,
    ModelUsageEntry,
    ModelStore,
    TouchModelOptions,
    UsageStoreOptions,
} from "../tensor/types";
const
    /** Default cache cap in MB. */
    DEFAULT_DISK_SPACE_MB = 512,
    /** Usage record key. */
    USAGE_KEY = "usage.json",
    /** Weight estimate for checkpoints without a declared size. */
    WEIGHT_ESTIMATE_BYTES = 180 * 1024 * 1024,
    /** Estimated bytes for one text file not yet cached. */
    TEXT_ESTIMATE_BYTES = 256 * 1024,
    /** Read the persisted usage, empty when absent or malformed. */
    readUsage = async (options: UsageStoreOptions): Promise<ModelUsage> => {
        const text = await options.store.readText(USAGE_KEY);
        if (!text) {
            return { models: {} };
        }
        try {
            const
                parsed = JSON.parse(text) as ModelUsage,
                models: ModelUsage["models"] = {};
            for (const id of MODEL_IDS) {
                const entry = parsed?.models?.[id];
                if (entry) {
                    models[id] = entry;
                }
            }
            return { models };
        } catch {
            return { models: {} };
        }
    },
    /** Persist the usage. */
    writeUsage = async (options: {
        store: ModelStore;
        usage: ModelUsage;
    }): Promise<void> => {
        await options.store.writeText({
            key: USAGE_KEY,
            text: JSON.stringify({ models: options.usage.models }),
        });
    },
    /** Size of one stored key in bytes. */
    storedSize = async (options: {
        store: ModelStore;
        key: string;
    }): Promise<number> => {
        const { store, key } = options;
        if (store.size) {
            return await store.size(key);
        }
        const bytes = await store.readBytes(key);
        if (bytes) {
            return bytes.length;
        }
        const text = await store.readText(key);
        return text ? new TextEncoder().encode(text).length : 0;
    },
    /** True when one checkpoint weight file is present. */
    checkpointPresent = async (options: {
        store: ModelStore;
        spec: ModelSpec;
    }): Promise<boolean> => {
        const
            { store, spec } = options,
            key = checkpointKey({
                checkpoint: spec.checkpoint,
                file: spec.weightFile,
            });
        return await store.exists(key);
    },
    /** True when any quality of one checkpoint is present. */
    checkpointCached = async (options: {
        store: ModelStore;
        checkpoint: CheckpointId;
    }): Promise<boolean> => {
        const spec = MODEL_SPECS[options.checkpoint];
        for (const file of weightFiles(spec)) {
            const key = checkpointKey({
                checkpoint: options.checkpoint,
                file,
            });
            if (await options.store.exists(key)) {
                return true;
            }
        }
        return false;
    },
    /** True when any checkpoint of a model is present. */
    modelPresent = async (options: {
        store: ModelStore;
        model: ModelId;
    }): Promise<boolean> => {
        for (const checkpoint of CHECKPOINT_IDS) {
            if (MODEL_SPECS[checkpoint].id !== options.model) {
                continue;
            }
            if (
                await checkpointCached({
                    store: options.store,
                    checkpoint,
                })
            ) {
                return true;
            }
        }
        return false;
    },
    /** Actual cached size of every checkpoint of one model in bytes. */
    modelSize = async (options: {
        store: ModelStore;
        model: ModelId;
    }): Promise<number> => {
        let total = 0;
        for (const checkpoint of CHECKPOINT_IDS) {
            const spec = MODEL_SPECS[checkpoint];
            if (spec.id !== options.model) {
                continue;
            }
            for (const file of checkpointFiles(spec)) {
                total += await storedSize({
                    store: options.store,
                    key: checkpointKey({ checkpoint, file }),
                });
            }
        }
        return total;
    },
    /** Estimated size of one checkpoint once fully cached. */
    estimateSize = async (options: {
        store: ModelStore;
        spec: ModelSpec;
    }): Promise<number> => {
        const
            { store, spec } = options,
            { checkpoint } = spec;
        let total = 0;
        for (const file of spec.files) {
            const size = await storedSize({
                store,
                key: checkpointKey({ checkpoint, file }),
            });
            if (size > 0) {
                total += size;
            } else if (file === spec.weightFile) {
                total += spec.weightBytes > 0
                    ? spec.weightBytes
                    : WEIGHT_ESTIMATE_BYTES;
            } else {
                total += TEXT_ESTIMATE_BYTES;
            }
        }
        return total;
    },
    /** Record the size and last use of one cached model. */
    touchModel = async (options: TouchModelOptions): Promise<void> => {
        const
            { store, model } = options,
            usage = await readUsage({ store }),
            entry: ModelUsageEntry = {
                bytes: await modelSize({ store, model }),
                lastUsed: Date.now(),
            };
        usage.models[model] = entry;
        await writeUsage({ store, usage });
    },
    /** Drop the usage record of one model. */
    forgetModel = async (options: TouchModelOptions): Promise<void> => {
        const
            { store, model } = options,
            usage = await readUsage({ store });
        delete usage.models[model];
        await writeUsage({ store, usage });
    },
    /** Prune least recently used models until the target fits the cap. */
    ensureSpace = async (options: EnsureSpaceOptions): Promise<void> => {
        const
            { store } = options,
            wanted = resolveSpec({
                model: options.model,
                language: options.language,
                quality: options.quality,
            }),
            capped = Number.isFinite(options.allowedDiskSpaceMB)
                ? options.allowedDiskSpaceMB
                : DEFAULT_DISK_SPACE_MB,
            cap = Math.max(1, capped) * 1024 * 1024,
            usage = await readUsage({ store });
        if (
            await checkpointPresent({
                store,
                spec: wanted,
            })
        ) {
            await writeUsage({ store, usage });
            return;
        }
        const
            sizes = new Map<ModelId, number>(),
            stored: ModelId[] = [];
        let current = 0;
        for (const id of MODEL_IDS) {
            if (!(await modelPresent({ store, model: id }))) {
                continue;
            }
            const size = await modelSize({ store, model: id });
            sizes.set(id, size);
            stored.push(id);
            current += size;
            usage.models[id] = {
                bytes: size,
                lastUsed: usage.models[id]?.lastUsed ?? 0,
            };
        }
        const
            required = await estimateSize({ store, spec: wanted }),
            candidates = stored
                .filter((id) => id !== wanted.id)
                .sort((a, b) => {
                    const left = usage.models[a]?.lastUsed ?? 0;
                    const right = usage.models[b]?.lastUsed ?? 0;
                    return left - right;
                });
        for (const id of candidates) {
            if (current + required <= cap) {
                break;
            }
            await deleteModel({ store, model: id });
            current -= sizes.get(id) ?? 0;
            delete usage.models[id];
        }
        await writeUsage({ store, usage });
    };

export {
    DEFAULT_DISK_SPACE_MB,
    ensureSpace,
    touchModel,
    forgetModel,
    readUsage,
};
