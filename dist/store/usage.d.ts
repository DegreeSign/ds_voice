import type { EnsureSpaceOptions, ModelUsage, TouchModelOptions, UsageStoreOptions } from "../tensor/types";
declare const 
/** Default cache cap in MB. */
DEFAULT_DISK_SPACE_MB = 512, 
/** Read the persisted usage, empty when absent or malformed. */
readUsage: (options: UsageStoreOptions) => Promise<ModelUsage>, 
/** Record the size and last use of one cached model. */
touchModel: (options: TouchModelOptions) => Promise<void>, 
/** Drop the usage record of one model. */
forgetModel: (options: TouchModelOptions) => Promise<void>, 
/** Prune least recently used models until the target fits the cap. */
ensureSpace: (options: EnsureSpaceOptions) => Promise<void>;
export { DEFAULT_DISK_SPACE_MB, ensureSpace, touchModel, forgetModel, readUsage, };
