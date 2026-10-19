// Browser replacement for the Node store.
import type { ModelStore, NodeStoreOptions } from "../tensor/types";
const
    /** Node store is unavailable in the browser bundle. */
    createNodeStore = (options: NodeStoreOptions = {}): ModelStore => {
        throw new Error(`node store unavailable ${options.cacheDir ?? ""}`);
    };

export { createNodeStore };
