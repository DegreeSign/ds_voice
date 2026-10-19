// Platform store selection.
import { detectPlatform } from "../speech/env";
import { Platform } from "../tensor/types";
import type { ModelStore } from "../tensor/types";
let store: Promise<ModelStore> | null = null;
const
    /** Build the store for the current runtime. */
    createStore = async (): Promise<ModelStore> => {
        if (detectPlatform() === Platform.Node) {
            const { createNodeStore } = await import("./node");
            return createNodeStore();
        }
        const { createBrowserStore } = await import("./browser");
        return createBrowserStore();
    },
    /** Shared store for the current runtime. */
    getStore = (): Promise<ModelStore> => {
        if (!store) {
            store = createStore();
        }
        return store;
    };

export { getStore };
