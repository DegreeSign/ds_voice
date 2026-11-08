// Platform store selection.
import { detectPlatform } from "../../speech/session/env";
import { Platform } from "../../tensor/core/types";
import type { ModelStore } from "../../tensor/core/types";
let store: Promise<ModelStore> | null = null;
const
    /** Build the store for the current runtime. */
    createStore = async (): Promise<ModelStore> => {
        if (detectPlatform() === Platform.Node) {
            const { createNodeStore } = await import("../node/node");
            return createNodeStore();
        }
        const { createBrowserStore } = await import("../browser/browser");
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
