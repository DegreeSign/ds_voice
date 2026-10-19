import type { ModelStore } from "../tensor/types";
declare const 
/** Shared store for the current runtime. */
getStore: () => Promise<ModelStore>;
export { getStore };
