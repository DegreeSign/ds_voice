import type { CreatePoolOptions, TensorPool } from "./types";
/** Create a large shared memory, or null when unavailable. */
declare const createSharedMemory: () => WebAssembly.Memory | null, 
/** Browser pools start asynchronously, so the sync entry stays null. */
createPool: (_options: CreatePoolOptions) => TensorPool | null, 
/** Create a worker pool bound to one shared memory. */
createPoolAsync: (options: CreatePoolOptions) => Promise<TensorPool | null>;
export { createPool, createPoolAsync, createSharedMemory };
