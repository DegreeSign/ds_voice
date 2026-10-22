import type { CreatePoolOptions, TensorPool } from "./types";
/** Create a large shared memory, or null when unavailable. */
declare const createSharedMemory: () => WebAssembly.Memory | null, 
/** Create a worker pool bound to one shared memory. */
createPool: (options: CreatePoolOptions) => TensorPool | null, 
/** Asynchronous form of createPool, Node starts synchronously. */
createPoolAsync: (options: CreatePoolOptions) => Promise<TensorPool | null>;
export { createPool, createPoolAsync, createSharedMemory };
