import type { CreatePoolOptions, TensorPool } from "./types";
declare const 
/** No shared memory outside Node. */
createSharedMemory: () => WebAssembly.Memory | null, 
/** No worker pool outside Node. */
createPool: (_options: CreatePoolOptions) => TensorPool | null, 
/** No worker pool outside Node. */
createPoolAsync: (_options: CreatePoolOptions) => Promise<TensorPool | null>;
export { createPool, createPoolAsync, createSharedMemory };
