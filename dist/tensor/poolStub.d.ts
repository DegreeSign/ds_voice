import type { CreatePoolOptions, TensorPool } from "./types";
declare const 
/** No shared memory outside Node. */
createSharedMemory: () => WebAssembly.Memory | null, 
/** No worker pool outside Node. */
createPool: (_options: CreatePoolOptions) => TensorPool | null;
export { createPool, createSharedMemory };
