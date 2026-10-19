// Browser stub, no worker pool.
import type { CreatePoolOptions, TensorPool } from "./types";
const
    /** No shared memory outside Node. */
    createSharedMemory = (): WebAssembly.Memory | null => null,
    /** No worker pool outside Node. */
    createPool = (_options: CreatePoolOptions): TensorPool | null => null;

export { createPool, createSharedMemory };
