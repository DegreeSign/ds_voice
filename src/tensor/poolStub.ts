// Browser stub, no worker pool.
import type { CreatePoolOptions, TensorPool } from "./types";
const
    /** No shared memory outside Node. */
    createSharedMemory = (): WebAssembly.Memory | null => null,
    /** No worker pool outside Node. */
    createPool = (_options: CreatePoolOptions): TensorPool | null => null,
    /** No worker pool outside Node. */
    createPoolAsync = async (
        _options: CreatePoolOptions,
    ): Promise<TensorPool | null> => null;

export { createPool, createPoolAsync, createSharedMemory };
