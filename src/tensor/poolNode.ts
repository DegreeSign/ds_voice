// Node worker pool over the shared kernel memory.
import { availableParallelism } from "node:os";
import { Worker } from "node:worker_threads";
import { KERNEL_SHARED_BASE64 } from "./kernelsWasm";
import type {
    CreatePoolOptions,
    PoolMatmulOptions,
    TensorPool,
} from "./types";
const
    /** Most worker threads used by one pool. */
    MAX_THREADS = 8,
    /** Control slots shared with the workers. */
    SLOT_READY = 0,
    SLOT_GENERATION = 1,
    SLOT_DONE = 2,
    SLOT_FAILED = 3,
    SLOT_INPUT = 4,
    SLOT_WEIGHT = 5,
    SLOT_BIAS = 6,
    SLOT_OUT = 7,
    SLOT_ROWS = 8,
    SLOT_IN_DIM = 9,
    SLOT_OUT_DIM = 10,
    SLOT_THREADS = 11,
    SLOT_COUNT = 16,
    /** Worker body, evaluated with the kernel bytes and shared memory. */
    WORKER_SOURCE = `
const { workerData } = require("node:worker_threads");
const { memory, control, index, threads, base64 } = workerData;
const binary = atob(base64);
const bytes = new Uint8Array(binary.length);
for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
}
const instance = new WebAssembly.Instance(new WebAssembly.Module(bytes), {
    env: { memory },
});
const matmul = instance.exports.matmul;
Atomics.add(control, 0, 1);
Atomics.notify(control, 0);
let last = 0;
for (; ;) {
    if (Atomics.load(control, 1) === last) {
        Atomics.wait(control, 1, last);
    }
    last = Atomics.load(control, 1);
    try {
        const rows = Atomics.load(control, 8);
        const body = Atomics.load(control, 11);
        const start = Math.floor((rows * index) / body);
        const end = Math.floor((rows * (index + 1)) / body);
        if (end > start) {
            const inDim = Atomics.load(control, 9);
            const outDim = Atomics.load(control, 10);
            matmul(
                Atomics.load(control, 4) + start * inDim * 4,
                Atomics.load(control, 5),
                Atomics.load(control, 6),
                Atomics.load(control, 7) + start * outDim * 4,
                end - start,
                inDim,
                outDim,
            );
        }
    } catch {
        Atomics.store(control, 3, 1);
    }
    Atomics.add(control, 2, 1);
    Atomics.notify(control, 2);
}
`;

/** Create a large shared memory, or null when unavailable. */
const createSharedMemory = (): WebAssembly.Memory | null => {
    try {
        return new WebAssembly.Memory({
            initial: 65536,
            maximum: 65536,
            shared: true,
        });
    } catch {
        return null;
    }
},
    /** Create a worker pool bound to one shared memory. */
    createPool = (options: CreatePoolOptions): TensorPool | null => {
        const
            threads = Math.max(
                1,
                Math.min(MAX_THREADS, availableParallelism()),
            ),
            buffer = options.memory.buffer;
        if (threads < 2 || !(buffer instanceof SharedArrayBuffer)) {
            return null;
        }
        const
            control = new Int32Array(
                new SharedArrayBuffer(SLOT_COUNT * 4),
            ),
            workers: Worker[] = [],
            close = (): void => {
                for (const worker of workers) {
                    void worker.terminate();
                }
                workers.length = 0;
            };
        try {
            for (let i = 0; i < threads; i += 1) {
                const worker = new Worker(WORKER_SOURCE, {
                    eval: true,
                    workerData: {
                        memory: options.memory,
                        control,
                        index: i,
                        threads,
                        base64: KERNEL_SHARED_BASE64,
                    },
                });
                worker.unref();
                workers.push(worker);
            }
            const deadline = Date.now() + 20_000;
            while (
                Atomics.load(control, SLOT_READY) < threads &&
                Date.now() < deadline
            ) {
                Atomics.wait(
                    control,
                    SLOT_READY,
                    Atomics.load(control, SLOT_READY),
                    100,
                );
            }
            if (Atomics.load(control, SLOT_READY) < threads) {
                close();
                return null;
            }
        } catch {
            close();
            return null;
        }
        return {
            threads,
            matmul: (request: PoolMatmulOptions): boolean => {
                Atomics.store(control, SLOT_INPUT, request.input);
                Atomics.store(control, SLOT_WEIGHT, request.weight);
                Atomics.store(control, SLOT_BIAS, request.bias);
                Atomics.store(control, SLOT_OUT, request.out);
                Atomics.store(control, SLOT_ROWS, request.rows);
                Atomics.store(control, SLOT_IN_DIM, request.inDim);
                Atomics.store(control, SLOT_OUT_DIM, request.outDim);
                Atomics.store(control, SLOT_THREADS, threads);
                Atomics.store(control, SLOT_FAILED, 0);
                Atomics.store(control, SLOT_DONE, 0);
                Atomics.add(control, SLOT_GENERATION, 1);
                Atomics.notify(control, SLOT_GENERATION, threads);
                const deadline = Date.now() + 120_000;
                while (
                    Atomics.load(control, SLOT_DONE) < threads &&
                    Date.now() < deadline
                ) {
                    Atomics.wait(
                        control,
                        SLOT_DONE,
                        Atomics.load(control, SLOT_DONE),
                        100,
                    );
                }
                if (
                    Atomics.load(control, SLOT_DONE) < threads ||
                    Atomics.load(control, SLOT_FAILED) === 1
                ) {
                    close();
                    return false;
                }
                return true;
            },
            close,
        };
    };

export { createPool, createSharedMemory };
