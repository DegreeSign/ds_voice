// Browser worker pool over the shared kernel memory.
import { KERNEL_SHARED_BASE64 } from "./kernelsWasm";
import type {
    CreatePoolOptions,
    PoolMatmulOptions,
    TensorPool,
} from "./types";
const
    /** Most worker threads used by one pool. */
    MAX_THREADS = 8,
    /** How long to wait for the workers to start. */
    START_TIMEOUT_MS = 20_000,
    /** How long one split matmul may run. */
    JOB_TIMEOUT_MS = 120_000,
    /** Delay between asynchronous readiness polls. */
    POLL_MS = 20,
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
    /** Worker body, built from the kernel bytes and shared memory. */
    WORKER_SOURCE = `
let control = null;
let index = 0;
let threads = 1;
self.onmessage = (event) => {
    const init = event.data;
    control = init.control;
    index = init.index;
    threads = init.threads;
    const binary = atob(init.base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) {
        bytes[i] = binary.charCodeAt(i);
    }
    let matmul = null;
    try {
        const instance = new WebAssembly.Instance(
            new WebAssembly.Module(bytes),
            { env: { memory: init.memory } },
        );
        matmul = instance.exports.matmul;
    } catch {
        Atomics.store(control, 3, 1);
        Atomics.add(control, 0, 1);
        Atomics.notify(control, 0);
        return;
    }
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
};
`,
    /** True when this thread may block on Atomics. */
    canBlock = (): boolean => {
        if (typeof SharedArrayBuffer === "undefined") {
            return false;
        }
        try {
            const scratch = new Int32Array(new SharedArrayBuffer(4));
            Atomics.wait(scratch, 0, 1, 0);
            return true;
        } catch {
            return false;
        }
    },
    /** True when shared memory is allowed in this context. */
    isolation = (): boolean => {
        const scope = globalThis as {
            crossOriginIsolated?: boolean;
        };
        return scope.crossOriginIsolated === true;
    },
    /** Wait on a control slot, false when the target is late. */
    waitOn = (options: {
        control: Int32Array;
        slot: number;
        target: number;
        timeout: number;
    }): boolean => {
        const { control, slot, target, timeout } = options;
        const deadline = Date.now() + timeout;
        while (Atomics.load(control, slot) < target) {
            if (Date.now() >= deadline) {
                return false;
            }
            try {
                Atomics.wait(
                    control,
                    slot,
                    Atomics.load(control, slot),
                    100,
                );
            } catch {
                return false;
            }
        }
        return true;
    },
    /** Poll a control slot without blocking the event loop. */
    waitAsync = async (options: {
        control: Int32Array;
        slot: number;
        target: number;
        timeout: number;
    }): Promise<boolean> => {
        const { control, slot, target, timeout } = options;
        const deadline = Date.now() + timeout;
        while (Atomics.load(control, slot) < target) {
            if (Date.now() >= deadline) {
                return false;
            }
            await new Promise((resolve) => {
                setTimeout(resolve, POLL_MS);
            });
        }
        return true;
    };

/** Create a large shared memory, or null when unavailable. */
const createSharedMemory = (): WebAssembly.Memory | null => {
    if (
        typeof SharedArrayBuffer === "undefined" ||
        !isolation()
    ) {
        return null;
    }
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
    /** Browser pools start asynchronously, so the sync entry stays null. */
    createPool = (_options: CreatePoolOptions): TensorPool | null => null,
    /** Create a worker pool bound to one shared memory. */
    createPoolAsync = async (
        options: CreatePoolOptions,
    ): Promise<TensorPool | null> => {
        if (
            typeof SharedArrayBuffer === "undefined" ||
            typeof Worker === "undefined" ||
            typeof Blob === "undefined" ||
            typeof URL === "undefined" ||
            !canBlock()
        ) {
            return null;
        }
        const
            cores = typeof navigator === "undefined"
                ? 1
                : navigator.hardwareConcurrency || 1,
            threads = Math.max(
                1,
                Math.min(MAX_THREADS, options.threads ?? cores),
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
            urls: string[] = [],
            close = (): void => {
                for (const worker of workers) {
                    worker.terminate();
                }
                workers.length = 0;
                for (const url of urls) {
                    URL.revokeObjectURL(url);
                }
                urls.length = 0;
            };
        try {
            const source = URL.createObjectURL(
                new Blob([WORKER_SOURCE], { type: "text/javascript" }),
            );
            urls.push(source);
            for (let index = 0; index < threads; index += 1) {
                const worker = new Worker(source);
                worker.postMessage({
                    memory: options.memory,
                    control,
                    index,
                    threads,
                    base64: KERNEL_SHARED_BASE64,
                });
                workers.push(worker);
            }
            if (!await waitAsync({
                control,
                slot: SLOT_READY,
                target: threads,
                timeout: START_TIMEOUT_MS,
            })) {
                close();
                return null;
            }
            if (Atomics.load(control, SLOT_FAILED) === 1) {
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
                if (!waitOn({
                    control,
                    slot: SLOT_DONE,
                    target: threads,
                    timeout: JOB_TIMEOUT_MS,
                })) {
                    close();
                    return false;
                }
                if (Atomics.load(control, SLOT_FAILED) === 1) {
                    close();
                    return false;
                }
                return true;
            },
            close,
        };
    };

export { createPool, createPoolAsync, createSharedMemory };
