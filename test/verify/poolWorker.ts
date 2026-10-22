// Worker bundle that exercises the browser thread pool.
import {
    getKernel,
    prepareThreads,
    wasmLinear,
} from "../../src/tensor/wasm";
const
    /** Run one pooled matmul and report the pool size. */
    run = async (): Promise<{
        threads: number;
        value: number;
        shared: boolean;
    }> => {
        await prepareThreads(2);
        const
            kernel = getKernel(),
            rows = 200,
            inDim = 200,
            outDim = 100,
            input = new Float32Array(rows * inDim).fill(1),
            weight = new Float32Array(outDim * inDim);
        for (let out = 0; out < outDim; out += 1) {
            weight[out * inDim + out] = 1;
        }
        const output = wasmLinear({ input, weight, rows, inDim, outDim });
        return {
            threads: kernel?.pool?.threads ?? 0,
            value: output?.[0] ?? 0,
            shared: kernel?.shared ?? false,
        };
    };

self.onmessage = (): void => {
    void run().then(
        (result) => {
            self.postMessage({ ok: true, ...result });
        },
        (error: unknown) => {
            self.postMessage({
                ok: false,
                error: error instanceof Error ? error.message : String(error),
            });
        },
    );
};
