// SIMD kernel runtime with a growable linear memory arena.
import { KERNEL_BASE64, KERNEL_SHARED_BASE64 } from "./kernelsWasm";
import { createPool, createPoolAsync, createSharedMemory } from "./poolNode";
import type {
    AttentionOptions,
    Conv1dOptions,
    Conv2dOptions,
    LinearOptions,
    PointwiseOptions,
    RelativeBiasOptions,
    SiluOptions,
    WasmKernel,
} from "./types";
let threadLimit = 1;
const
    /** Score rows kept in the attention scratch area. */
    ATTENTION_BLOCK = 16,
    /** Page size of the WebAssembly memory. */
    PAGE = 65536,
    /** Least work before splitting a matmul across threads. */
    MIN_POOL_WORK = 4_000_000,
    /** Fewer rows than this never split across threads. */
    MIN_POOL_ROWS = 8,
    /** Decode the embedded kernel bytes. */
    decode = (text: string) => {
        const
            binary = atob(text),
            out = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i += 1) {
            out[i] = binary.charCodeAt(i);
        }
        return out;
    },
    /** Create the kernel once, or null when SIMD is unavailable. */
    create = (): WasmKernel | null => {
        try {
            const
                sharedMemory = threadLimit >= 2
                    ? createSharedMemory()
                    : null,
                memory = sharedMemory ?? new WebAssembly.Memory({
                    initial: 64,
                    maximum: 65536,
                }),
                bytes = sharedMemory
                    ? KERNEL_SHARED_BASE64
                    : KERNEL_BASE64,
                module = new WebAssembly.Module(decode(bytes)),
                instance = new WebAssembly.Instance(module, {
                    env: { memory },
                }),
                exports = instance.exports as unknown as {
                    matmul: WasmKernel["matmul"];
                    attention: WasmKernel["attention"];
                    conv1d: WasmKernel["conv1d"];
                    conv2d: WasmKernel["conv2d"];
                    pointwise: WasmKernel["pointwise"];
                    silu: WasmKernel["silu"];
                    gelu: WasmKernel["gelu"];
                    relativeBias: WasmKernel["relativeBias"];
                    arenaBase: () => number;
                };
            return {
                memory,
                matmul: exports.matmul,
                attention: exports.attention,
                conv1d: exports.conv1d,
                conv2d: exports.conv2d,
                pointwise: exports.pointwise,
                silu: exports.silu,
                gelu: exports.gelu,
                relativeBias: exports.relativeBias,
                next: Math.ceil((exports.arenaBase() + 1) / 16) * 16,
                tensors: new WeakMap(),
                free: new Map(),
                input: 0,
                inputSize: 0,
                output: 0,
                outputSize: 0,
                scores: 0,
                scoresSize: 0,
                bias: 0,
                biasSize: 0,
                bd: 0,
                bdSize: 0,
                shared: sharedMemory !== null,
                pool: undefined,
                poolTried: false,
            };
        } catch {
            return null;
        }
    };

let kernelState: WasmKernel | null | undefined;

/** Weight arrays whose arena region was already freed. */
const released = new WeakSet<Float32Array>();

const
    /** Default worker count. */
    DEFAULT_THREADS = 4,
    /** Request the worker count used by the next pool. */
    setThreadCount = (threads?: number): void => {
        const value = Math.max(1, Math.floor(threads ?? DEFAULT_THREADS));
        if (value === threadLimit) {
            return;
        }
        threadLimit = value;
        if (value >= 2 && kernelState && !kernelState.shared) {
            kernelState = undefined;
        }
        const pool = kernelState?.pool;
        if (pool && pool.threads !== value) {
            pool.close();
            if (kernelState) {
                kernelState.pool = null;
                kernelState.poolTried = false;
            }
        }
    },
    /** Worker threads used by the current decode. */
    kernelThreads = (): number => kernelState?.pool?.threads ?? 1,
    /** Create the shared kernel and worker pool before a decode. */
    prepareThreads = async (threads?: number): Promise<void> => {
        setThreadCount(threads);
        const kernel = getKernel();
        if (!kernel || threadLimit < 2 || !kernel.shared || kernel.pool) {
            return;
        }
        kernel.poolTried = true;
        kernel.pool = await createPoolAsync({
            memory: kernel.memory,
            threads: threadLimit,
        });
    },
    /** Shared kernel instance, null when the engine lacks SIMD. */
    getKernel = (): WasmKernel | null => {
        if (kernelState === undefined) {
            kernelState = create();
        }
        return kernelState;
    },
    /** Grow the arena and return an aligned offset. */
    allocate = (kernel: WasmKernel, bytes: number): number => {
        const size = Math.ceil(bytes / 16) * 16;
        const spare = kernel.free.get(size);
        if (spare && spare.length > 0) {
            const offset = spare.pop() as number;
            if (spare.length === 0) {
                kernel.free.delete(size);
            }
            return offset;
        }
        const
            aligned = kernel.next % 16 === 0
                ? kernel.next
                : kernel.next + (16 - (kernel.next % 16)),
            start = aligned,
            end = start + size;
        if (end > kernel.memory.buffer.byteLength) {
            kernel.memory.grow(
                Math.ceil((end - kernel.memory.buffer.byteLength) / PAGE),
            );
        }
        kernel.next = end;
        return start;
    },
    /** Return one arena region to the free list. */
    release = (kernel: WasmKernel, offset: number, bytes: number): void => {
        const
            size = Math.ceil(bytes / 16) * 16,
            spare = kernel.free.get(size);
        if (spare) {
            spare.push(offset);
            return;
        }
        kernel.free.set(size, [offset]);
    },
    /** Drop the arena copy of one weight array. */
    releaseTensor = (value: Float32Array): void => {
        const kernel = kernelState;
        if (!kernel || released.has(value)) {
            return;
        }
        released.add(value);
        const mapped = kernel.tensors.get(value);
        if (mapped !== undefined) {
            kernel.tensors.delete(value);
            release(kernel, mapped, value.byteLength);
            return;
        }
        if (value.buffer !== kernel.memory.buffer) {
            return;
        }
        release(kernel, value.byteOffset, value.byteLength);
    },
    /** Copy a tensor into the arena once. */
    tensorOffset = (kernel: WasmKernel, value: Float32Array): number => {
        if (value.buffer === kernel.memory.buffer) {
            return value.byteOffset;
        }
        const hit = kernel.tensors.get(value);
        if (hit !== undefined) {
            return hit;
        }
        const offset = allocate(kernel, value.byteLength);
        new Uint8Array(kernel.memory.buffer).set(
            new Uint8Array(value.buffer, value.byteOffset, value.byteLength),
            offset,
        );
        kernel.tensors.set(value, offset);
        return offset;
    },
    /** Copy a byte tensor into the arena once. */
    bytesOffset = (kernel: WasmKernel, value: Uint8Array): number => {
        const offset = allocate(kernel, value.byteLength);
        new Uint8Array(kernel.memory.buffer).set(value, offset);
        return offset;
    },
    /** Copy transient floats into the arena. */
    copyFloats = (
        kernel: WasmKernel,
        offset: number,
        value: Float32Array,
        length: number,
    ): void => {
        new Uint8Array(kernel.memory.buffer).set(
            new Uint8Array(value.buffer, value.byteOffset, length * 4),
            offset,
        );
    },
    /** Reserve a reusable scratch area. */
    scratch = (
        kernel: WasmKernel,
        which: "input" | "output" | "scores" | "bias" | "bd",
        floats: number,
    ): number => {
        const
            sizeKey = `${which}Size` as const,
            offsetKey = which,
            current = kernel[sizeKey];
        if (current >= floats && kernel[offsetKey] !== 0) {
            return kernel[offsetKey];
        }
        const
            offset = allocate(kernel, floats * 4);
        kernel[sizeKey] = floats;
        kernel[offsetKey] = offset;
        return offset;
    },
    /** Reserve an f32 arena region the weights are decoded into. */
    reserveFloats = (floats: number): Float32Array | null => {
        const kernel = getKernel();
        if (!kernel || !kernel.shared || floats <= 0) {
            return null;
        }
        try {
            const offset = allocate(kernel, floats * 4);
            return new Float32Array(kernel.memory.buffer, offset, floats);
        } catch {
            return null;
        }
    },
    /** Lazily create the worker pool for a shared kernel arena. */
    poolFor = (kernel: WasmKernel): WasmKernel["pool"] => {
        if (threadLimit < 2 || !kernel.shared || kernel.poolTried) {
            return kernel.pool;
        }
        kernel.poolTried = true;
        kernel.pool = createPool({
            memory: kernel.memory,
            threads: threadLimit,
        });
        return kernel.pool;
    },
    /** Run a linear layer with the SIMD kernel, or null to fall back. */
    wasmLinear = (options: LinearOptions): Float32Array | null => {
        const
            kernel = getKernel(),
            { input, weight, bias, rows, inDim, outDim } = options;
        if (!kernel || rows <= 0 || inDim <= 0 || outDim <= 0) {
            return null;
        }
        if (
            weight.length < inDim * outDim ||
            input.length < rows * inDim
        ) {
            return null;
        }
        try {
            const
                weightOffset = tensorOffset(kernel, weight),
                biasOffset = bias ? tensorOffset(kernel, bias) : 0,
                inputOffset = scratch(kernel, "input", rows * inDim),
                outputOffset = scratch(kernel, "output", rows * outDim),
                out = options.out ?? new Float32Array(rows * outDim);
            copyFloats(kernel, inputOffset, input, rows * inDim);
            const
                work = rows * inDim * outDim,
                pool = work >= MIN_POOL_WORK && rows >= MIN_POOL_ROWS
                    ? poolFor(kernel)
                    : null,
                split = pool
                    ? pool.matmul({
                        input: inputOffset,
                        weight: weightOffset,
                        bias: biasOffset,
                        out: outputOffset,
                        rows,
                        inDim,
                        outDim,
                    })
                    : false;
            if (!split) {
                if (pool) {
                    kernel.pool = null;
                }
                kernel.matmul(
                    inputOffset,
                    weightOffset,
                    biasOffset,
                    outputOffset,
                    rows,
                    inDim,
                    outDim,
                );
            }
            out.set(new Float32Array(
                kernel.memory.buffer,
                outputOffset,
                rows * outDim,
            ));
            return out;
        } catch {
            return null;
        }
    },
    /** Run multi-head attention with the SIMD kernel, or null to fall back. */
    wasmAttention = (options: AttentionOptions): Float32Array | null => {
        const
            kernel = getKernel(),
            { q, k, v, heads, headDim, bias } = options,
            { seqQ, seqK } = options;
        if (!kernel || seqQ <= 0 || seqK <= 0 || heads <= 0 || headDim <= 0) {
            return null;
        }
        const dim = heads * headDim;
        if (
            q.length < seqQ * dim ||
            k.length < seqK * dim ||
            v.length < seqK * dim
        ) {
            return null;
        }
        try {
            const
                qOffset = scratch(kernel, "input", seqQ * dim + seqK * dim * 2),
                kOffset = qOffset + seqQ * dim * 4,
                vOffset = kOffset + seqK * dim * 4,
                outputOffset = scratch(kernel, "output", seqQ * dim),
                scoresOffset = scratch(
                    kernel,
                    "scores",
                    ATTENTION_BLOCK * seqK,
                ),
                out = options.out ?? new Float32Array(seqQ * dim),
                scale = options.scale ?? 1 / Math.sqrt(headDim);
            copyFloats(kernel, qOffset, q, seqQ * dim);
            copyFloats(kernel, kOffset, k, seqK * dim);
            copyFloats(kernel, vOffset, v, seqK * dim);
            let biasOffset = 0;
            if (bias) {
                biasOffset = scratch(kernel, "bias", heads * seqQ * seqK);
                copyFloats(kernel, biasOffset, bias, heads * seqQ * seqK);
            }
            kernel.attention(
                qOffset,
                kOffset,
                vOffset,
                biasOffset,
                outputOffset,
                scoresOffset,
                seqQ,
                seqK,
                heads,
                headDim,
                scale,
                options.causal ? 1 : 0,
            );
            out.set(new Float32Array(
                kernel.memory.buffer,
                outputOffset,
                seqQ * dim,
            ));
            return out;
        } catch {
            return null;
        }
    },
    /** Run a 1d convolution with the SIMD kernel, or null to fall back. */
    wasmConv1d = (
        options: Conv1dOptions,
    ): { data: Float32Array; length: number } | null => {
        const
            kernel = getKernel(),
            { input, weight, bias } = options,
            { inChannels, outChannels, inLength, kernel: k } = options,
            { stride, padding } = options,
            length = Math.floor(
                (inLength + 2 * padding - k) / stride,
            ) + 1;
        if (
            !kernel ||
            length <= 0 ||
            weight.length < outChannels * inChannels * k ||
            input.length < inChannels * inLength
        ) {
            return null;
        }
        try {
            const
                inputOffset = scratch(kernel, "input", inChannels * inLength),
                weightOffset = tensorOffset(kernel, weight),
                biasOffset = tensorOffset(kernel, bias),
                outputOffset = scratch(kernel, "output", outChannels * length),
                data = new Float32Array(outChannels * length);
            copyFloats(kernel, inputOffset, input, inChannels * inLength);
            kernel.conv1d(
                inputOffset,
                weightOffset,
                biasOffset,
                outputOffset,
                inChannels,
                outChannels,
                inLength,
                k,
                stride,
                padding,
            );
            data.set(new Float32Array(
                kernel.memory.buffer,
                outputOffset,
                outChannels * length,
            ));
            return { data, length };
        } catch {
            return null;
        }
    },
    /** Run a 2d convolution with the SIMD kernel, or null to fall back. */
    wasmConv2d = (
        options: Conv2dOptions,
    ): { data: Float32Array; height: number; width: number } | null => {
        const
            kernel = getKernel(),
            { input, weight, bias } = options,
            { inChannels, inHeight, inWidth, outChannels, kernel: k } = options,
            { stride, groups } = options;
        if (!kernel) {
            return null;
        }
        const
            pad = (k - 1) / 2,
            height = Math.floor((inHeight + 2 * pad - k) / stride) + 1,
            width = Math.floor((inWidth + 2 * pad - k) / stride) + 1;
        if (
            height <= 0 ||
            width <= 0 ||
            weight.length < outChannels * (inChannels / groups) * k * k ||
            input.length < inChannels * inHeight * inWidth
        ) {
            return null;
        }
        try {
            const
                inputOffset = scratch(kernel, "input", inChannels * inHeight * inWidth),
                weightOffset = tensorOffset(kernel, weight),
                biasOffset = bias ? tensorOffset(kernel, bias) : 0,
                outputOffset = scratch(kernel, "output", outChannels * height * width),
                data = new Float32Array(outChannels * height * width);
            copyFloats(kernel, inputOffset, input, inChannels * inHeight * inWidth);
            kernel.conv2d(
                inputOffset,
                weightOffset,
                biasOffset,
                outputOffset,
                inChannels,
                inHeight,
                inWidth,
                outChannels,
                k,
                stride,
                groups,
            );
            data.set(new Float32Array(
                kernel.memory.buffer,
                outputOffset,
                outChannels * height * width,
            ));
            return { data, height, width };
        } catch {
            return null;
        }
    },
    /** Apply GELU in place with the SIMD kernel. */
    wasmGelu = (options: SiluOptions): boolean => {
        const
            kernel = getKernel(),
            { values } = options,
            offset = options.offset ?? 0,
            length = options.length ?? values.length - offset;
        if (!kernel || length <= 0) {
            return false;
        }
        try {
            const
                source = values.subarray(offset, offset + length),
                pointer = scratch(kernel, "input", length);
            copyFloats(kernel, pointer, source, length);
            kernel.gelu(pointer, length);
            values.set(
                new Float32Array(kernel.memory.buffer, pointer, length),
                offset,
            );
            return true;
        } catch {
            return false;
        }
    },
    /** Apply SiLU in place with the SIMD kernel. */
    wasmSilu = (options: SiluOptions): boolean => {
        const
            kernel = getKernel(),
            { values } = options,
            offset = options.offset ?? 0,
            length = options.length ?? values.length - offset;
        if (!kernel || length <= 0) {
            return false;
        }
        try {
            const
                source = values.subarray(offset, offset + length),
                pointer = scratch(kernel, "input", length);
            copyFloats(kernel, pointer, source, length);
            kernel.silu(pointer, length);
            values.set(
                new Float32Array(kernel.memory.buffer, pointer, length),
                offset,
            );
            return true;
        } catch {
            return false;
        }
    },
    /** Run a 1x1 stride one convolution with the SIMD kernel. */
    wasmPointwise = (
        options: PointwiseOptions,
    ): Float32Array | null => {
        const
            kernel = getKernel(),
            { input, weight, bias, channels, outChannels, spatial } = options;
        if (
            !kernel ||
            channels <= 0 ||
            outChannels <= 0 ||
            spatial <= 0 ||
            input.length < channels * spatial ||
            weight.length < outChannels * channels
        ) {
            return null;
        }
        try {
            const
                inputOffset = scratch(
                    kernel,
                    "input",
                    channels * spatial + spatial * channels,
                ),
                tmpIn = inputOffset + channels * spatial * 4,
                outputOffset = scratch(
                    kernel,
                    "output",
                    outChannels * spatial + spatial * outChannels,
                ),
                tmpOut = outputOffset + outChannels * spatial * 4,
                weightOffset = tensorOffset(kernel, weight),
                biasOffset = bias ? tensorOffset(kernel, bias) : 0,
                data = new Float32Array(outChannels * spatial);
            copyFloats(kernel, inputOffset, input, channels * spatial);
            kernel.pointwise(
                inputOffset,
                weightOffset,
                biasOffset,
                outputOffset,
                tmpIn,
                tmpOut,
                channels,
                outChannels,
                spatial,
            );
            data.set(new Float32Array(
                kernel.memory.buffer,
                outputOffset,
                outChannels * spatial,
            ));
            return data;
        } catch {
            return null;
        }
    },
    /** Build the relative position bias with the SIMD kernel. */
    wasmRelativeBias = (options: RelativeBiasOptions): Float32Array | null => {
        const
            kernel = getKernel(),
            { q, relK, seq, heads, headDim, scale } = options,
            hidden = heads * headDim,
            p = 2 * seq - 1;
        if (!kernel || seq <= 0 || heads <= 0 || headDim <= 0) {
            return null;
        }
        if (q.length < seq * hidden || relK.length < p * hidden) {
            return null;
        }
        try {
            const
                qOffset = scratch(kernel, "input", seq * hidden + p * hidden),
                relOffset = qOffset + seq * hidden * 4,
                outputOffset = scratch(kernel, "output", heads * seq * seq),
                bdOffset = scratch(kernel, "bd", seq * p),
                out = new Float32Array(heads * seq * seq);
            copyFloats(kernel, qOffset, q, seq * hidden);
            copyFloats(kernel, relOffset, relK, p * hidden);
            kernel.relativeBias(
                qOffset,
                relOffset,
                outputOffset,
                bdOffset,
                seq,
                heads,
                headDim,
                scale,
            );
            out.set(new Float32Array(
                kernel.memory.buffer,
                outputOffset,
                heads * seq * seq,
            ));
            return out;
        } catch {
            return null;
        }
    };

export {
    getKernel,
    kernelThreads,
    prepareThreads,
    releaseTensor,
    reserveFloats,
    setThreadCount,
    wasmAttention,
    wasmConv1d,
    wasmConv2d,
    wasmGelu,
    wasmLinear,
    wasmPointwise,
    wasmRelativeBias,
    wasmSilu,
};
