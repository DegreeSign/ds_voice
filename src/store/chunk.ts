// Chunked binary storage helpers.
const
    /** Bytes held by one stored chunk. */
    CHUNK_BYTES = 4 * 1024 * 1024,
    /** Manifest key for one file. */
    manifestKey = (key: string): string => `${key}.parts`,
    /** Chunk key for one file and index. */
    partKey = (options: { key: string; index: number }): string => {
        const { key, index } = options;
        return `${key}.part${index}`;
    },
    /** Encode a list of chunk sizes. */
    encodeManifest = (sizes: number[]): string => JSON.stringify({ sizes }),
    /** Decode a list of chunk sizes. */
    decodeManifest = (text: string): number[] => {
        const parsed = JSON.parse(text) as { sizes?: number[] };
        return parsed.sizes ?? [];
    },
    /** Split a stream into fixed size chunks. */
    streamChunks = async (options: {
        stream: ReadableStream<Uint8Array>;
        onChunk: (chunk: Uint8Array) => Promise<void>;
    }): Promise<void> => {
        const reader = options.stream.getReader();
        let pending = new Uint8Array(0);
        for (;;) {
            const result = await reader.read();
            if (result.done) {
                break;
            }
            const merged = new Uint8Array(pending.length + result.value.length);
            merged.set(pending, 0);
            merged.set(result.value, pending.length);
            pending = merged;
            while (pending.length >= CHUNK_BYTES) {
                await options.onChunk(pending.subarray(0, CHUNK_BYTES));
                pending = pending.slice(CHUNK_BYTES);
            }
        }
        if (pending.length > 0) {
            await options.onChunk(pending);
        }
    },
    /** True when a key or its manifest is present. */
    chunkedExists = async (options: {
        key: string;
        read: (key: string) => Promise<string | undefined>;
    }): Promise<boolean> => {
        const { key, read } = options;
        if ((await read(manifestKey(key))) !== undefined) {
            return true;
        }
        return (await read(key)) !== undefined;
    },
    /** True when a key has the manifest of a chunked binary. */
    chunkedReady = async (options: {
        key: string;
        read: (key: string) => Promise<string | undefined>;
    }): Promise<boolean> => {
        const { key, read } = options;
        return (await read(manifestKey(key))) !== undefined;
    },
    /** Size of a chunked file in bytes, without reading it. */
    chunkedSize = async (options: {
        key: string;
        read: (key: string) => Promise<string | undefined>;
    }): Promise<number> => {
        const
            { key, read } = options,
            manifest = await read(manifestKey(key));
        if (manifest !== undefined) {
            let total = 0;
            for (const size of decodeManifest(manifest)) {
                total += size;
            }
            return total;
        }
        const text = await read(key);
        return text === undefined
            ? 0
            : new TextEncoder().encode(text).length;
    },
    /** Assemble a file from its manifest chunks. */
    readChunks = async (options: {
        key: string;
        read: (key: string) => Promise<string | undefined>;
        decode: (text: string) => Uint8Array;
    }): Promise<Uint8Array | null> => {
        const
            { key, read, decode } = options,
            manifest = await read(manifestKey(key));
        if (manifest === undefined) {
            return null;
        }
        const sizes = decodeManifest(manifest);
        let total = 0;
        for (const size of sizes) {
            total += size;
        }
        const out = new Uint8Array(total);
        let offset = 0;
        for (let i = 0; i < sizes.length; i += 1) {
            const text = await read(partKey({ key, index: i }));
            if (text === undefined) {
                throw new Error(`missing chunk ${key} ${i}`);
            }
            const chunk = decode(text);
            out.set(chunk, offset);
            offset += chunk.length;
        }
        return out;
    },
    /** Write a stream as a manifest plus chunks. */
    writeChunkStream = async (options: {
        key: string;
        stream: ReadableStream<Uint8Array>;
        onProgress?: (loaded: number) => void;
        encode: (chunk: Uint8Array) => string;
        has?: (key: string) => Promise<boolean>;
        write: (key: string, text: string) => Promise<void>;
        remove: (key: string) => Promise<void>;
    }): Promise<void> => {
        const
            { key, stream, onProgress, encode, write } = options,
            sizes: number[] = [];
        // drop manifest, write in progress
        await options.remove(manifestKey(key));
        let index = 0;
        await streamChunks({
            stream,
            onChunk: async (chunk) => {
                await write(partKey({ key, index }), encode(chunk));
                sizes.push(chunk.length);
                index += 1;
                onProgress?.(chunk.length);
            },
        });
        await write(manifestKey(key), encodeManifest(sizes));
        // clear leftover tail chunks
        const has = options.has;
        if (has) {
            for (; await has(partKey({ key, index })); index += 1) {
                await options.remove(partKey({ key, index }));
            }
        }
    },
    /** Remove a key, its manifest and every chunk. */
    removeChunks = async (options: {
        key: string;
        read: (key: string) => Promise<string | undefined>;
        has?: (key: string) => Promise<boolean>;
        remove: (key: string) => Promise<void>;
    }): Promise<void> => {
        const
            { key, read, remove } = options,
            has = options.has ??
                (async (inner: string) => (await read(inner)) !== undefined),
            manifest = await read(manifestKey(key));
        let index = 0;
        if (manifest !== undefined) {
            const sizes = decodeManifest(manifest);
            for (; index < sizes.length; index += 1) {
                await remove(partKey({ key, index }));
            }
        } else {
            // sweep orphan chunks
            for (; await has(partKey({ key, index })); index += 1) {
                await remove(partKey({ key, index }));
            }
        }
        await remove(manifestKey(key));
        await remove(key);
    };

export {
    chunkedExists,
    chunkedReady,
    chunkedSize,
    readChunks,
    writeChunkStream,
    removeChunks,
};
