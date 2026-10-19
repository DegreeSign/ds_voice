declare const 
/** True when a key or its manifest is present. */
chunkedExists: (options: {
    key: string;
    read: (key: string) => Promise<string | undefined>;
}) => Promise<boolean>, 
/** True when a key has the manifest of a chunked binary. */
chunkedReady: (options: {
    key: string;
    read: (key: string) => Promise<string | undefined>;
}) => Promise<boolean>, 
/** Size of a chunked file in bytes, without reading it. */
chunkedSize: (options: {
    key: string;
    read: (key: string) => Promise<string | undefined>;
}) => Promise<number>, 
/** Assemble a file from its manifest chunks. */
readChunks: (options: {
    key: string;
    read: (key: string) => Promise<string | undefined>;
    decode: (text: string) => Uint8Array;
}) => Promise<Uint8Array | null>, 
/** Write a stream as a manifest plus chunks. */
writeChunkStream: (options: {
    key: string;
    stream: ReadableStream<Uint8Array>;
    onProgress?: (loaded: number) => void;
    encode: (chunk: Uint8Array) => string;
    has?: (key: string) => Promise<boolean>;
    write: (key: string, text: string) => Promise<void>;
    remove: (key: string) => Promise<void>;
}) => Promise<void>, 
/** Remove a key, its manifest and every chunk. */
removeChunks: (options: {
    key: string;
    read: (key: string) => Promise<string | undefined>;
    has?: (key: string) => Promise<boolean>;
    remove: (key: string) => Promise<void>;
}) => Promise<void>;
export { chunkedExists, chunkedReady, chunkedSize, readChunks, writeChunkStream, removeChunks, };
