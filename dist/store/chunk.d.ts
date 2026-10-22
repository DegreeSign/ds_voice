import type { ModelPartStore } from "../tensor/types";
declare const 
/** Bytes held by one stored chunk. */
CHUNK_BYTES: number, 
/** Manifest key for one file. */
manifestKey: (key: string) => string, 
/** Chunk key for one file and index. */
partKey: (options: {
    key: string;
    index: number;
}) => string, 
/** Encode a list of chunk sizes. */
encodeManifest: (sizes: number[]) => string, 
/** Decode a list of chunk sizes. */
decodeManifest: (text: string) => number[], 
/** True when every chunk of a manifest is present. */
completeSizes: (sizes: number[]) => boolean, 
/** Assemble a file from its stored chunks. */
readPartFile: (options: {
    parts: ModelPartStore;
    key: string;
}) => Promise<Uint8Array | null>, 
/** Size of a chunked file in bytes. */
sizeOfPartFile: (options: {
    parts: ModelPartStore;
    key: string;
}) => Promise<number>, 
/** True when a manifest exists and every chunk is present. */
hasCompletePartFile: (options: {
    parts: ModelPartStore;
    key: string;
}) => Promise<boolean>, 
/** Write a stream as a manifest plus chunks. */
streamPartFile: (options: {
    parts: ModelPartStore;
    key: string;
    stream: ReadableStream<Uint8Array>;
    onProgress?: (loaded: number) => void;
}) => Promise<void>, 
/** Remove every chunk of a file, leaving the manifest. */
removePartFile: (options: {
    parts: ModelPartStore;
    key: string;
}) => Promise<void>, 
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
export { CHUNK_BYTES, chunkedExists, chunkedReady, chunkedSize, completeSizes, decodeManifest, encodeManifest, hasCompletePartFile, manifestKey, partKey, readChunks, readPartFile, removeChunks, removePartFile, sizeOfPartFile, streamPartFile, writeChunkStream, };
