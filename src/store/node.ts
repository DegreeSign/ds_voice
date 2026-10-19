// Node model store.
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { delFile, red, safeFolder, wrt } from "@degreesign/cache";
import {
    chunkedExists,
    chunkedReady,
    chunkedSize,
    readChunks,
    removeChunks,
    writeChunkStream,
} from "./chunk";
import type { ModelStore, NodeStoreOptions } from "../tensor/types";
const
    /** Default model cache folder. */
    DEFAULT_CACHE_DIR = `${homedir()}/.cache/decoder`,
    /** Encode bytes as base64. */
    encodeBase64 = (bytes: Uint8Array): string => {
        const view = Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength);
        return view.toString("base64");
    },
    /** Decode base64 to bytes. */
    decodeBase64 = (text: string): Uint8Array => {
        const buffer = Buffer.from(text, "base64");
        return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    },
    /** Create a filesystem model store. */
    createNodeStore = (options: NodeStoreOptions = {}): ModelStore => {
        const cacheDir = options.cacheDir ?? DEFAULT_CACHE_DIR;
        safeFolder(cacheDir);
        const
            absolute = (key: string): string => `${cacheDir}/${key}`,
            readStored = async (key: string): Promise<string | undefined> =>
                red(absolute(key), true),
            ensureFolder = (key: string): void => {
                const index = key.lastIndexOf("/");
                safeFolder(index >= 0 ? absolute(key.slice(0, index)) : cacheDir);
            },
            exists = async (key: string): Promise<boolean> =>
                await chunkedExists({ key, read: readStored }),
            size = async (key: string): Promise<number> =>
                await chunkedSize({ key, read: readStored }),
            readText = async (key: string): Promise<string | null> =>
                red(absolute(key), true) ?? null,
            readBytes = async (key: string): Promise<Uint8Array | null> =>
                await readChunks({ key, read: readStored, decode: decodeBase64 }),
            hasBinary = async (key: string): Promise<boolean> =>
                await chunkedReady({ key, read: readStored }),
            writeText = async (input: {
                key: string;
                text: string;
            }): Promise<void> => {
                ensureFolder(input.key);
                wrt(absolute(input.key), input.text);
            },
            writeStream = async (input: {
                key: string;
                stream: ReadableStream<Uint8Array>;
                onProgress?: (loaded: number) => void;
            }): Promise<void> => {
                ensureFolder(input.key);
                await writeChunkStream({
                    key: input.key,
                    stream: input.stream,
                    onProgress: input.onProgress,
                    encode: encodeBase64,
                    has: async (key) => existsSync(absolute(key)),
                    write: async (key, text) => {
                        wrt(absolute(key), text);
                    },
                    remove: async (key) => {
                        delFile(absolute(key));
                    },
                });
            },
            remove = async (key: string): Promise<void> =>
                await removeChunks({
                    key,
                    read: readStored,
                    has: async (inner) => existsSync(absolute(inner)),
                    remove: async (inner) => {
                        delFile(absolute(inner));
                    },
                }),
            removeRaw = async (key: string): Promise<void> => {
                delFile(absolute(key));
            };

        return {
            exists,
            readText,
            readBytes,
            hasBinary,
            size,
            writeText,
            remove,
            removeRaw,
            writeStream,
        };
    };

export { createNodeStore };
