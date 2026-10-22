// Node model store.
import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { delFile, red, safeFolder, wrt } from "@degreesign/cache";
import {
    decodeManifest,
    encodeManifest,
    hasCompletePartFile,
    manifestKey,
    partKey,
    readPartFile,
    removeChunks,
    sizeOfPartFile,
    streamPartFile,
} from "./chunk";
import type {
    ModelPartStore,
    ModelStore,
    NodeStoreOptions,
} from "../tensor/types";
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
            parts: ModelPartStore = {
                readSizes: async (key: string): Promise<number[] | null> => {
                    const text = await readStored(manifestKey(key));
                    return text === undefined ? null : decodeManifest(text);
                },
                writeSizes: async (input: {
                    key: string;
                    sizes: number[];
                }): Promise<void> => {
                    const key = manifestKey(input.key);
                    ensureFolder(key);
                    wrt(absolute(key), encodeManifest(input.sizes));
                },
                read: async (input: {
                    key: string;
                    index: number;
                }): Promise<Uint8Array | null> => {
                    const text = await readStored(partKey(input));
                    return text === undefined ? null : decodeBase64(text);
                },
                write: async (input: {
                    key: string;
                    index: number;
                    bytes: Uint8Array;
                }): Promise<void> => {
                    const key = partKey(input);
                    ensureFolder(key);
                    wrt(absolute(key), encodeBase64(input.bytes));
                },
                remove: async (input: {
                    key: string;
                    index: number;
                }): Promise<void> => {
                    delFile(absolute(partKey(input)));
                },
            },
            exists = async (key: string): Promise<boolean> =>
                (await parts.readSizes(key)) !== null ||
                (await readStored(key)) !== undefined,
            size = async (key: string): Promise<number> => {
                if ((await parts.readSizes(key)) !== null) {
                    return await sizeOfPartFile({ parts, key });
                }
                const text = await readStored(key);
                return text === undefined
                    ? 0
                    : new TextEncoder().encode(text).length;
            },
            readText = async (key: string): Promise<string | null> =>
                (await readStored(key)) ?? null,
            readBytes = async (key: string): Promise<Uint8Array | null> =>
                await readPartFile({ parts, key }),
            hasBinary = async (key: string): Promise<boolean> =>
                await hasCompletePartFile({ parts, key }),
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
                await streamPartFile({
                    parts,
                    key: input.key,
                    stream: input.stream,
                    onProgress: input.onProgress,
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
            writeStream,
            remove,
            removeRaw,
            parts,
        };
    };

export { createNodeStore };
