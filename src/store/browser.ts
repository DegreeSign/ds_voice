// Browser model store.
import {
    configureStorageSync,
    readLarge,
    saveLarge,
} from "@degreesign/storage";
import {
    chunkedExists,
    chunkedReady,
    chunkedSize,
    readChunks,
    removeChunks,
    writeChunkStream,
} from "./chunk";
import type { ModelStore } from "../tensor/types";
const
    /** Storage namespace for the models. */
    STORAGE_KEY = "ds_voice",
    /** IndexedDB database name. */
    DB_NAME = "ds_voice",
    /** IndexedDB object store name. */
    STORE_NAME = "models",
    /** Encode bytes as base64. */
    encodeBase64 = (bytes: Uint8Array): string => {
        let text = "";
        const block = 0x8000;
        for (let i = 0; i < bytes.length; i += block) {
            text += String.fromCharCode(...bytes.subarray(i, i + block));
        }
        return btoa(text);
    },
    /** Decode base64 to bytes. */
    decodeBase64 = (text: string): Uint8Array => {
        const
            binary = atob(text),
            out = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i += 1) {
            out[i] = binary.charCodeAt(i);
        }
        return out;
    },
    /** Read one stored string, or undefined. */
    readStored = async (key: string): Promise<string | undefined> =>
        await readLarge<string>(key),
    /** Create an IndexedDB model store. */
    createBrowserStore = (): ModelStore => {
        configureStorageSync({
            storageKey: STORAGE_KEY,
            dbName: DB_NAME,
            storeName: STORE_NAME,
            hideErrors: true,
        });
        const
            exists = async (key: string): Promise<boolean> =>
                await chunkedExists({ key, read: readStored }),
            size = async (key: string): Promise<number> =>
                await chunkedSize({ key, read: readStored }),
            readText = async (key: string): Promise<string | null> =>
                (await readLarge<string>(key)) ?? null,
            readBytes = async (key: string): Promise<Uint8Array | null> =>
                await readChunks({ key, read: readStored, decode: decodeBase64 }),
            hasBinary = async (key: string): Promise<boolean> =>
                await chunkedReady({ key, read: readStored }),
            writeText = async (input: {
                key: string;
                text: string;
            }): Promise<void> => {
                await saveLarge({ key: input.key, data: input.text });
            },
            writeStream = async (input: {
                key: string;
                stream: ReadableStream<Uint8Array>;
                onProgress?: (loaded: number) => void;
            }): Promise<void> => {
                await writeChunkStream({
                    key: input.key,
                    stream: input.stream,
                    onProgress: input.onProgress,
                    encode: encodeBase64,
                    write: async (key, text) => {
                        await saveLarge({ key, data: text });
                    },
                    remove: async (key) => {
                        await saveLarge({ key, data: undefined });
                    },
                });
            },
            remove = async (key: string): Promise<void> =>
                await removeChunks({
                    key,
                    read: readStored,
                    remove: async (inner) => {
                        await saveLarge({ key: inner, data: undefined });
                    },
                }),
            removeRaw = async (key: string): Promise<void> => {
                await saveLarge({ key, data: undefined });
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
        };
    };

export { createBrowserStore };
