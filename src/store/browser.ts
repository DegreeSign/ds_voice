// Browser model store.
import {
    configureStorageSync,
    readLarge,
    saveLarge,
} from "@degreesign/storage";
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
} from "../tensor/types";
const
    /** Storage namespace for the models. */
    STORAGE_KEY = "ds_voice",
    /** IndexedDB database name. */
    DB_NAME = "ds_voice",
    /** IndexedDB object store name. */
    STORE_NAME = "models",
    /** Binary database name for weight chunks. */
    BINARY_DB = "ds_voice_parts",
    /** Binary object store name for weight chunks. */
    BINARY_STORE = "parts",
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
    /** True when IndexedDB is reachable. */
    hasIndexedDb = (): boolean => typeof indexedDB !== "undefined";

let binaryDb: Promise<IDBDatabase> | null = null;

const
    /** Open the binary chunk database once. */
    openBinaryDb = (): Promise<IDBDatabase> => {
        if (binaryDb) {
            return binaryDb;
        }
        binaryDb = new Promise<IDBDatabase>((resolve, reject) => {
            const request = indexedDB.open(BINARY_DB, 1);
            request.onupgradeneeded = () => {
                if (!request.result.objectStoreNames.contains(BINARY_STORE)) {
                    request.result.createObjectStore(BINARY_STORE);
                }
            };
            request.onsuccess = () => {
                resolve(request.result);
            };
            request.onerror = () => {
                reject(request.error ?? new Error("binary db open failed"));
            };
        });
        return binaryDb;
    },
    /** Run one IndexedDB request and await its result. */
    requestResult = async <T>(request: IDBRequest<T>): Promise<T> => {
        return await new Promise<T>((resolve, reject) => {
            request.onsuccess = () => {
                resolve(request.result);
            };
            request.onerror = () => {
                reject(request.error ?? new Error("binary db request failed"));
            };
        });
    },
    /** Read one binary record, or null. */
    readBinary = async (key: string): Promise<Uint8Array | null> => {
        if (!hasIndexedDb()) {
            return null;
        }
        const
            db = await openBinaryDb(),
            value = await requestResult<unknown>(
                db.transaction(BINARY_STORE, "readonly")
                    .objectStore(BINARY_STORE)
                    .get(key),
            );
        if (value instanceof Uint8Array) {
            return value;
        }
        if (value instanceof ArrayBuffer) {
            return new Uint8Array(value);
        }
        if (value instanceof Blob) {
            return new Uint8Array(await value.arrayBuffer());
        }
        return null;
    },
    /** Write one binary record. */
    writeBinary = async (options: {
        key: string;
        bytes: Uint8Array;
    }): Promise<void> => {
        const db = await openBinaryDb();
        await requestResult(
            db.transaction(BINARY_STORE, "readwrite")
                .objectStore(BINARY_STORE)
                .put(options.bytes.slice(), options.key),
        );
    },
    /** Remove one binary record. */
    removeBinary = async (key: string): Promise<void> => {
        if (!hasIndexedDb()) {
            return;
        }
        const db = await openBinaryDb();
        await requestResult(
            db.transaction(BINARY_STORE, "readwrite")
                .objectStore(BINARY_STORE)
                .delete(key),
        );
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
            parts: ModelPartStore = {
                readSizes: async (key: string): Promise<number[] | null> => {
                    const text = await readStored(manifestKey(key));
                    return text === undefined ? null : decodeManifest(text);
                },
                writeSizes: async (input: {
                    key: string;
                    sizes: number[];
                }): Promise<void> => {
                    await saveLarge({
                        key: manifestKey(input.key),
                        data: encodeManifest(input.sizes),
                    });
                },
                read: async (input: {
                    key: string;
                    index: number;
                }): Promise<Uint8Array | null> => {
                    const part = partKey(input);
                    const binary = await readBinary(part);
                    if (binary) {
                        return binary;
                    }
                    const text = await readStored(part);
                    return text === undefined ? null : decodeBase64(text);
                },
                write: async (input: {
                    key: string;
                    index: number;
                    bytes: Uint8Array;
                }): Promise<void> => {
                    if (hasIndexedDb()) {
                        await writeBinary({
                            key: partKey(input),
                            bytes: input.bytes,
                        });
                        return;
                    }
                    await saveLarge({
                        key: partKey(input),
                        data: encodeBase64(input.bytes),
                    });
                },
                remove: async (input: {
                    key: string;
                    index: number;
                }): Promise<void> => {
                    const part = partKey(input);
                    await removeBinary(part);
                    await saveLarge({ key: part, data: undefined });
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
                (await readLarge<string>(key)) ?? null,
            readBytes = async (key: string): Promise<Uint8Array | null> =>
                await readPartFile({ parts, key }),
            hasBinary = async (key: string): Promise<boolean> =>
                await hasCompletePartFile({ parts, key }),
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
                    has: async (inner) =>
                        (await readBinary(inner)) !== null ||
                        (await readStored(inner)) !== undefined,
                    remove: async (inner) => {
                        await removeBinary(inner);
                        await saveLarge({ key: inner, data: undefined });
                    },
                }),
            removeRaw = async (key: string): Promise<void> => {
                await removeBinary(key);
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
            parts,
        };
    };

export { createBrowserStore };
