// Node filesystem access for audio files.
const
    /** Read one audio file into bytes. */
    readAudioFile = async (path: string): Promise<Uint8Array> => {
        const { readFile } = await import("node:fs/promises");
        const buffer = await readFile(path);
        return new Uint8Array(buffer);
    };

export { readAudioFile };
