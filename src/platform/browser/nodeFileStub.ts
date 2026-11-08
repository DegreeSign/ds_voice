// Browser replacement for Node file access.
const
    /** Filesystem reads are unavailable in the browser bundle. */
    readAudioFile = async (path: string): Promise<Uint8Array> => {
        throw new Error(`filesystem unavailable ${path}`);
    };

export { readAudioFile };
