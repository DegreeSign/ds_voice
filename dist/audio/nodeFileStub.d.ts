declare const 
/** Filesystem reads are unavailable in the browser bundle. */
readAudioFile: (path: string) => Promise<Uint8Array>;
export { readAudioFile };
