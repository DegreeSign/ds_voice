declare const 
/** Bytes of one bundled voice file, or undefined. */
getBundledVoiceFile: (file: string) => Uint8Array | undefined, 
/** True when the file ships with the package. */
isBundledVoiceFile: (file: string) => boolean;
export { getBundledVoiceFile, isBundledVoiceFile };
