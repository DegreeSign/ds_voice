/** Options for WAV parsing. */
export interface ReadWavOptions {
    bytes: Uint8Array;
}

/** Decoded WAV payload. */
export interface WavData {
    samples: Float32Array;
    sampleRate: number;
    channels: number;
    bitsPerSample: number;
    format: number;
}

/** Options for resampling. */
export interface ResampleOptions {
    input: Float32Array;
    inputRate: number;
    outputRate: number;
}
