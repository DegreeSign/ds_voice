/** Inverse STFT with centered frames, matching torch.istft. */
declare const inverseStft: (options: {
    magnitude: Float32Array;
    phase: Float32Array;
    bins: number;
    frames: number;
    nFft: number;
    hop: number;
}) => Float32Array;
export { inverseStft };
