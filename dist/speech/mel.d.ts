import type { ComputeMelOptions, FftPlan, MelResult } from "../tensor/types";
declare const 
/** Build a Bluestein plan for transform length n. */
buildFftPlan: (n: number) => FftPlan, 
/** Real transform of one frame, returning the power spectrum. */
powerSpectrum: (options: {
    plan: FftPlan;
    frame: Float32Array;
    window: Float32Array;
    bins: number;
    scratchRe: Float32Array;
    scratchIm: Float32Array;
    convRe: Float32Array;
    convIm: Float32Array;
    kernelRe: Float32Array;
    kernelIm: Float32Array;
    out: Float32Array;
    outOffset: number;
    stride: number;
}) => void, 
/** Filterbank energies, mel major [mels, frames]. */
melEnergies: (options: {
    filters: Float32Array;
    power: Float32Array;
    bins: number;
    mels: number;
    frames: number;
}) => Float32Array, 
/** Run the full log-mel pipeline for one chunk. */
computeMel: (options: ComputeMelOptions) => MelResult;
export { buildFftPlan, powerSpectrum, melEnergies, computeMel };
