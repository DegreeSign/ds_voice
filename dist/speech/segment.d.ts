import type { SegmentOptions, SpeechChunk } from "../tensor/types";
declare const 
/** Detect speech and cut into chunks of at most 30 seconds. */
planChunks: (options: SegmentOptions) => SpeechChunk[], 
/** Remove a repeated leading phrase at a join. */
dedupeBoundary: (options: {
    previous: string;
    next: string;
}) => string, 
/** Join two text pieces with the right spacing. */
joinText: (options: {
    previous: string;
    next: string;
}) => string;
export { planChunks, dedupeBoundary, joinText };
