// Energy based speech chunking.
import type { SegmentOptions, SpeechChunk } from "../tensor/types";
const
    /** Characters that need no joining space. */
    CJK_PATTERN = /[\u2e80-\u9fff\uac00-\ud7af\uf900-\ufaff]/,
    /** One frame energy in dBFS. */
    frameDb = (options: {
        pcm: Float32Array;
        start: number;
        length: number;
    }): number => {
        const { pcm, start, length } = options;
        let sum = 0;
        const end = Math.min(start + length, pcm.length);
        for (let i = start; i < end; i += 1) {
            sum += pcm[i] * pcm[i];
        }
        const
            count = Math.max(1, end - start),
            rms = Math.sqrt(sum / count);
        return 20 * Math.log10(Math.max(rms, 1e-7));
    },
    /** Default chunk energy floor in dBFS. */
    DEFAULT_DB_FLOOR = -50,
    /** True when a buffer carries signal above the floor. */
    hasSignal = (options: SegmentOptions): boolean => {
        const
            { pcm, samplingRate } = options,
            frameLength = Math.round(0.02 * samplingRate);
        if (pcm.length < frameLength) {
            // too short to judge
            return true;
        }
        return frameDb({ pcm, start: 0, length: pcm.length })
            > (options.dbFloor ?? DEFAULT_DB_FLOOR);
    },
    /** Detect speech and cut into chunks of at most 30 seconds. */
    planChunks = (options: SegmentOptions): SpeechChunk[] => {
        const
            { pcm, samplingRate } = options,
            frameLength = Math.round(0.02 * samplingRate),
            frameCount = Math.floor(pcm.length / frameLength);
        if (frameCount === 0) {
            return [];
        }
        const db: number[] = [];
        for (let f = 0; f < frameCount; f += 1) {
            db.push(
                frameDb({
                    pcm,
                    start: f * frameLength,
                    length: frameLength,
                }),
            );
        }
        const
            sortedDb = [...db].sort((a, b) => a - b),
            noiseFloor = sortedDb[Math.floor(sortedDb.length / 4)],
            threshold = Math.min(-30, Math.max(-70, noiseFloor + 10)),
            maxChunk = (options.maxChunkSeconds ?? 30) * samplingRate,
            minPause = Math.round((options.minPauseSeconds ?? 0.3) * 1000),
            minChunk = Math.round((options.minChunkSeconds ?? 0.2) * samplingRate),
            dbFloor = options.dbFloor ?? DEFAULT_DB_FLOOR,
            cutPoints: number[] = [];
        let silenceRun = 0;
        for (let f = 0; f < frameCount; f += 1) {
            if (db[f] <= threshold) {
                silenceRun += 1;
            } else {
                if (silenceRun * 20 >= minPause) {
                    const centerFrame = f - silenceRun / 2;
                    cutPoints.push(Math.round(centerFrame * frameLength));
                }
                silenceRun = 0;
            }
        }
        const chunks: SpeechChunk[] = [];
        let pos = 0;
        const
            limitStep = Math.round(28 * samplingRate),
            overlap = Math.round(2 * samplingRate);
        while (pos < pcm.length) {
            const limit = pos + maxChunk;
            let
                end = pcm.length,
                overlapped = false;
            if (limit < pcm.length) {
                const candidate = largestCut({
                    cuts: cutPoints,
                    low: pos + minChunk,
                    high: limit,
                });
                if (candidate >= 0) {
                    end = candidate;
                } else {
                    end = Math.min(pos + limitStep, pcm.length);
                    overlapped = true;
                }
            }
            const chunkEnd = Math.min(end, pcm.length);
            if (chunkEnd - pos >= minChunk) {
                const mean = frameDb({ pcm, start: pos, length: chunkEnd - pos });
                if (mean > dbFloor) {
                    chunks.push({
                        start: pos / samplingRate,
                        end: chunkEnd / samplingRate,
                    });
                }
            }
            if (chunkEnd >= pcm.length) {
                break;
            }
            pos = overlapped ? Math.max(chunkEnd - overlap, pos + 1) : chunkEnd;
        }
        return chunks;
    },
    /** Largest cut point within a range, or negative. */
    largestCut = (options: {
        cuts: number[];
        low: number;
        high: number;
    }): number => {
        const { cuts, low, high } = options;
        let best = -1;
        for (const cut of cuts) {
            if (cut >= low && cut <= high && cut > best) {
                best = cut;
            }
        }
        return best;
    },
    /** Remove a repeated leading phrase at a join. */
    dedupeBoundary = (options: {
        previous: string;
        next: string;
    }): string => {
        const
            { previous, next } = options,
            prevWords = previous.split(/\s+/).filter(Boolean),
            nextWords = next.split(/\s+/).filter(Boolean);
        let match = 0;
        const max = Math.min(prevWords.length, nextWords.length);
        for (let n = max; n > 0; n -= 1) {
            const
                tail = prevWords.slice(prevWords.length - n).join(" "),
                head = nextWords.slice(0, n).join(" ");
            if (tail === head) {
                match = n;
                break;
            }
        }
        let text = nextWords.slice(match).join(" ");
        if (match === 0 && CJK_PATTERN.test(`${previous}${next}`)) {
            const
                limit = Math.min(previous.length, next.length, 40);
            for (let n = limit; n >= 4; n -= 1) {
                if (previous.endsWith(next.slice(0, n))) {
                    text = next.slice(n);
                    break;
                }
            }
        }
        return text;
    },
    /** Join two text pieces with the right spacing. */
    joinText = (options: {
        previous: string;
        next: string;
    }): string => {
        const
            { previous, next } = options,
            left = previous.slice(-1),
            right = next.slice(0, 1);
        if (!previous || !next) {
            return `${previous}${next}`;
        }
        if (/\s/.test(left) || /\s/.test(right)) {
            return `${previous}${next}`;
        }
        if (CJK_PATTERN.test(left) || CJK_PATTERN.test(right)) {
            return `${previous}${next}`;
        }
        if (/^[.,!?;:%)\]}，。！？；：、）》」』】]/.test(next)) {
            return `${previous}${next}`;
        }
        return `${previous} ${next}`;
    };

export { planChunks, dedupeBoundary, joinText, hasSignal };
