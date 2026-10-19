import type { BuildWordsOptions, Segment, Word } from "../../tensor/types";
declare const 
/** Build words from generated tokens and their times. */
buildWords: (options: BuildWordsOptions) => Word[], 
/** Group words into sentence segments. */
buildSegments: (options: {
    words: Word[];
    pauseThreshold?: number;
}) => Segment[];
export { buildWords, buildSegments };
