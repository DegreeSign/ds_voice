import type { GgufFile, SenseVoiceVocabulary } from "../../tensor/core/types";
declare const 
/** Build a detokenizer from the GGUF token list. */
createSenseVoiceVocabulary: (options: {
    file: GgufFile;
}) => SenseVoiceVocabulary;
export { createSenseVoiceVocabulary };
