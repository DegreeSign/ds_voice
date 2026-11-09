// The bundled voice library. One Apache-2.0 model per voice.
import { VoiceFamily, VoiceGender, VoiceId } from "./types";
import type { VoiceSpec } from "./types";
const
    /** Fixed line rendered for every preview clip. */
    PREVIEW = "Hi, this is a voice preview. Ha ha ha!",
    /** Sample rate of the bundled preview clips. */
    PREVIEW_RATE = 16000,
    /** Primary model repository of each voice. */
    MOSS_GGUF = "Luigi/MOSS-TTS-Nano-100M-GGUF",
    MOSS_TOKENIZER = "aoiandroid/moss-tts-nano-onnx-fp32",
    /** Every voice in the library. */
    VOICES: Record<VoiceId, VoiceSpec> = {
        [VoiceId.Male]: {
            id: VoiceId.Male,
            label: "Male",
            gender: VoiceGender.Male,
            family: VoiceFamily.Moss,
            speaker: "default",
            sampleRate: 48000,
            previewRate: PREVIEW_RATE,
            license: "Apache-2.0",
            repository: MOSS_GGUF,
            files: [
                {
                    repo: MOSS_GGUF,
                    file: "moss_nano.gguf",
                    bytes: 125_043_936,
                },
                {
                    repo: MOSS_GGUF,
                    file: "moss_codec.gguf",
                    bytes: 13_279_328,
                },
                {
                    repo: MOSS_TOKENIZER,
                    file: "tokenizer/tokenizer.json",
                    bytes: 1_056_852,
                },
            ],
            preview: PREVIEW,
        },
        [VoiceId.Female]: {
            id: VoiceId.Female,
            label: "Female",
            gender: VoiceGender.Female,
            family: VoiceFamily.Moss,
            speaker: "default",
            sampleRate: 48000,
            previewRate: PREVIEW_RATE,
            license: "Apache-2.0",
            repository: MOSS_GGUF,
            files: [
                {
                    repo: MOSS_GGUF,
                    file: "moss_nano.gguf",
                    bytes: 125_043_936,
                },
                {
                    repo: MOSS_GGUF,
                    file: "moss_codec.gguf",
                    bytes: 13_279_328,
                },
                {
                    repo: MOSS_TOKENIZER,
                    file: "tokenizer/tokenizer.json",
                    bytes: 1_056_852,
                },
            ],
            preview: PREVIEW,
        },
    },
    /** Every voice, in library order. */
    listVoices = (): VoiceSpec[] => [
        VOICES[VoiceId.Male],
        VOICES[VoiceId.Female],
    ],
    /** One voice by id. */
    getVoiceSpec = (voice: VoiceId): VoiceSpec => {
        const spec = VOICES[voice];
        if (!spec) {
            throw new Error(`unknown voice ${voice}`);
        }
        return spec;
    };

export { PREVIEW, PREVIEW_RATE, VOICES, listVoices, getVoiceSpec };
