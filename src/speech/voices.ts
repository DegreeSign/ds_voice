// The bundled voice library. One Apache-2.0 model per voice.
import { VoiceFamily, VoiceGender, VoiceId } from "./types";
import type { VoiceSpec } from "./types";
const
    /** Fixed line rendered for every preview clip. */
    PREVIEW = "Hi, this is a voice preview. Ha ha ha!",
    /** Sample rate of the bundled preview clips. */
    PREVIEW_RATE = 16000,
    /** Primary model repository of each voice. */
    KOKORO_GGUF = "cstr/kokoro-82m-GGUF",
    KOKORO_VOICES = "cstr/kokoro-voices-GGUF",
    SOPRANO_REPO = "ekwek/Soprano-1.1-80M",
    SOPRANO_GGUF = "BricksDisplay/Soprano-1.1-80M-GGUF",
    QWEN_REPO = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    MOSS_GGUF = "Luigi/MOSS-TTS-Nano-100M-GGUF",
    MOSS_TOKENIZER = "aoiandroid/moss-tts-nano-onnx-fp32",
    /** Every voice in the library. */
    VOICES: Record<VoiceId, VoiceSpec> = {
        [VoiceId.Female]: {
            id: VoiceId.Female,
            label: "Female 1",
            gender: VoiceGender.Female,
            family: VoiceFamily.Kokoro,
            speaker: "af_heart",
            sampleRate: 24000,
            previewRate: PREVIEW_RATE,
            license: "Apache-2.0",
            repository: KOKORO_GGUF,
            files: [
                {
                    repo: KOKORO_GGUF,
                    file: "kokoro-82m-q8_0.gguf",
                    bytes: 141_322_336,
                },
                {
                    repo: KOKORO_VOICES,
                    file: "kokoro-voice-af_heart.gguf",
                    bytes: 522_784,
                },
            ],
            preview: PREVIEW,
        },
        [VoiceId.Female2]: {
            id: VoiceId.Female2,
            label: "Female 2",
            gender: VoiceGender.Female,
            family: VoiceFamily.Soprano,
            speaker: "default",
            sampleRate: 32000,
            previewRate: PREVIEW_RATE,
            license: "Apache-2.0",
            repository: SOPRANO_GGUF,
            files: [
                {
                    repo: SOPRANO_GGUF,
                    file: "Soprano-1.1-80M.F16.gguf",
                    bytes: 159_632_416,
                },
                {
                    repo: SOPRANO_GGUF,
                    file: "codec-F32.gguf",
                    bytes: 121_423_904,
                },
                {
                    repo: SOPRANO_REPO,
                    file: "tokenizer.json",
                    bytes: 1_630_675,
                },
            ],
            preview: PREVIEW,
        },
        [VoiceId.Male]: {
            id: VoiceId.Male,
            label: "Male",
            gender: VoiceGender.Male,
            family: VoiceFamily.Kokoro,
            speaker: "am_michael",
            sampleRate: 24000,
            previewRate: PREVIEW_RATE,
            license: "Apache-2.0",
            repository: KOKORO_GGUF,
            files: [
                {
                    repo: KOKORO_GGUF,
                    file: "kokoro-82m-q8_0.gguf",
                    bytes: 141_322_336,
                },
                {
                    repo: "",
                    file: "kokoro-voice-am_michael.gguf",
                    bytes: 522_528,
                    bundled: true,
                },
            ],
            preview: PREVIEW,
        },
        [VoiceId.Male2]: {
            id: VoiceId.Male2,
            label: "Male 2",
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
    },
    /** Every voice, in library order. */
    listVoices = (): VoiceSpec[] => [
        VOICES[VoiceId.Female],
        VOICES[VoiceId.Female2],
        VOICES[VoiceId.Male],
        VOICES[VoiceId.Male2],
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
