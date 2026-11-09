// Voice synthesis dispatch. Only the selected model is downloaded.
import { getStore } from "../platform/store";
import { loadVoice } from "./download";
import { getVoiceSpec } from "./voices";
import { getBundledVoiceFile } from "./bundled";
import { prepareKokoro, synthesizeKokoro } from "./engines/kokoro";
import { prepareMoss, synthesizeMoss } from "./engines/moss";
import { prepareSoprano, synthesizeSoprano } from "./engines/soprano";
import { VoiceFamily } from "./types";
import type {
    LoadVoiceOptions,
    SynthesizeVoiceOptions,
    VoiceAudio,
    VoiceSpec,
} from "./types";
const
    /** Read one stored voice file. */
    readVoiceFile = async (options: {
        voice: string;
        file: string;
    }): Promise<Uint8Array> => {
        const bundled = getBundledVoiceFile(options.file);
        if (bundled) {
            return bundled;
        }
        const
            store = await getStore(),
            bytes = await store.readBytes(
                `tts/${options.voice}/${options.file}`,
            );
        if (!bytes) {
            throw new Error(`voice file ${options.file} is not loaded`);
        }
        return bytes;
    },
    /** Load one voice and reader. */
    libraryVoice = async (options: LoadVoiceOptions): Promise<{
        spec: VoiceSpec;
        load: (file: string) => Promise<Uint8Array>;
    }> => {
        const spec = getVoiceSpec(options.voice);
        if (
            spec.family !== VoiceFamily.Soprano &&
            spec.family !== VoiceFamily.Kokoro &&
            spec.family !== VoiceFamily.Moss
        ) {
            throw new Error(`voice engine ${spec.family} is not available`);
        }
        await loadVoice({
            voice: options.voice,
            weightsBaseUrl: options.weightsBaseUrl,
            onProgress: options.onProgress,
        });
        return {
            spec,
            load: async (file: string): Promise<Uint8Array> =>
                await readVoiceFile({ voice: spec.id, file }),
        };
    },
    /** Synthesize text with the selected library voice. */
    synthesize = async (
        options: SynthesizeVoiceOptions,
    ): Promise<VoiceAudio> => {
        const { spec, load } = await libraryVoice(options);
        if (spec.family === VoiceFamily.Kokoro) {
            return await synthesizeKokoro({
                voice: spec.id,
                text: options.text,
                load,
            });
        }
        if (spec.family === VoiceFamily.Moss) {
            return await synthesizeMoss({
                voice: spec.id,
                text: options.text,
                load,
            });
        }
        return await synthesizeSoprano({
            voice: spec.id,
            text: options.text,
            load,
        });
    },
    /** Load and parse the selected voice, without speaking. */
    prepareVoice = async (options: LoadVoiceOptions): Promise<void> => {
        const { spec, load } = await libraryVoice(options);
        if (spec.family === VoiceFamily.Kokoro) {
            await prepareKokoro({ voice: spec.id, load });
            return;
        }
        if (spec.family === VoiceFamily.Moss) {
            await prepareMoss({ voice: spec.id, load });
            return;
        }
        await prepareSoprano({ voice: spec.id, load });
    };

export { prepareVoice, synthesize };
