// Browser microphone capture.
import { resampleAudio } from "./resample";
import type { MicrophoneCapture } from "../tensor/types";
const
    /** Samples per audio worklet block. */
    TARGET_RATE = 16000,
    /** Inline worklet source pushing raw frames to the main thread. */
    WORKLET_SOURCE = `
class DsVoiceCapture extends AudioWorkletProcessor {
    process(inputs) {
        const channel = inputs[0]?.[0];
        if (channel && channel.length > 0) {
            this.port.postMessage(channel.slice());
        }
        return true;
    }
}
registerProcessor("ds-voice-capture", DsVoiceCapture);
`,
    /** Start microphone capture, returning a stop handle. */
    captureMicrophone = (options: MicrophoneCapture): (() => Promise<void>) => {
        let
            stopped = false,
            release = async (): Promise<void> => undefined;
        const promise = navigator.mediaDevices
            .getUserMedia({
                audio: {
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                },
            })
            .then(async (stream) => {
                const
                    context = new AudioContext({ sampleRate: TARGET_RATE }),
                    source = context.createMediaStreamSource(stream),
                    gain = context.createGain();
                gain.gain.value = 0;
                const
                    push = (samples: Float32Array): void => {
                        if (stopped) {
                            return;
                        }
                        options.onChunk(
                            context.sampleRate === TARGET_RATE
                                ? samples
                                : resampleAudio({
                                    input: samples,
                                    inputRate: context.sampleRate,
                                    outputRate: TARGET_RATE,
                                }),
                        );
                    };
                let
                    node: AudioWorkletNode | null = null,
                    processor: ScriptProcessorNode | null = null,
                    moduleUrl = "";
                if (context.audioWorklet) {
                    moduleUrl = URL.createObjectURL(
                        new Blob([WORKLET_SOURCE], {
                            type: "application/javascript",
                        }),
                    );
                    await context.audioWorklet.addModule(moduleUrl);
                    node = new AudioWorkletNode(context, "ds-voice-capture", {
                        numberOfInputs: 1,
                        numberOfOutputs: 1,
                        channelCount: 1,
                    });
                    node.port.onmessage = (event: MessageEvent<Float32Array>) => {
                        push(event.data);
                    };
                    source.connect(node);
                    node.connect(gain);
                } else {
                    processor = context.createScriptProcessor(4096, 1, 1);
                    processor.onaudioprocess = (event: AudioProcessingEvent) => {
                        push(event.inputBuffer.getChannelData(0).slice());
                    };
                    source.connect(processor);
                    processor.connect(gain);
                }
                gain.connect(context.destination);
                release = async (): Promise<void> => {
                    stopped = true;
                    if (node) {
                        node.port.onmessage = null;
                        node.disconnect();
                    }
                    if (processor) {
                        processor.onaudioprocess = null;
                        processor.disconnect();
                    }
                    source.disconnect();
                    gain.disconnect();
                    for (const track of stream.getTracks()) {
                        track.stop();
                    }
                    if (moduleUrl) {
                        URL.revokeObjectURL(moduleUrl);
                    }
                    await context.close();
                };
                if (stopped) {
                    await release();
                }
            })
            .catch((reason: Error) => {
                options.onError(reason);
            });
        void promise;
        return async (): Promise<void> => {
            stopped = true;
            await release();
        };
    };

export { captureMicrophone };
