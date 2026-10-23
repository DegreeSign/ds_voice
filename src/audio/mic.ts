// Browser microphone capture.
import { resampleAudio } from "./resample";
import type {
    MicrophoneCapture,
    MicrophoneCaptureHandle,
} from "../tensor/types";
const
    /** Working sample rate of the engine. */
    TARGET_RATE = 16000,
    /** Analyser resolution for the waveform. */
    FFT_SIZE = 256,
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
    /** Start microphone capture, returning a handle. */
    captureMicrophone = (
        options: MicrophoneCapture,
    ): MicrophoneCaptureHandle => {
        let
            context: AudioContext | null = null,
            stream: MediaStream | null = null,
            source: MediaStreamAudioSourceNode | null = null,
            analyser: AnalyserNode | null = null,
            gain: GainNode | null = null,
            captureNode: AudioWorkletNode | ScriptProcessorNode | null = null,
            levels: Uint8Array<ArrayBuffer> | null = null,
            moduleUrl = ``,
            frame = 0,
            stopped = false,
            paused = false;
        const
            /** Handle for a capture that never starts. */
            idle: MicrophoneCaptureHandle = {
                ready: Promise.resolve(),
                stop: async (): Promise<void> => undefined,
                pause: async (): Promise<void> => undefined,
                resume: async (): Promise<void> => undefined,
            },
            /** Stop the waveform loop. */
            stopFrequency = (): void => {
                if (!frame) {
                    return;
                }
                cancelAnimationFrame(frame);
                frame = 0;
            },
            /** Run the waveform loop. */
            startFrequency = (): void => {
                if (
                    !options.onFrequency || !analyser || !levels
                    || frame || stopped || paused
                ) {
                    return;
                }
                const tick = (): void => {
                    if (
                        !analyser || !levels || !options.onFrequency
                        || stopped || paused
                    ) {
                        frame = 0;
                        return;
                    }
                    analyser.getByteFrequencyData(levels);
                    options.onFrequency(levels);
                    frame = requestAnimationFrame(tick);
                };
                frame = requestAnimationFrame(tick);
            },
            /** Forward one captured frame. */
            push = (samples: Float32Array): void => {
                if (stopped || paused || samples.length === 0) {
                    return;
                }
                options.onChunk(
                    context?.sampleRate === TARGET_RATE
                        ? samples
                        : resampleAudio({
                            input: samples,
                            inputRate: context?.sampleRate ?? TARGET_RATE,
                            outputRate: TARGET_RATE,
                        }),
                );
            },
            /** Release every audio node and the microphone. */
            cleanUp = async (): Promise<void> => {
                stopFrequency();
                if (captureNode) {
                    if (`port` in captureNode) {
                        captureNode.port.onmessage = null;
                    } else {
                        captureNode.onaudioprocess = null;
                    }
                    captureNode.disconnect();
                    captureNode = null;
                }
                analyser?.disconnect();
                analyser = null;
                source?.disconnect();
                source = null;
                gain?.disconnect();
                gain = null;
                for (const track of stream?.getTracks() ?? []) {
                    track.stop();
                }
                stream = null;
                levels = null;
                if (moduleUrl) {
                    URL.revokeObjectURL(moduleUrl);
                    moduleUrl = ``;
                }
                if (context && context.state !== `closed`) {
                    const active = context;
                    context = null;
                    await active.close().catch(() => undefined);
                }
            },
            /** Fail one capture. */
            fail = (reason: Error): void => {
                if (stopped) {
                    return;
                }
                stopped = true;
                void cleanUp().then(() => options.onError(reason));
            };

        // resume first so Safari keeps the click gesture
        if (!navigator.mediaDevices?.getUserMedia) {
            const reason = new Error(`Microphone unavailable`);
            queueMicrotask(() => options.onError(reason));
            return {
                ...idle,
                ready: Promise.reject(reason),
            };
        }
        try {
            context = new AudioContext({ sampleRate: TARGET_RATE });
        } catch (error) {
            const reason = error instanceof Error
                ? error
                : new Error(String(error));
            queueMicrotask(() => options.onError(reason));
            return {
                ...idle,
                ready: Promise.reject(reason),
            };
        }
        const activeContext = context;
        void activeContext.resume().catch(() => undefined);
        const ready = navigator.mediaDevices
            .getUserMedia({
                audio: {
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                },
            })
            .then(async (input) => {
                if (stopped) {
                    for (const track of input.getTracks()) {
                        track.stop();
                    }
                    await cleanUp();
                    return;
                }
                stream = input;
                const
                    inputSource = activeContext
                        .createMediaStreamSource(input),
                    output = activeContext.createGain();
                source = inputSource;
                gain = output;
                output.gain.value = 0;
                if (options.onFrequency) {
                    const node = activeContext.createAnalyser();
                    node.fftSize = FFT_SIZE;
                    analyser = node;
                    levels = new Uint8Array(node.frequencyBinCount);
                    inputSource.connect(node);
                    node.connect(output);
                }
                let worklet = false;
                if (activeContext.audioWorklet) {
                    try {
                        moduleUrl = URL.createObjectURL(new Blob(
                            [WORKLET_SOURCE],
                            { type: `application/javascript` },
                        ));
                        await activeContext.audioWorklet.addModule(moduleUrl);
                        if (stopped) {
                            await cleanUp();
                            return;
                        }
                        const node = new AudioWorkletNode(
                            activeContext,
                            `ds-voice-capture`,
                            {
                                numberOfInputs: 1,
                                numberOfOutputs: 1,
                                channelCount: 1,
                            },
                        );
                        node.port.onmessage = (
                            event: MessageEvent<Float32Array>,
                        ) => push(event.data);
                        inputSource.connect(node);
                        node.connect(output);
                        captureNode = node;
                        worklet = true;
                    } catch (error) {
                        console.log(`Capture worklet unavailable`, error);
                    }
                }
                if (!worklet) {
                    const node = activeContext.createScriptProcessor(
                        4096,
                        1,
                        1,
                    );
                    node.onaudioprocess = (event) => {
                        push(event.inputBuffer.getChannelData(0).slice());
                    };
                    inputSource.connect(node);
                    node.connect(output);
                    captureNode = node;
                }
                output.connect(activeContext.destination);
                startFrequency();
                if (paused) {
                    await activeContext.suspend().catch(() => undefined);
                }
            })
            .catch((reason: Error) => {
                fail(reason);
                throw reason;
            });

        return {
            ready,
            stop: async (): Promise<void> => {
                if (stopped) {
                    return;
                }
                stopped = true;
                await cleanUp();
            },
            pause: async (): Promise<void> => {
                if (paused || stopped) {
                    return;
                }
                paused = true;
                stopFrequency();
                if (context) {
                    await context.suspend().catch(() => undefined);
                }
            },
            resume: async (): Promise<void> => {
                if (!paused || stopped) {
                    return;
                }
                paused = false;
                if (context) {
                    await context.resume().catch(() => undefined);
                }
                startFrequency();
            },
        };
    };

export { captureMicrophone };
