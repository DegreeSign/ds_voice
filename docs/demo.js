// CDN speech demo page.
const
    engine = window.dsVoice,
    transcript = document.getElementById("transcript"),
    listenButton = document.getElementById("listen"),
    stopButton = document.getElementById("stop"),
    audioInput = document.getElementById("audio_file"),
    transcribeFileButton = document.getElementById("transcribe_file"),
    sttProgress = document.getElementById("stt_progress"),
    sttStatus = document.getElementById("stt_status"),
    voiceSelect = document.getElementById("voice"),
    textInput = document.getElementById("text"),
    useClone = document.getElementById("use_clone"),
    speakButton = document.getElementById("speak"),
    progress = document.getElementById("progress"),
    status = document.getElementById("status"),
    playback = document.getElementById("playback"),
    recordCloneButton = document.getElementById("record_clone"),
    clearCloneButton = document.getElementById("clear_clone"),
    cloneProgress = document.getElementById("clone_progress"),
    cloneStatus = document.getElementById("clone_status"),
    SLIDERS = [
        { input: "pitch", output: "pitch_value", digits: 1 },
        { input: "rate", output: "rate_value", digits: 2 },
        { input: "gain", output: "gain_value", digits: 2 },
        { input: "temperature", output: "temperature_value", digits: 2 },
        { input: "seed", output: "seed_value", digits: 0 },
    ],
    MOSS_ONLY = ["temperature", "seed"],
    NEUTRAL = { pitch: 0, rate: 1, gain: 1, temperature: 0.8, seed: 0 },
    VOICE_DEFAULTS = {
        female: {
            pitch: 1.1,
            rate: 1.05,
            gain: 2,
            temperature: 0.8,
            seed: 0,
        },
    };
let
    playbackUrl = "",
    lastProgressAt = 0,
    synthesizing = false,
    session = null,
    cloned = null,
    poll = 0;

/** Stop the live transcript poll. */
const stopPoll = () => {
    if (!poll) {
        return;
    }
    window.clearInterval(poll);
    poll = 0;
};

/** Set the speech to text controls for one busy state. */
const setSttBusy = (busy) => {
    listenButton.disabled = busy;
    transcribeFileButton.disabled = busy;
    stopButton.disabled = !busy;
};

/** Show one download or decode update. */
const showSttProgress = (update) => {
    sttProgress.value = Math.round(update.ratio * 100);
    sttStatus.textContent = `${update.phase} ${sttProgress.value}%`;
};

/** Follow one session until its transcript arrives. */
const followSession = async ({ label, create }) => {
    if (!engine?.transcribe) {
        sttStatus.textContent = "The speech bundle is not loaded.";
        return;
    }
    setSttBusy(true);
    transcript.value = "";
    sttProgress.value = 0;
    sttStatus.textContent = `Starting ${label}...`;
    stopPoll();
    const active = create();
    session = active;
    poll = window.setInterval(() => {
        const live = active.fullText();
        if (live.text) {
            transcript.value = live.text;
        }
        if (live.msElapsed > 0) {
            sttStatus.textContent = `Decoding ${(
                live.msElapsed / 1000
            ).toFixed(1)} s`;
        }
    }, 200);
    try {
        const result = await active.done;
        transcript.value = result.text;
        const seconds = result.duration.toFixed(1);
        sttStatus.textContent = result.text
            ? `${label} done. ${seconds} s of audio.`
            : `${label} done. No speech detected.`;
        sttProgress.value = 100;
    } catch (error) {
        sttStatus.textContent = `Failed: ${error?.message ?? error}`;
    } finally {
        stopPoll();
        if (session === active) {
            session = null;
        }
        setSttBusy(false);
    }
};

/** Transcribe the microphone with the low quality weights. */
const startListening = () => {
    void followSession({
        label: "microphone",
        create: () => engine.transcribe({
            model: engine.ModelId.WhisperBase,
            language: engine.Language.auto,
            quality: engine.Quality.Low,
            onProgress: showSttProgress,
        }),
    });
};

/** Transcribe the picked audio file. */
const startFile = () => {
    const file = audioInput.files?.[0];
    if (!file) {
        sttStatus.textContent = "Pick an audio file first.";
        return;
    }
    void followSession({
        label: "file",
        create: () => engine.transcribe({
            audioFilePath: file,
            model: engine.ModelId.WhisperBase,
            language: engine.Language.auto,
            quality: engine.Quality.Low,
            onProgress: showSttProgress,
        }),
    });
};

/** Stop the active transcription and keep its text. */
const stopTranscription = () => {
    if (!session) {
        return;
    }
    sttStatus.textContent = "Finishing...";
    stopButton.disabled = true;
    void session.stop().catch(() => undefined);
};

/** Fill the voice list and select Female. */
const fillVoices = () => {
    const voices = engine?.listVoices?.() ?? [];
    voiceSelect.textContent = "";
    for (const spec of voices) {
        const option = document.createElement("option");
        option.value = spec.id;
        option.textContent = `${spec.label} (${spec.id})`;
        voiceSelect.append(option);
    }
    const preferred = voices.find((spec) => spec.id === "female")
        ?? voices[0];
    if (preferred) {
        voiceSelect.value = preferred.id;
    }
};

/** True when the selected voice uses MOSS. */
const isMoss = () => engine?.getVoiceSpec?.(voiceSelect.value)?.family === "moss";

/** Disable the MOSS-only controls for the other engines. */
const updateControls = () => {
    const moss = isMoss();
    for (const id of MOSS_ONLY) {
        const
            input = document.getElementById(id),
            slider = input?.closest(".slider");
        if (!input || !slider) {
            continue;
        }
        input.disabled = !moss;
        slider.style.opacity = moss ? "1" : "0.4";
    }
};

/** Read one slider number. */
const readNumber = (id) => Number(document.getElementById(id).value);

/** Refresh the label of one slider. */
const updateSlider = (slider) => {
    const value = readNumber(slider.input);
    document.getElementById(slider.output).textContent =
        value.toFixed(slider.digits);
};

/** Load the default sliders of one voice. */
const applyDefaults = (voice) => {
    const defaults = VOICE_DEFAULTS[voice] ?? NEUTRAL;
    for (const slider of SLIDERS) {
        const input = document.getElementById(slider.input);
        if (!input) {
            continue;
        }
        input.value = String(defaults[slider.input]);
        updateSlider(slider);
    }
};

/** Encode mono PCM as a 16 bit WAV file. */
const wavBytes = (audio) => {
    const
        samples = audio.pcm,
        data = new Uint8Array(samples.length * 2),
        view = new DataView(data.buffer);
    for (let i = 0; i < samples.length; i += 1) {
        const value = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(i * 2, Math.round(value * 32767), true);
    }
    const
        header = new Uint8Array(44),
        headerView = new DataView(header.buffer),
        write = (offset, text) => {
            for (let i = 0; i < text.length; i += 1) {
                header[offset + i] = text.charCodeAt(i);
            }
        };
    write(0, "RIFF");
    headerView.setUint32(4, 36 + data.length, true);
    write(8, "WAVE");
    write(12, "fmt ");
    headerView.setUint32(16, 16, true);
    headerView.setUint16(20, 1, true);
    headerView.setUint16(22, 1, true);
    headerView.setUint32(24, audio.sampleRate, true);
    headerView.setUint32(28, audio.sampleRate * 2, true);
    headerView.setUint16(32, 2, true);
    headerView.setUint16(34, 16, true);
    write(36, "data");
    headerView.setUint32(40, data.length, true);
    const out = new Uint8Array(header.length + data.length);
    out.set(header, 0);
    out.set(data, header.length);
    return out;
};

/** Play one rendered clip. */
const play = (audio) => {
    const url = URL.createObjectURL(
        new Blob([wavBytes(audio)], { type: "audio/wav" }),
    );
    if (playbackUrl) {
        URL.revokeObjectURL(playbackUrl);
    }
    playbackUrl = url;
    playback.src = url;
    void playback.play().catch(() => undefined);
};

/** Set the clone controls for one busy state. */
const setCloneBusy = (busy) => {
    recordCloneButton.disabled = busy;
    clearCloneButton.disabled = busy || !cloned;
};

/** Record a reference and keep its codes for synthesis. */
const recordClone = async () => {
    if (!engine?.cloneVoice) {
        cloneStatus.textContent = "The voice bundle has no cloning support.";
        return;
    }
    setCloneBusy(true);
    cloneProgress.value = 0;
    cloneStatus.textContent = "Recording 3 s from the microphone...";
    try {
        cloned = await engine.cloneVoice({
            durationMs: 3000,
            onProgress: (update) => {
                const name = update.file || "encoder";
                cloneProgress.value = Math.round(update.ratio * 100);
                cloneStatus.textContent =
                    `${update.phase} ${name} ${cloneProgress.value}%`;
            },
        });
        useClone.checked = true;
        useClone.disabled = false;
        cloneProgress.value = 100;
        cloneStatus.textContent =
            `Reference ready. ${cloned.frames} frames.`;
    } catch (error) {
        const reason = error?.message ?? error;
        useClone.checked = cloned !== null;
        useClone.disabled = cloned === null;
        cloneProgress.value = 0;
        cloneStatus.textContent = cloned
            ? `Failed: ${reason}. Previous reference kept.`
            : `Failed: ${reason}`;
    } finally {
        setCloneBusy(false);
    }
};

/** Drop the recorded reference. */
const clearClone = () => {
    cloned = null;
    useClone.checked = false;
    useClone.disabled = true;
    cloneProgress.value = 0;
    cloneStatus.textContent = "No reference recorded.";
    clearCloneButton.disabled = true;
};

/** Render the current text with the current controls. */
const speak = async () => {
    const text = textInput.value.trim();
    if (!text) {
        status.textContent = "Enter some text first.";
        return;
    }
    if (!engine?.synthesize) {
        status.textContent = "The voice bundle is not loaded.";
        return;
    }
    const started = performance.now();
    synthesizing = false;
    speakButton.disabled = true;
    progress.value = 0;
    status.textContent = "Preparing voice...";
    lastProgressAt = Date.now();
    const ticker = window.setInterval(() => {
        if (!synthesizing && Date.now() - lastProgressAt > 1200) {
            synthesizing = true;
            progress.removeAttribute("value");
        }
        if (synthesizing) {
            const seconds = ((performance.now() - started) / 1000).toFixed(1);
            status.textContent = `Synthesizing... ${seconds} s`;
        }
    }, 500);
    const reference = useClone.checked ? cloned?.codes : undefined;
    try {
        const audio = await engine.synthesize({
            voice: voiceSelect.value,
            text,
            reference,
            pitch: readNumber("pitch"),
            rate: readNumber("rate"),
            gain: readNumber("gain"),
            seed: isMoss() ? readNumber("seed") : undefined,
            temperature: isMoss() ? readNumber("temperature") : undefined,
            onProgress: (update) => {
                lastProgressAt = Date.now();
                synthesizing = false;
                progress.value = Math.round(update.ratio * 100);
                status.textContent = `${update.phase} ${update.file || "voice"} `
                    + `${Math.round(update.ratio * 100)}%`;
            },
        });
        progress.value = 100;
        play(audio);
        const seconds = (audio.pcm.length / audio.sampleRate).toFixed(2);
        status.textContent = `Done in ${(
            (performance.now() - started) / 1000
        ).toFixed(1)} s. ${seconds} s at ${audio.sampleRate} Hz.`;
    } catch (error) {
        progress.value = 0;
        status.textContent = `Failed: ${error?.message ?? error}`;
    } finally {
        window.clearInterval(ticker);
        synthesizing = false;
        speakButton.disabled = false;
    }
};

if (!engine) {
    sttStatus.textContent = "The speech bundle did not load.";
    status.textContent = "The voice bundle did not load.";
    cloneStatus.textContent = "The voice bundle did not load.";
}
fillVoices();
applyDefaults(voiceSelect.value);
for (const slider of SLIDERS) {
    const input = document.getElementById(slider.input);
    input.addEventListener("input", () => updateSlider(slider));
}
voiceSelect.addEventListener("change", () => {
    applyDefaults(voiceSelect.value);
    updateControls();
});
listenButton.addEventListener("click", startListening);
stopButton.addEventListener("click", stopTranscription);
transcribeFileButton.addEventListener("click", startFile);
recordCloneButton.addEventListener("click", () => void recordClone());
clearCloneButton.addEventListener("click", clearClone);
speakButton.addEventListener("click", () => void speak());
updateControls();
