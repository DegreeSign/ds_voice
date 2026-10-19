// Manual model recorder.
// Serve repo root.
const
    /** Cache cap that keeps every model while they are compared. */
    DISK_MB = 4096,
    /** Recording container preference, first supported wins. */
    MIME_TYPES = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/ogg;codecs=opus",
        "audio/mp4",
    ],
    /** Human labels for the model ids. */
    MODEL_LABELS = {
        "sensevoice-small": "SenseVoice small",
        "whisper-base": "Whisper base",
    },
    /** Languages per model, empty means every language. */
    MODEL_LANGUAGES = {
        "sensevoice-small": ["zh", "yue", "en", "ja", "ko"],
        "whisper-base": [],
    },
    /** Every supported model, in spec order. */
    MODEL_IDS = Object.values(window.dsVoice?.ModelId ?? {}),
    engine = window.dsVoice,
    button = document.getElementById("record"),
    rerecordButton = document.getElementById("rerecord"),
    languageSelect = document.getElementById("language"),
    status = document.getElementById("status"),
    playback = document.getElementById("playback"),
    results = document.getElementById("results");
let
    recorder = null,
    stream = null,
    chunks = [],
    ticker = 0,
    startedAt = 0,
    phase = "idle",
    playbackUrl = "",
    runToken = 0,
    activeSession = null,
    hasClip = false;

/** Fill the language dropdown from the package enum. */
const fillLanguages = () => {
    const
        values = Object.values(engine?.Language ?? {}),
        codes = ["auto"].concat(
            values.filter((value) => value !== "auto").sort(),
        ),
        names = typeof Intl.DisplayNames === "function"
            ? new Intl.DisplayNames(["en"], { type: "language" })
            : null;
    languageSelect.textContent = "";
    for (const code of codes) {
        const
            option = document.createElement("option"),
            label = code === "auto"
                ? "Auto detect"
                : `${names?.of(code) ?? code} (${code})`;
        option.value = code;
        option.textContent = label;
        languageSelect.append(option);
    }
};

fillLanguages();

/** Choose a MediaRecorder container the browser can decode again. */
const pickMimeType = () => {
    for (const type of MIME_TYPES) {
        if (MediaRecorder.isTypeSupported(type)) {
            return type;
        }
    }
    return "";
};

/** Mark the button while the microphone is open. */
const showRecording = (active) => {
    button.classList.toggle("record_button_live", active);
    button.textContent = active ? "Stop and test" : "Record";
};

/** True when one model can transcribe a language. */
const supportsLanguage = (options) => {
    const languages = MODEL_LANGUAGES[options.model];
    return !languages || languages.length === 0
        || languages.includes(options.language);
};

/** Models that can transcribe the selected language. */
const visibleModelIds = () => {
    const language = languageSelect.value;
    return language === "auto"
        ? MODEL_IDS
        : MODEL_IDS.filter((model) => supportsLanguage({ model, language }));
};

/** Build one empty results row per model. */
const resetRows = (ids) => {
    const rows = new Map();
    results.textContent = "";
    for (const id of ids) {
        const
            row = document.createElement("tr"),
            modelCell = document.createElement("td"),
            stateCell = document.createElement("td"),
            timeCell = document.createElement("td"),
            textCell = document.createElement("td"),
            label = MODEL_LABELS[id] ?? id;
        row.className = "results_row";
        modelCell.className = "results_cell results_model";
        stateCell.className = "results_cell";
        timeCell.className = "results_cell";
        textCell.className = "results_cell results_text";
        modelCell.textContent = label;
        stateCell.textContent = "Queued";
        timeCell.textContent = "";
        textCell.textContent = "";
        row.append(modelCell, stateCell, timeCell, textCell);
        results.append(row);
        rows.set(id, { stateCell, timeCell, textCell });
    }
    return rows;
};

resetRows(visibleModelIds());

/** Run every matching model sequentially on the same recording. */
const testModels = async (blob) => {
    const
        run = ++runToken,
        language = languageSelect.value,
        ids = visibleModelIds(),
        rows = resetRows(ids);
    let failures = 0;
    hasClip = true;
    languageSelect.disabled = true;
    status.textContent = "Testing models, first run downloads the weights.";
    for (const id of ids) {
        if (run !== runToken) {
            return;
        }
        const cells = rows.get(id);
        let session = null;
        cells.stateCell.textContent = "Loading";
        try {
            session = engine.transcribe({
                model: id,
                audioFilePath: blob,
                language,
                allowedDiskSpaceMB: DISK_MB,
                onProgress: (progress) => {
                    if (run !== runToken) {
                        return;
                    }
                    cells.stateCell.textContent = progress.total > 0
                        ? `Downloading ${Math.round(progress.ratio * 100)}%`
                        : "Downloading";
                },
            });
            activeSession = session;
            const transcript = await session.done;
            if (run !== runToken) {
                return;
            }
            const decodeSeconds = transcript.meta.rtf * transcript.duration;
            cells.stateCell.textContent = "Done";
            cells.timeCell.textContent = `${decodeSeconds.toFixed(1)} s`;
            cells.textCell.textContent = transcript.text || "(empty)";
        } catch (error) {
            if (run !== runToken) {
                return;
            }
            failures += 1;
            cells.stateCell.textContent = "Failed";
            cells.textCell.textContent = error?.message ?? String(error);
        } finally {
            if (activeSession === session) {
                activeSession = null;
            }
        }
    }
    if (run !== runToken) {
        return;
    }
    phase = "idle";
    languageSelect.disabled = false;
    button.disabled = false;
    button.textContent = "Record again";
    rerecordButton.disabled = false;
    status.textContent = failures === 0
        ? `All ${ids.length} models finished.`
        : `${failures} of ${ids.length} models failed.`;
};

/** Cancel the running model test. */
const cancelTests = () => {
    runToken += 1;
    languageSelect.disabled = false;
    const session = activeSession;
    activeSession = null;
    if (session) {
        void session.stop().catch(() => undefined);
    }
};

/** Stop the microphone. */
const stopRecording = () => {
    window.clearInterval(ticker);
    ticker = 0;
    recorder?.stop();
    for (const track of stream?.getTracks() ?? []) {
        track.stop();
    }
    stream = null;
};

/** Close the microphone and build the clip. */
const onRecorderStop = () => {
    const
        type = recorder?.mimeType || "audio/webm",
        blob = new Blob(chunks, { type });
    chunks = [];
    recorder = null;
    if (playbackUrl) {
        URL.revokeObjectURL(playbackUrl);
    }
    playbackUrl = URL.createObjectURL(blob);
    playback.src = playbackUrl;
    playback.classList.add("playback_ready");
    hasClip = true;
    rerecordButton.disabled = false;
    status.textContent = "Recorded, running the models.";
    void testModels(blob);
};

/** Open the microphone and start collecting chunks. */
const startRecording = async () => {
    let input = null;
    try {
        input = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
            },
        });
        const
            mimeType = pickMimeType(),
            options = mimeType ? { mimeType } : {};
        recorder = new MediaRecorder(input, options);
    } catch (error) {
        for (const track of input?.getTracks() ?? []) {
            track.stop();
        }
        status.textContent =
            `Recording unavailable: ${error?.message ?? String(error)}`;
        phase = "idle";
        button.disabled = false;
        button.textContent = hasClip ? "Record again" : "Record";
        rerecordButton.disabled = !hasClip;
        return;
    }
    stream = input;
    chunks = [];
    recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            chunks.push(event.data);
        }
    };
    recorder.onstop = onRecorderStop;
    recorder.start();
    startedAt = performance.now();
    ticker = window.setInterval(() => {
        const seconds = ((performance.now() - startedAt) / 1000).toFixed(1);
        status.textContent = `Recording ${seconds} s`;
    }, 200);
    phase = "recording";
    button.disabled = false;
    rerecordButton.disabled = true;
    showRecording(true);
    status.textContent = "Recording 0.0 s";
};

/** Start a fresh recording from the current phase. */
const beginRecording = async () => {
    phase = "starting";
    button.disabled = true;
    rerecordButton.disabled = true;
    await startRecording();
};

/** Toggle between recording and running the models. */
const onPress = async () => {
    if (!engine) {
        status.textContent =
            "Run yarn build first, the browser bundle is missing.";
        return;
    }
    if (phase === "testing" || phase === "starting") {
        return;
    }
    if (phase === "recording") {
        phase = "testing";
        button.disabled = true;
        rerecordButton.disabled = false;
        showRecording(false);
        button.textContent = "Testing";
        stopRecording();
        return;
    }
    await beginRecording();
};

/** Cancel the remaining models and record a new clip. */
const onRerecord = async () => {
    if (!engine) {
        status.textContent =
            "Run yarn build first, the browser bundle is missing.";
        return;
    }
    if (phase !== "testing" && phase !== "idle") {
        return;
    }
    cancelTests();
    await beginRecording();
};

button.addEventListener("click", () => {
    void onPress();
});

rerecordButton.addEventListener("click", () => {
    void onRerecord();
});

languageSelect.addEventListener("change", () => {
    if (phase === "testing") {
        return;
    }
    resetRows(visibleModelIds());
});
