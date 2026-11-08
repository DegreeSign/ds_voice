# Changelog

All notable changes to `@degreesign/voice` are recorded here. This project
follows semantic versioning.

## 2.0.1

### Added

- `@degreesign/voice`: voice library with `listVoices`, `getVoiceSpec`,
  `getVoiceSample`, `loadVoice`, `isVoiceCached` and `removeVoice`.
- `VoiceId`, `VoiceGender` and `VoiceFamily` enums.
- Three Apache-2.0 voices filed as `female_1` (Kokoro-82M), `female_2`
  (Soprano-1.1-80M) and `male_1` (Qwen3-TTS 0.6B). Each voice ships a 16 kHz
  preview clip in the bundle, so its weights are only fetched on first use.
- `loadVoice` keeps only the selected voice in the store and drops the others.
- `synthesize({ voice, text })`: render speech with the selected voice. The
  Soprano engine is available for `female_2`.
- `prepareVoice({ voice })`: download and parse the selected voice without
  synthesizing, so the first `synthesize` for that voice reuses resident
  weights. Only one voice stays resident and a different voice replaces it.

## 0.7.0

### Added

- `Quality` enum with `High`, `Medium` and `Low` weight precision, defaulting
  to `High` F16.
- `ThreadCount` enum with `Two`, `Four` and `Eight` worker counts, defaulting
  to `Four`.

## 0.6.0

### Added

- `@degreesign/voice/dictation`: `createDictation(options)` for headless
  microphone dictation with pause, cancel, partial text, model progress and
  worker fallback.
- `isDictationSupported()` on `@degreesign/voice/dictation`.
- `DictationState`, `DictationPermission` and `DictationErrorSource` enums.
- `ModelId`, `Language` and `TimestampMode` re-exported from the dictation
  entry.
- `dist/browser/dictation.worker.js`: prebuilt worker asset used by the
  dictation controller.

## 0.2.0

### Added

- `loadModel(options?)`: download a model when needed and return it loaded.
- `isModelCached(options?)`: true when every file of the selected model is
  already stored, reading the manifests only and never touching the network.
- `ProgressPhase` enum: `download`, `verify` and `decode`, reported on
  `DownloadProgress.phase`.
- `ModelOptionsBase.weightsBaseUrl`: serve the weights from a mirror instead of
  Hugging Face.
- `ModelOptionsBase.weightCacheMB`: cap the F32 weight cache in MB, 128 by
  default.
- `ModelOptionsBase.threads`: worker threads for the decode, 1 by default.
- `dist/browser/sensevoice.min.js` (`dsVoiceSenseVoice`) and
  `dist/browser/whisper.min.js` (`dsVoiceWhisper`) family bundles next to the
  full `dsVoice` bundle.

### Changed

- `transcribe` and `preloadVoice` accept `weightsBaseUrl`, `weightCacheMB` and
  `threads`, and report progress across every file of a checkpoint.
- `DownloadProgress.loaded`, `DownloadProgress.total` and
  `DownloadProgress.ratio` are now aggregate over the whole checkpoint, so a
  caller can render one monotonic bar. `DownloadProgress.file` still names the
  file in flight and `phase` labels the step.
- Whisper base is the default model. `SenseVoiceSmall` remains available with
  `model: ModelId.SenseVoiceSmall`.
- Whisper checkpoints now load from Q8_0 GGUF weights (about 85 MB) while the
  configuration and tokenizer still come from the `openai` repositories.
- Browser weight chunks are stored as binary IndexedDB records, weight
  downloads resume over HTTP `Range` and run four chunks at a time, and the
  browser decode can use a worker pool when the page is cross origin isolated.
