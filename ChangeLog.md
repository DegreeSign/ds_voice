# Changelog

All notable changes to `@degreesign/voice` are recorded here. This project
follows semantic versioning.

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
