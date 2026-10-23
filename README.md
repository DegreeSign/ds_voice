# @degreesign/voice

Offline speech recognition for the browser and Node.js with the SenseVoice and
Whisper model families, running fully locally with no Python, no ONNX Runtime
and no runtime dependencies.

[![npm version](https://img.shields.io/npm/v/@degreesign/voice.svg)](https://www.npmjs.com/package/@degreesign/voice)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](#licensing)
[![node](https://img.shields.io/badge/node-%3E%3D24-brightgreen.svg)](#faq)
[![types](https://img.shields.io/badge/types-included-blue.svg)](#faq)

## Table of contents

- [Why](#why)
- [Install](#install)
- [Quick start](#quick-start)
- [Dictation controller](#dictation-controller)
- [Public API](#public-api)
- [Models](#models)
- [Audio input](#audio-input)
- [Disk budget](#disk-budget)
- [Storage](#storage)
- [Performance](#performance)
- [FAQ](#faq)
- [Licensing](#licensing)

## Why

- Offline: weights download once, then every transcription runs on the machine.
- Streaming: `transcribe` returns a live handle immediately and keeps updating.
- One API: the same entry autodetects Node.js and the browser.
- Any audio: PCM, a blob, a URL or a filesystem path, or the microphone in the
  browser.
- Typed: written in TypeScript with declarations for every export.
- No runtime dependencies: the cache, storage and MP3 decoder are bundled in.

## Install

```bash
npm install @degreesign/voice
```

```bash
yarn add @degreesign/voice
```

```bash
pnpm add @degreesign/voice
```

The package builds to four bundles from one source:

- `dist/node/degreesign.node.min.js`, the CommonJS entry used by Node.
- `dist/browser/degreesign.min.js`, the UMD entry with both model families,
  exporting the `dsVoice` global.
- `dist/browser/sensevoice.min.js`, the UMD entry with only SenseVoice,
  exporting `dsVoiceSenseVoice`.
- `dist/browser/whisper.min.js`, the UMD entry with only Whisper, exporting
  `dsVoiceWhisper`.

A worker that serves one family can load the matching bundle and skip the other
family, its tokenizer and its graph. The full bundle keeps the complete API.

Build from source with `yarn build`; typecheck with `yarn typecheck`.

### CDN

```html
<script src="https://cdn.jsdelivr.net/npm/@degreesign/voice/dist/browser/degreesign.min.js"></script>
```

## Quick start

Preload a model once, then transcribe. The input is 16 kHz mono `Float32Array`
PCM.

```ts
import { Language, ModelId, preloadVoice, transcribe } from "@degreesign/voice";

await preloadVoice({ model: ModelId.WhisperBase });

const session = transcribe({ pcm, language: Language.auto });
const live = session.fullText();
console.log(live.text, live.msElapsed);

const result = await session.done;
console.log(result.text);
```

In the browser the same functions are on the `dsVoice` global:

```html
<script src="https://cdn.jsdelivr.net/npm/@degreesign/voice/dist/browser/degreesign.min.js"></script>
<script>
  const { ModelId, transcribe } = window.dsVoice;
  const session = transcribe({ model: ModelId.WhisperBase });
  setInterval(() => {
    document.body.textContent = session.fullText().text;
  }, 1000);
  onStopButton.addEventListener("click", () => session.stop());
</script>
```

A browser session with no `pcm` and no `audioFilePath` asks for microphone
access and keeps recording until `stop`. A Node session with no audio source
rejects through `done` because there is no microphone.

Pass `microphone: false` to capture audio elsewhere and feed it to the session
with `push`. This keeps capture and its UI on the page while the decode runs in
a worker, where `getUserMedia` and `AudioContext` are not available.

```ts
const session = transcribe({ model: ModelId.WhisperBase, microphone: false });

session.push(pcm); // 16 kHz mono Float32Array, as often as audio arrives
setInterval(() => {
  document.body.textContent = session.fullText().text;
}, 200);
await session.stop();
```

## Dictation controller

`@degreesign/voice/dictation` adds a headless controller for live microphone
dictation. It owns the worker, capture, pause state, model progress and partial
text. The page keeps its own markup, styles and copy.

Copy the prebuilt worker next to the page, or serve it from any same origin
path:

```bash
cp node_modules/@degreesign/voice/dist/browser/dictation.worker.js public/dictation.worker.js
```

```ts
import {
  createDictation,
  DictationState,
  Language,
} from "@degreesign/voice/dictation";

const dictation = createDictation({
  workerUrl: "/dictation.worker.js",
  language: Language.en,
  onState: (state, { modelReady, ratio }) => {
    status.textContent = state === DictationState.recording && modelReady
      ? "Listening"
      : `${state} ${Math.round(ratio * 100)}%`;
  },
  onText: (text, final) => {
    input.value = text;
    if (final) {
      input.readOnly = false;
    }
  },
  onFrequency: (levels) => {
    wave.style.opacity = String(levels[2] / 255);
  },
  onError: ({ source, message }) => {
    console.error(source, message);
  },
});

micButton.onclick = () => {
  input.readOnly = true;
  void dictation.start();
};
pauseButton.onclick = () => dictation.togglePause();
doneButton.onclick = () => void dictation.finish();
cancelButton.onclick = () => dictation.cancel();
```

One controller belongs to one input or component. All controllers on a page
share one worker, and starting one cancels the active one. A cancelled
dictation emits `onText("", true)` so the page can restore its base text.

### Controller methods

| Member | Description |
| ------ | ----------- |
| `start()` | Ask for the microphone and start recording. Resolves once capture runs. |
| `pause()` / `resume()` | Suspend or resume capture without dropping audio. |
| `togglePause()` | Pause or resume, whichever applies. |
| `finish()` | Flush the tail and resolve after the final text arrives. |
| `cancel()` | Drop the session and emit an empty final text. |
| `reset()` | Alias for `cancel`. |
| `active()` | True while the controller owns the UI. |
| `state()` | Current `DictationState`. |
| `destroy()` | Cancel and detach the controller. |

### Options

| Field | Description |
| ----- | ----------- |
| `workerUrl` | Prebuilt `dictation.worker.js` URL. Required unless `workerFactory` is set. |
| `workerFactory` | Build the worker yourself, for bundlers or tests. |
| `workerFallback` | Rebuild the worker from source when its script is blocked. Defaults to `true`. |
| `model` | Speech model, `ModelId.WhisperBase` by default. |
| `language` | Audio language, `Language.en` by default. |
| `quality` | Weight precision, `Quality.High` by default. |
| `threads` | Worker threads, `ThreadCount.Four` by default. |
| `timestamps` | Timestamp detail, `TimestampMode.none` by default. |
| `updateIntervalMs` | Live decode cadence, 1000 ms by default. |
| `partialIntervalMs` | Partial text poll cadence, 200 ms by default. |
| `onState` | Receives the state and `{ ratio, modelReady, permission }`. |
| `onText` | Receives live text and the final text. |
| `onFrequency` | Receives a reused `Uint8Array` of analyser bins once per animation frame. |
| `onError` | Receives `{ source, message }`, where source is `microphone`, `model` or `worker`. |
| `capture` | Advanced capture override. |

### States

`idle` is ready, `requesting` waits for permission, `recording` captures audio
and `paused` keeps the session open. `loading` means finish was pressed while
the model was still loading. `working` means finish was pressed and the final
decode is running. Both return to `idle` after `onText(text, true)`.

## Public API

The engine entry exports four functions and six enums.

| Export | Signature | Description |
| ------ | --------- | ----------- |
| `transcribe` | `transcribe(options?)` | Start a session and return it immediately. |
| `preloadVoice` | `preloadVoice(options?)` | Download and load a model ahead of time. |
| `loadModel` | `loadModel(options?)` | Download a model when needed and return it loaded. |
| `isModelCached` | `isModelCached(options?)` | True when every file of the model is stored. |
| `ModelId` | `SenseVoiceSmall`, `WhisperBase` | The supported model selection. |
| `Language` | every language the bundled models support, plus `auto` | The supported language selection. |
| `Quality` | `High`, `Medium`, `Low` | Weight precision, `High` (F16) by default. |
| `ThreadCount` | `Two`, `Four`, `Eight` | Worker threads, `Four` by default. |
| `TimestampMode` | `none`, `segment`, `word` | The timestamp detail selection. |
| `ProgressPhase` | `download`, `verify`, `decode` | The phase reported by a load. |

The `@degreesign/voice/dictation` entry exports `createDictation`,
`isDictationSupported`, the dictation enums and the same `ModelId`, `Language`,
`Quality`, `ThreadCount` and `TimestampMode` enums.

`ModelId.WhisperBase` covers two checkpoints. An explicit `Language.en` loads
the English only `openai/whisper-base.en`; every other language loads the
multilingual `openai/whisper-base`. Both share one enum value, one usage record
and one `meta.model`.

### transcribe

`transcribe` returns a `TranscribeSession` without waiting for anything.

| Member | Signature | Description |
| ------ | --------- | ----------- |
| `fullText` | `() => { text, msElapsed }` | Latest text and decode time, cheap and synchronous. |
| `push` | `(pcm) => void` | Append 16 kHz mono samples to a session with `microphone: false`. |
| `stop` | `() => Promise<Transcript>` | Stop capture, flush the tail, resolve. |
| `cancel` | `() => void` | Drop pending audio and resolve without a final decode. |
| `done` | `Promise<Transcript>` | Resolves when a finite source ends, `stop` or `cancel` is called. |
| `error` | `Error \| null` | First asynchronous failure. |

`transcribe` accepts `{ model?, pcm?, audioFilePath?, microphone?, language?,
quality?, timestamps?, lowLatency?, itn?, allowedDiskSpaceMB?,
updateIntervalMs?, threads?, onProgress? }`, which is `TranscribeBase`
combined with the source and download fields.

| Field | Type | Description |
| ----- | ---- | ----------- |
| `model` | `ModelId` | Model to use. Defaults to `ModelId.WhisperBase`. |
| `pcm` | `Float32Array` | 16 kHz mono samples. Highest priority source. |
| `audioFilePath` | `string \| Blob` | Browser URL or blob, or a Node filesystem path. |
| `microphone` | `boolean` | Capture the microphone. Defaults to `true`; `false` accepts `push` input. |
| `language` | `Language` | Language of the audio. `WhisperBase` loads English for `Language.en`; `Language.auto` detects it. |
| `quality` | `Quality` | Weight precision. `Quality.High` (F16) by default, `Medium` (Q8_0) or `Low` (Q4_K_M). |
| `timestamps` | `TimestampMode` | `TimestampMode.none`, `.segment` or `.word`. |
| `lowLatency` | `boolean` | Decode at the true length, without 30 second padding. Defaults to `true`. |
| `itn` | `boolean` | Request inverse text normalization output. |
| `allowedDiskSpaceMB` | `number` | Cache cap in MB. Defaults to `512`. |
| `updateIntervalMs` | `number` | Live session update cadence. Defaults to `1000`. |
| `weightsBaseUrl` | `string` | Weights host, defaults to `https://huggingface.co`. |
| `weightCacheMB` | `number` | F32 weight cache cap in MB. Defaults to the whole checkpoint, capped at `1024`. |
| `threads` | `ThreadCount` | Worker threads for the decode. Defaults to `ThreadCount.Four`. |
| `onProgress` | `(progress) => void` | Download progress across every file of the model. |

Timing in `msElapsed` and `meta.rtf` covers decode work only. Model download,
model load, audio decoding and waiting for audio sit outside the measurement.

### preloadVoice

`preloadVoice` accepts `{ model?, language?, quality?, allowedDiskSpaceMB?,
weightsBaseUrl?, weightCacheMB?, threads?, onProgress? }` and resolves once the
model is in memory. Later `transcribe` calls reuse it. Pass `language:
Language.en` to preload the English Whisper checkpoint; omit it to preload the
multilingual one.

`loadModel` takes the same model selection and returns the loaded `Model`, while
`isModelCached` reads the stored manifests only and never touches the network.
Together they let a caller show a cached badge, label the download size and
render one monotonic progress bar.

`DownloadProgress` reports `phase`, plus `loaded` and `total` summed over every
file of the checkpoint, so `ratio` never restarts at zero between files. The
`file` field names the file currently moving.

```ts
const cached = await isModelCached({ model: ModelId.WhisperBase });
const model = await loadModel({
  model: ModelId.WhisperBase,
  onProgress: ({ phase, ratio }) => {
    console.log(phase, Math.round(ratio * 100));
  },
});
```

### Weights mirror

Set `weightsBaseUrl` to serve the checkpoints from your own host:

```ts
await preloadVoice({
  model: ModelId.WhisperBase,
  weightsBaseUrl: "https://models.example.com",
});
```

The mirror keeps the repository layout and serves every request below it:
`<base>/<repo>/resolve/main/<file>`. Whisper keeps its GGUF weights in the
`handy-computer` repositories and its text files in the original `openai`
repositories, so both are requested. Send
`Cache-Control: public, max-age=31536000, immutable` and support `Range`
requests to enable resumable, parallel downloads.

### Types

| Type | Fields | Description |
| ---- | ------ | ----------- |
| `Transcript` | `text`, `language`, `duration`, `segments`, `meta` | Final output contract. |
| `Segment` | `start`, `end`, `text`, `words` | Sentence level segment, times in seconds. |
| `Word` | `word`, `start`, `end` | One word with times in seconds. |
| `TextSnapshot` | `text`, `msElapsed` | One live reading, with decode time in milliseconds. |
| `DownloadProgress` | `model`, `file`, `phase`, `loaded`, `total`, `ratio` | Load progress across every file. |
| `TranscribeBase` | `pcm`, `language`, `timestamps`, `lowLatency`, `itn` | Decode parameters shared by every request. |
| `ModelOptionsBase` | `model`, `language`, `quality`, `weightsBaseUrl`, `weightCacheMB`, `threads`, `onProgress` | Model selection shared by every load. |
| `DiskBudgetOptions` | `model`, `language`, `quality`, `weightsBaseUrl`, `weightCacheMB`, `threads`, `onProgress`, `allowedDiskSpaceMB` | Model selection under a cache cap. |

## Models

| `ModelId` | Family | Languages |
| --------- | ------ | --------- |
| `SenseVoiceSmall` | sensevoice | zh, yue, en, ja, ko |
| `WhisperBase` | whisper | multilingual, English |

Whisper base is the default. It is the autoregressive encoder and decoder with
text, segment and word timestamps, DTW word alignment and long-form chunking.
The Whisper selection uses `handy-computer/whisper-base.en-gguf` with the
`openai/whisper-base.en` text files for an explicit `Language.en`, and
`handy-computer/whisper-base-gguf` with the `openai/whisper-base` text files for
every other language, including automatic detection. SenseVoice small is
available with `model: ModelId.SenseVoiceSmall`. It is a non-autoregressive
SAN-M encoder with a CTC head, a kaldi fbank plus LFR frontend and a
SentencePiece vocabulary.

### Weight quality

Every checkpoint is published as F16, Q8_0 and Q4_K_M. `Quality` selects the
weight file. `Quality.High` is the default.

| `Quality` | Weights | Whisper base | Whisper base.en | SenseVoiceSmall |
| --------- | ------- | ------------ | --------------- | --------------- |
| `High` | F16 | 151 MB | 151 MB | 470 MB |
| `Medium` | Q8_0 | 85 MB | 85 MB | 253 MB |
| `Low` | Q4_K_M | 59 MB | 59 MB | 146 MB |

All three formats are dequantized to F32 on first access before the kernels
run, so the transcription kernels are the same at every quality. Quality
changes the download size and the first-window dequantization cost, not the
steady-state decode path. `Quality.Low` has the smallest download and
`Quality.High` keeps the most precision.

## Audio input

The first source that is present wins:

1. `pcm`, already 16 kHz mono.
2. `audioFilePath`, decoded and resampled to 16 kHz.
3. A manual feed with `microphone: false`, appended through `push`.
4. The browser microphone, recorded until `stop`.

In Node a path is read from the filesystem and decoded by magic bytes. WAV,
FLAC and MP3 are supported by bundled decoders. In the browser a string is
fetched and decoded with the Web Audio API, which adds OGG, WebM and AAC
support. A local browser file must be wrapped in an object URL first.

The microphone uses `getUserMedia`, an `AudioWorklet` capture with a
`ScriptProcessor` fallback, and needs a secure context plus a user gesture.
Decoding blocks the calling thread, so a browser page that must stay responsive
captures audio itself and runs the session in a worker with `microphone: false`.

Streaming keeps two regions of text: committed text that will not change, and a
partial tail that is refreshed. Pending audio is committed at silence
boundaries, or forced at 30 seconds. Committed audio is dropped, so memory
stays bounded while the text keeps growing. For finite inputs the chunks are
drained as fast as inference allows, so a 60 second file does not wait 60
seconds. Only a live session uses wall clock updates.

## Disk budget

`allowedDiskSpaceMB` caps the total bytes this package stores. Before a download
the package estimates what the model needs. When the current cache plus the new
model would exceed the cap, other models are deleted least recently used first.
A model already in the cache is never pruned. When one model alone exceeds the
cap it is still loaded, and every other model is removed to make room. The
default F16 checkpoint is 151 MB for Whisper base and 470 MB for SenseVoice
small; `Quality.Low` reduces those to 59 MB and 146 MB.

## Storage

The model files are cached per platform:

- Node.js uses `@degreesign/cache` to store the files under `~/.cache/decoder`.
- Browsers use `@degreesign/storage` to store the files in IndexedDB.

Weight files are stored as fixed size binary chunks with a small manifest, and
text files as plain JSON. On browsers the chunks live in IndexedDB as binary
records, so a 146 MB model occupies about 146 MB instead of base64 text. A
`usage.json` record tracks the size and last use of each model for the disk
budget. The two libraries are bundled into the output, so the published package
has no runtime dependencies.

A weight download requests four chunks at a time with HTTP `Range` headers and
records each completed chunk in the manifest. An interrupted download resumes
from the stored chunks, and a failed chunk is retried with backoff before the
whole file is retried as one stream. A final size check still removes the file
when it does not match `weightBytes`.

## Performance

The graph runs on the CPU with WebAssembly SIMD kernels. `threads` defaults to
`ThreadCount.Four`; use `ThreadCount.Two` or `ThreadCount.Eight` to change the
worker count. Node.js can use worker threads directly; browsers need cross
origin isolation, served with `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp`, and the session should run inside
a worker because the split wait blocks its thread. Without isolation, or with a
single worker, the engine uses the single thread path. On a 3.1 GHz eight thread
i7, an 11 second English clip decodes in about 6 seconds on Whisper Base F16
with `threads: 4`, close to the Q4_K_M time on the same machine.

The default F16 weights are dequantized into a byte capped F32 cache on first
access. By default that cache holds the whole checkpoint once, so a window
never pays for the same tensor twice; pass `weightCacheMB` to keep the resident
set smaller at the cost of re-dequantization. Q8_0 and Q4_K_M checkpoints use
the same cache and the same F32 kernels, so the steady-state transcription path
does not change with quality. The arena behind the kernels reuses freed
regions, so a long running session keeps a steady footprint instead of growing
with every window.

Updates arrive as fast as inference allows. Inference blocks the calling
thread, so in the browser run the session inside a worker for a responsive UI.
A session commits its pending audio at a pause, at 12 seconds without one, or
when stopped, and decodes the whole pending buffer on every update. Raise
`updateIntervalMs` when the pending window is long and the decode cannot keep
up with the update cadence.

## FAQ

### What is @degreesign/voice?

A self contained TypeScript package that runs the SenseVoice and Whisper
speech recognition models locally on CPU.

### Is it free?

Yes. The runtime code is MIT licensed. The bundled decoders and the model
weights keep their own licenses, listed under [Licensing](#licensing).

### Does it run in the browser and Node.js?

Yes. One entry autodetects the runtime and selects a filesystem or IndexedDB
store, and webpack emits a CommonJS bundle, a full UMD bundle and one UMD
bundle per model family.

### Does it need Python or ONNX Runtime?

No. The graph, the frontends and the tokenizers run locally in TypeScript and
WebAssembly.

### Does it have runtime dependencies?

No. The cache, storage and MP3 decoder are bundled into the output.

### Where are models stored?

Node.js caches under `~/.cache/decoder` through `@degreesign/cache`; browsers
cache in IndexedDB through `@degreesign/storage`.

## Licensing

Whisper weights and tokenizer data are Apache-2.0, and the F16, Q8_0 and
Q4_K_M GGUF conversions keep that license. The SenseVoice F16, Q8_0 and Q4_K_M
weights are distributed under the FunASR Model Open Source License Agreement
v1.1. The bundled MP3 decoder is `minimp3`, which is CC0, together with the
Apache-2.0 `minimp3-wasm` bindings. See `THIRD_PARTIES.md`. The runtime code in
this folder is original work, licensed under MIT. See `LICENSE`.

## Keywords

`@degreesign/voice`, offline speech recognition, speech to text, speech
recognition, transcription, ASR, Whisper, SenseVoice, CTC, local inference,
on device, streaming, microphone, browser, Node.js, TypeScript, no Python,
no ONNX Runtime.
