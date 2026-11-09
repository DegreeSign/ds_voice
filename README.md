# @degreesign/voice

Turn speech into text and text into speech, privately, on the user's own
device. Use it for dictation, voice notes, transcription, speech to text, text
to speech and TTS, with no cloud service, account or network connection after
setup.

[![npm version](https://img.shields.io/npm/v/@degreesign/voice.svg)](https://www.npmjs.com/package/@degreesign/voice)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](#licensing)
[![node](https://img.shields.io/badge/node-%3E%3D24-brightgreen.svg)](#install)
[![types](https://img.shields.io/badge/types-included-blue.svg)](#quick-start)

## Table of contents

- [Why](#why)
- [Install](#install)
- [Quick start](#quick-start)
- [Speech to text](#speech-to-text)
- [Text to speech](#text-to-speech)
- [Dictation](#dictation)
- [Demo](#demo)
- [Models](#models)
- [Storage and performance](#storage-and-performance)
- [Architecture](#architecture)
- [Licensing](#licensing)

## Why

- Offline: weights download once, then every decode runs on the machine.
- Streaming: `transcribe` returns a live handle and keeps updating while audio
  arrives.
- One API: the same entry autodetects Node.js and the browser.
- Any audio: 16 kHz mono PCM, a `Blob` or `File`, a URL, a filesystem path, or
  the browser microphone.
- Typed: TypeScript declarations cover every export.
- Two directions: speech to text and text to speech.
- Self contained: the cache, storage and audio decoders are bundled at build
  time.

[DoListy](https://dolisty.com) uses the package for task dictation and voice
notes.

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

Node.js 24 or newer is required. The main entry exports recognition and the
voice library. The dictation controller has its own entry.

The published files are:

- `dist/node/degreesign.node.min.js`, the CommonJS entry for Node.
- `dist/browser/degreesign.min.js`, the full UMD entry as `window.dsVoice`.
- `dist/browser/whisper.min.js`, the Whisper only UMD entry as
  `window.dsVoiceWhisper`.
- `dist/browser/dictation.min.js`, the dictation UMD entry.
- `dist/browser/dictation.worker.js`, the prebuilt dictation worker.

Build from source with `yarn build` and typecheck with `yarn typecheck`.

### CDN

```html
<script src="https://cdn.jsdelivr.net/npm/@degreesign/voice/dist/browser/degreesign.min.js"></script>
```

Try the [live speech demo](https://degreesign.github.io/ds_voice/demo.html)
before installing. It runs speech to text and text to speech in the browser.

## Quick start

### Browser microphone

```html
<script src="https://cdn.jsdelivr.net/npm/@degreesign/voice/dist/browser/degreesign.min.js"></script>
<script>
  const { Language, transcribe } = window.dsVoice;
  const session = transcribe({ language: Language.auto });
  const timer = setInterval(() => {
    output.textContent = session.fullText().text;
  }, 500);
  stopButton.onclick = async () => {
    clearInterval(timer);
    const result = await session.stop();
    output.textContent = result.text;
  };
</script>
```

A browser session with no `pcm`, no `audioFilePath` and no manual feed asks for
microphone access and records until `stop`.

### PCM input

The input is 16 kHz mono `Float32Array` PCM.

```ts
import { Language, transcribe } from "@degreesign/voice";

const session = transcribe({ pcm, language: Language.auto });
console.log(session.fullText().text);

const result = await session.done;
console.log(result.text, result.duration, result.meta.rtf);
```

To warm up the model first, call `preloadVoice`. To feed audio as it arrives,
pass `microphone: false` and append samples with `push`.

## Speech to text

`transcribe(options?)` returns a `TranscribeSession` immediately.

| Member | Description |
| ------ | ----------- |
| `fullText()` | Latest text and decode time. Cheap and synchronous. |
| `push(pcm)` | Append 16 kHz mono samples with `microphone: false`. |
| `stop()` | Stop capture or input, flush the tail and resolve. |
| `cancel()` | Drop pending audio and resolve without a final decode. |
| `done` | Resolves after the input ends, `stop` or `cancel`. |
| `error` | First asynchronous failure, or `null`. |

### Options

| Field | Description |
| ----- | ----------- |
| `model` | `ModelId.WhisperBase`, the only model. |
| `pcm` | 16 kHz mono `Float32Array`, highest priority source. |
| `audioFilePath` | URL string, `Blob` or `File`, or a Node filesystem path. |
| `microphone` | Capture the microphone. Defaults to `true`; `false` accepts `push`. |
| `language` | Audio language or `Language.auto`. Defaults to automatic detection. |
| `quality` | `Quality.High` (F16) by default, `Medium` (Q8_0) or `Low` (Q4_K_M). |
| `timestamps` | `TimestampMode.none`, `.segment` or `.word`. |
| `lowLatency` | Skip 30 second padding. Defaults to `true`. |
| `allowedDiskSpaceMB` | Cache cap in MB. Defaults to `512`. |
| `updateIntervalMs` | Live update cadence in ms. Defaults to `1000`. |
| `weightsBaseUrl` | Weights host, defaults to `https://huggingface.co`. |
| `weightCacheMB` | F32 tensor cache cap in MB. |
| `threads` | `ThreadCount.Two`, `.Four` or `.Eight`. Defaults to `Four`. |
| `onProgress` | Download and load progress across every file. |

### Model loading

```ts
import {
    Language,
    ModelId,
    Quality,
    isModelCached,
    loadModel,
    preloadVoice,
} from "@degreesign/voice";

await preloadVoice({
    model: ModelId.WhisperBase,
    language: Language.en,
    quality: Quality.Low,
});

const cached = await isModelCached({
    model: ModelId.WhisperBase,
    language: Language.en,
    quality: Quality.Low,
});
const model = await loadModel({
    model: ModelId.WhisperBase,
    language: Language.en,
    quality: Quality.Low,
});
```

`preloadVoice` downloads and loads a model ahead of time. `loadModel` returns
the loaded model. `isModelCached` checks the store without using the network.
An explicit `Language.en` selects the English only checkpoint; every other
language, including `Language.auto`, uses the multilingual checkpoint.

### Audio and timing

The first source that is present wins: `pcm`, then `audioFilePath`, then a
manual feed with `microphone: false`, then the browser microphone.

In Node, paths are read from the filesystem and decoded as WAV, FLAC or MP3.
In the browser, a URL is fetched, a `Blob` or `File` is read directly, and
decoding uses the Web Audio API. Supported browser formats depend on the
browser.

Audio below `-50 dBFS` is treated as silence and returns empty text without
touching the model. Audible input shorter than one second is centred in
silence, so a clipped word keeps context on both sides.

A session commits pending audio at a pause, after 12 seconds without one, or
when stopped. Committed text no longer changes and its audio is released, so
memory stays bounded during long recordings. `msElapsed` and `meta.rtf` cover
decode work only, not download, load, audio decoding or waiting for audio.

### Weights mirror

Set `weightsBaseUrl` to serve checkpoints from your own host:

```ts
await preloadVoice({
    weightsBaseUrl: "https://models.example.com",
});
```

The mirror keeps the Hugging Face layout:
`<base>/<repo>/resolve/main/<file>`. Whisper reads GGUF weights from the
`handy-computer` repositories and text files from the `openai` repositories.
Support `Range` requests to enable resumable, parallel downloads.

## Text to speech

The voice library has two 48 kHz voices. Both use MOSS-TTS-Nano 100M and clone
bundled reference codes, so no reference recording is downloaded. Each voice
takes about 133 MB on first use. A 16 kHz preview clip is already bundled.

| Voice | Id | Model | Output |
| ----- | -- | ----- | ------ |
| Female | `VoiceId.Female` | MOSS-TTS-Nano 100M | 48 kHz |
| Male | `VoiceId.Male` | MOSS-TTS-Nano 100M | 48 kHz |

The text itself selects one of the 20 MOSS languages: Arabic, Chinese, Czech,
Danish, English, French, German, Greek, Hebrew, Hungarian, Italian, Japanese,
Korean, Persian, Polish, Portuguese, Russian, Spanish, Swedish and Turkish.

```ts
import {
    VoiceId,
    getVoiceSample,
    isVoiceCached,
    listVoices,
    loadVoice,
    removeVoice,
    synthesize,
} from "@degreesign/voice";

const voices = listVoices();
const sample = getVoiceSample(VoiceId.Female);
console.log(voices.map((voice) => voice.label), sample.sampleRate);

if (!(await isVoiceCached(VoiceId.Female))) {
    await loadVoice({
        voice: VoiceId.Female,
        onProgress: ({ phase, ratio }) => console.log(phase, ratio),
    });
}

const audio = await synthesize({
    voice: VoiceId.Female,
    text: "Hello from @degreesign/voice.",
});
// audio.pcm is mono Float32 at audio.sampleRate.

await removeVoice(VoiceId.Female);
```

| Function | Description |
| -------- | ----------- |
| `listVoices()` | Every voice spec, in library order. |
| `getVoiceSpec(voice)` | One spec with files, license and preview line. |
| `getVoiceSample(voice)` | Bundled WAV bytes and clip rate. |
| `isVoiceCached(voice)` | True when every voice file is stored. |
| `loadVoice(options)` | Download the voice files. Accepts `weightsBaseUrl`. |
| `prepareVoice(options)` | Download and parse the voice without speaking. |
| `synthesize(options)` | Render text and return `VoiceAudio`. |
| `removeVoice(voice)` | Remove the cached voice files. |

`synthesize` accepts optional `pitch` in semitones, `rate` and linear `gain`.
Pitch and rate apply after synthesis for every voice. The MOSS voices also
accept `seed` and `temperature`. Gain defaults to two, pitch to zero and rate
to one. The parsed weights stay resident until another voice replaces them.

## Dictation

`@degreesign/voice/dictation` is a headless controller for live microphone
dictation. It owns the worker, capture, pause state, model progress and partial
text. The page keeps its own markup and copy.

Copy the prebuilt worker next to the page, or serve it from a same origin path:

```bash
cp node_modules/@degreesign/voice/dist/browser/dictation.worker.js \
  public/dictation.worker.js
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
    onText: (text) => {
        input.value = text;
    },
    onError: ({ source, message }) => console.error(source, message),
});

micButton.onclick = () => void dictation.start();
pauseButton.onclick = () => dictation.togglePause();
doneButton.onclick = () => void dictation.finish();
cancelButton.onclick = () => dictation.cancel();
```

Main options are `workerUrl` or `workerFactory`, `model`, `language`,
`quality`, `threads`, `timestamps`, `updateIntervalMs`, `partialIntervalMs`,
`onState`, `onText`, `onFrequency`, `onError` and an advanced `capture`
override. One controller belongs to one input or component. Controllers on a
page share one worker, and starting one cancels the active one. A cancel emits
`onText("", true)` so the page can restore its base text.

The states are `idle`, `requesting`, `recording`, `paused`, `loading` and
`working`. The controller methods are `start`, `pause`, `resume`,
`togglePause`, `finish`, `cancel`, `reset`, `active`, `state` and `destroy`.

## Demo

Open the [live demo](https://degreesign.github.io/ds_voice/demo.html) to try
both directions in a browser before installing. It loads `@degreesign/voice`
from jsDelivr and lets you:

- Record from the microphone or pick an audio file and read the transcript.
- Pick a voice, tune pitch, rate, gain, temperature and seed, then listen.

The demo for speech to text uses `Quality.Low`, a 59 MB download. The demo
source is [docs/demo.html](docs/demo.html). Run
`node test/example/serve.cjs`, then open
`http://127.0.0.1:17544/docs/demo.html` to serve it locally.

## Models

| Id | Checkpoint | Languages |
| -- | ---------- | --------- |
| `ModelId.WhisperBase` | Whisper base | multilingual |
| `ModelId.WhisperBase` + `Language.en` | Whisper base.en | English |

Whisper base uses `handy-computer/whisper-base-gguf` with the
`openai/whisper-base` text files. The English checkpoint uses
`handy-computer/whisper-base.en-gguf` with `openai/whisper-base.en`. Both share
one `ModelId`, one usage record and one `meta.model`.

Weights are published as F16, Q8_0 and Q4_K_M. `Quality.High` is the default.

| `Quality` | Weights | Whisper base | Whisper base.en |
| --------- | ------- | ------------ | --------------- |
| `High` | F16 | 151 MB | 151 MB |
| `Medium` | Q8_0 | 85 MB | 85 MB |
| `Low` | Q4_K_M | 59 MB | 59 MB |

Quantized files are dequantized to F32 on first access. The kernels are the
same at every quality, so quality changes download size and first access cost,
not the steady state decode path.

## Storage and performance

Node.js stores model files through `@degreesign/cache` under
`~/.cache/decoder`. Browsers store them in IndexedDB through
`@degreesign/storage`.

Weights are stored as 4 MiB chunks with a manifest, text files as JSON, and
`usage.json` tracks size and last use for the disk budget. A download uses four
parallel `Range` requests and resumes from stored chunks. A failed chunk is
retried with backoff before the whole file falls back to one stream.

`allowedDiskSpaceMB` caps stored model bytes and defaults to 512. A model that
is already cached is never pruned.

The graph runs on the CPU with WebAssembly SIMD kernels. `threads` defaults to
`ThreadCount.Four`. Browsers need cross origin isolation with
`Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp` to use the worker pool; without
it the engine uses the single thread path. The F32 tensor cache holds the
checkpoint once by default; `weightCacheMB` lowers the cap.

## Architecture

The committed map at [docs/architecture.html](docs/architecture.html) shows the
files and functions as bands, from the public entry through the model family,
the tensor runtime, the platform stores and the audio input. It is one offline
HTML file. Open it directly and use the panel to trace a call.

## Licensing

The runtime code in this repository is MIT licensed. Whisper weights and token
data are Apache-2.0. The MOSS TTS weights, tokenizer and reference codes are
Apache-2.0. The bundled MP3 decoder is CC0 and the `minimp3-wasm` bindings are
Apache-2.0. See [THIRD_PARTIES.md](THIRD_PARTIES.md) and [LICENSE](LICENSE).
