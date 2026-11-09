# Third Parties

`@degreesign/voice` is an independent TypeScript runtime for the SenseVoice,
Whisper and text to speech models listed below. It loads published model
weights at runtime. The only upstream data it redistributes are the bundled
MOSS reference codes and generated preview clips listed below.

The runtime code in this repository is original work licensed under the MIT
License. See [LICENSE](LICENSE).

## Bundled code

### minimp3

- License: Creative Commons Zero v1.0 Universal (CC0-1.0)
- Source: https://github.com/lieff/minimp3
- Contents: the MP3 decoder compiled to WebAssembly and inlined in the bundle.

### minimp3-wasm

- License: Apache License 2.0
- Source: https://github.com/bashi/minimp3-wasm
- Contents: the WebAssembly build of minimp3, the wasm binary and its bindings.

## Model weights

### openai/whisper-base

- License: Apache License 2.0
- Source: https://huggingface.co/openai/whisper-base
- Contents: the official OpenAI Whisper base checkpoint, including the
  tokenizer data and configuration files from the same repository.

### openai/whisper-base.en

- License: Apache License 2.0
- Source: https://huggingface.co/openai/whisper-base.en
- Contents: the official OpenAI Whisper base English checkpoint, including the
  tokenizer data and configuration files from the same repository.

### handy-computer/whisper-base-gguf

- License: Apache License 2.0
- Sources: https://huggingface.co/handy-computer/whisper-base-gguf
  and https://huggingface.co/openai/whisper-base
- Contents: the F16, Q8_0 and Q4_K_M GGUF conversions of the OpenAI Whisper base
  checkpoint. The tokenizer data and configuration files are fetched from the
  original `openai/whisper-base` repository.

### handy-computer/whisper-base.en-gguf

- License: Apache License 2.0
- Sources: https://huggingface.co/handy-computer/whisper-base.en-gguf
  and https://huggingface.co/openai/whisper-base.en
- Contents: the F16, Q8_0 and Q4_K_M GGUF conversions of the OpenAI Whisper base
  English checkpoint. The tokenizer data and configuration files are fetched
  from the original `openai/whisper-base.en` repository.

### handy-computer/SenseVoiceSmall-gguf

- License: FunASR Model Open Source License Agreement v1.1
- Sources: https://huggingface.co/handy-computer/SenseVoiceSmall-gguf
  and https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE
- Contents: the F16, Q8_0 and Q4_K_M GGUF conversions of
  FunAudioLLM/SenseVoiceSmall by Alibaba Group / FunAudioLLM. The tokenizer and
  configuration are embedded in the same file.

### Luigi/MOSS-TTS-Nano-100M-GGUF

- License: Apache License 2.0
- Sources: https://huggingface.co/Luigi/MOSS-TTS-Nano-100M-GGUF,
  https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M and
  https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano
- Contents: the GGUF conversion of the MOSS-TTS-Nano 100M language model and
  the MOSS-Audio-Tokenizer-Nano decoder used by the `male` and `female`
  voices.

### aoiandroid/moss-tts-nano-onnx-fp32

- License: Apache License 2.0
- Sources: https://huggingface.co/aoiandroid/moss-tts-nano-onnx-fp32 and
  https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M
- Contents: the `tokenizer.json` conversion of the SentencePiece tokenizer used
  by the `male` and `female` voices.

### Bundled MOSS reference codes

- License: Apache License 2.0
- Sources: the `male` and `female` voices are conditioned on reference clips
  rendered with Kokoro-82M. The `male` clip uses `cstr/kokoro-82m-GGUF` with
  the `am_michael` speaker. The `female` clip uses `cstr/kokoro-82m-GGUF`
  with the `af_heart` speaker pack from `cstr/kokoro-voices-GGUF`. Both derive
  from https://huggingface.co/hexgrad/Kokoro-82M.
- Contents: the MOSS audio tokenizer codes of those clips, used to keep the
  `male` and `female` voices consistent without downloading a reference
  recording.

### Bundled voice previews

The two preview clips under `src/assets/voices` were synthesized from the
models above with the fixed line held in `src/speech/voices.ts`. They contain
no upstream audio recordings and are generated output of the listed weights.

## Attribution

Each weight set remains the property of its authors and is governed by the
license listed above. The licenses are granted by the original authors, not by
DegreeSign, and no affiliation with or endorsement by those authors is implied.
