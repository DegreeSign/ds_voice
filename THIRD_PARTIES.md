# Third Parties

`@degreesign/voice` is an independent TypeScript runtime for the SenseVoice and
Whisper speech recognition models. It loads published model weights at runtime
and ships no code, weights or data from the upstream projects.

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

### handy-computer/SenseVoiceSmall-gguf

- License: FunASR Model Open Source License Agreement v1.1
- Sources: https://huggingface.co/handy-computer/SenseVoiceSmall-gguf
  and https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE
- Contents: the Q8_0 GGUF conversion of FunAudioLLM/SenseVoiceSmall by Alibaba
  Group / FunAudioLLM. The tokenizer and configuration are embedded in the same
  file.

## Attribution

Each weight set remains the property of its authors and is governed by the
license listed above. The licenses are granted by the original authors, not by
DegreeSign, and no affiliation with or endorsement by those authors is implied.
