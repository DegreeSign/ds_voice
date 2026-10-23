// Shared runtime types.

/** Raw model configuration from config.json. */
export interface RawModelConfig {
    d_model: number;
    encoder_layers: number;
    decoder_layers: number;
    encoder_attention_heads: number;
    decoder_attention_heads: number;
    encoder_ffn_dim: number;
    decoder_ffn_dim: number;
    vocab_size: number;
    num_mel_bins: number;
    max_source_positions: number;
    max_target_positions: number;
    max_length: number;
    activation_function: string;
    scale_embedding: boolean;
    decoder_start_token_id: number;
    bos_token_id: number;
    eos_token_id: number;
    pad_token_id: number;
    suppress_tokens: number[];
    begin_suppress_tokens: number[];
    forced_decoder_ids: Array<[number, number | null]>;
}

/** Raw generation configuration from generation_config.json. */
export interface RawGenerationConfig {
    alignment_heads: Array<[number, number]>;
    begin_suppress_tokens: number[];
    bos_token_id: number;
    decoder_start_token_id: number;
    eos_token_id: number;
    forced_decoder_ids: Array<[number, number | null]>;
    is_multilingual: boolean;
    lang_to_id?: Record<string, number>;
    max_initial_timestamp_index: number;
    max_length: number;
    no_timestamps_token_id: number;
    pad_token_id: number;
    prev_sot_token_id: number;
    suppress_tokens: number[];
    task_to_id?: Record<string, number>;
}

/** Raw preprocessor configuration from preprocessor_config.json. */
export interface RawPreprocessorConfig {
    chunk_length: number;
    feature_size: number;
    hop_length: number;
    mel_filters: number[][];
    n_fft: number;
    n_samples: number;
    nb_max_frames: number;
    padding_side: string;
    padding_value: number;
    sampling_rate: number;
}

/** Normalized model shape used by the graph. */
export interface ModelConfig {
    dModel: number;
    encoderLayers: number;
    decoderLayers: number;
    encoderHeads: number;
    decoderHeads: number;
    encoderFfnDim: number;
    decoderFfnDim: number;
    vocabSize: number;
    numMelBins: number;
    headDim: number;
    maxSourcePositions: number;
    maxTargetPositions: number;
    maxLength: number;
    scaleEmbedding: boolean;
}

/** Token and generation constants. */
export interface Constants {
    eosTokenId: number;
    bosTokenId: number;
    padTokenId: number;
    decoderStartTokenId: number;
    isMultilingual: boolean;
    startOfTranscript: number;
    languageIds: Map<string, number>;
    idToLanguage: Map<string, string>;
    translateToken: number;
    transcribeToken: number;
    startOfLm: number;
    startOfPrev: number;
    noSpeechToken: number;
    noTimestampsToken: number;
    timestampBegin: number;
    timestampStep: number;
    suppressTokens: number[];
    beginSuppressTokens: number[];
    alignmentHeads: Array<[number, number]>;
    maxInitialTimestampIndex: number;
}

/** Normalized feature extractor settings. */
export interface MelConfig {
    samplingRate: number;
    nFft: number;
    hopLength: number;
    featureSize: number;
    chunkLength: number;
    nSamples: number;
    maxFrames: number;
}

/** Header entry for one tensor. */
export interface TensorInfo {
    name: string;
    dtype: string;
    shape: number[];
    dataOffset: number;
    numel: number;
}

/** Read only view over parsed safetensors. */
export interface TensorStore {
    names(): string[];
    has(name: string): boolean;
    info(name: string): TensorInfo;
    get(name: string): Float32Array;
    readonly count: number;
}

/** Parsed safetensors file and its backing buffer. */
export interface SafetensorsFile {
    store: TensorStore;
    buffer: Uint8Array;
    metadata: Record<string, string>;
}

/** Options for parsing a safetensors file. */
export interface ParseSafetensorsOptions {
    bytes: Uint8Array;
}

/** Model selection exposed by the package. */
export enum ModelId {
    /** SenseVoice small, 5 languages, the default. */
    SenseVoiceSmall = "sensevoice-small",
    /** Whisper base, English uses the English only checkpoint. */
    WhisperBase = "whisper-base",
}

/** Concrete checkpoint behind a public model selection. */
export type CheckpointId = ModelId | "whisper-base.en";

/** Architecture family behind a selection. */
export enum ModelFamily {
    /** Autoregressive encoder and decoder. */
    Whisper = "whisper",
    /** Non-autoregressive SAN-M encoder with a CTC head. */
    SenseVoice = "sensevoice",
}

/** Languages supported by the bundled models, `auto` detects one. */
export enum Language {
    /** Detect the language from the audio. */
    auto = "auto",
    /** Afrikaans. */
    af = "af",
    /** Amharic. */
    am = "am",
    /** Arabic. */
    ar = "ar",
    /** Assamese. */
    as = "as",
    /** Azerbaijani. */
    az = "az",
    /** Bashkir. */
    ba = "ba",
    /** Belarusian. */
    be = "be",
    /** Bulgarian. */
    bg = "bg",
    /** Bengali. */
    bn = "bn",
    /** Tibetan. */
    bo = "bo",
    /** Breton. */
    br = "br",
    /** Bosnian. */
    bs = "bs",
    /** Catalan. */
    ca = "ca",
    /** Czech. */
    cs = "cs",
    /** Welsh. */
    cy = "cy",
    /** Danish. */
    da = "da",
    /** German. */
    de = "de",
    /** Greek. */
    el = "el",
    /** English. */
    en = "en",
    /** Spanish. */
    es = "es",
    /** Estonian. */
    et = "et",
    /** Basque. */
    eu = "eu",
    /** Persian. */
    fa = "fa",
    /** Finnish. */
    fi = "fi",
    /** Faroese. */
    fo = "fo",
    /** French. */
    fr = "fr",
    /** Galician. */
    gl = "gl",
    /** Gujarati. */
    gu = "gu",
    /** Hawaiian. */
    haw = "haw",
    /** Hausa. */
    ha = "ha",
    /** Hebrew. */
    he = "he",
    /** Hindi. */
    hi = "hi",
    /** Croatian. */
    hr = "hr",
    /** Haitian Creole. */
    ht = "ht",
    /** Hungarian. */
    hu = "hu",
    /** Armenian. */
    hy = "hy",
    /** Indonesian. */
    id = "id",
    /** Icelandic. */
    is = "is",
    /** Italian. */
    it = "it",
    /** Japanese. */
    ja = "ja",
    /** Javanese. */
    jw = "jw",
    /** Georgian. */
    ka = "ka",
    /** Kazakh. */
    kk = "kk",
    /** Khmer. */
    km = "km",
    /** Kannada. */
    kn = "kn",
    /** Korean. */
    ko = "ko",
    /** Latin. */
    la = "la",
    /** Luxembourgish. */
    lb = "lb",
    /** Lingala. */
    ln = "ln",
    /** Lao. */
    lo = "lo",
    /** Lithuanian. */
    lt = "lt",
    /** Latvian. */
    lv = "lv",
    /** Malagasy. */
    mg = "mg",
    /** Maori. */
    mi = "mi",
    /** Macedonian. */
    mk = "mk",
    /** Malayalam. */
    ml = "ml",
    /** Mongolian. */
    mn = "mn",
    /** Marathi. */
    mr = "mr",
    /** Malay. */
    ms = "ms",
    /** Maltese. */
    mt = "mt",
    /** Myanmar. */
    my = "my",
    /** Nepali. */
    ne = "ne",
    /** Dutch. */
    nl = "nl",
    /** Nynorsk. */
    nn = "nn",
    /** Norwegian. */
    no = "no",
    /** Occitan. */
    oc = "oc",
    /** Punjabi. */
    pa = "pa",
    /** Polish. */
    pl = "pl",
    /** Pashto. */
    ps = "ps",
    /** Portuguese. */
    pt = "pt",
    /** Romanian. */
    ro = "ro",
    /** Russian. */
    ru = "ru",
    /** Sanskrit. */
    sa = "sa",
    /** Sindhi. */
    sd = "sd",
    /** Sinhala. */
    si = "si",
    /** Slovak. */
    sk = "sk",
    /** Slovenian. */
    sl = "sl",
    /** Shona. */
    sn = "sn",
    /** Somali. */
    so = "so",
    /** Albanian. */
    sq = "sq",
    /** Serbian. */
    sr = "sr",
    /** Sundanese. */
    su = "su",
    /** Swedish. */
    sv = "sv",
    /** Swahili. */
    sw = "sw",
    /** Tamil. */
    ta = "ta",
    /** Telugu. */
    te = "te",
    /** Tajik. */
    tg = "tg",
    /** Thai. */
    th = "th",
    /** Turkmen. */
    tk = "tk",
    /** Tagalog. */
    tl = "tl",
    /** Turkish. */
    tr = "tr",
    /** Tatar. */
    tt = "tt",
    /** Ukrainian. */
    uk = "uk",
    /** Urdu. */
    ur = "ur",
    /** Uzbek. */
    uz = "uz",
    /** Vietnamese. */
    vi = "vi",
    /** Yiddish. */
    yi = "yi",
    /** Yoruba. */
    yo = "yo",
    /** Cantonese. */
    yue = "yue",
    /** Chinese. */
    zh = "zh",
}

/** Chunk access used by resumable weight downloads. */
export interface ModelPartStore {
    /** Chunk sizes of a file, zero for a missing chunk, or null. */
    readSizes(key: string): Promise<number[] | null>;
    /** Record the chunk sizes, a zero marks a chunk still missing. */
    writeSizes(options: { key: string; sizes: number[] }): Promise<void>;
    /** Read one chunk, or null when absent. */
    read(options: {
        key: string;
        index: number;
    }): Promise<Uint8Array | null>;
    /** Write one chunk. */
    write(options: {
        key: string;
        index: number;
        bytes: Uint8Array;
    }): Promise<void>;
    /** Remove one chunk. */
    remove(options: { key: string; index: number }): Promise<void>;
}

/** Persistence backend for the cached model files. */
export interface ModelStore {
    /** True when a key, weights or text, is present. */
    exists(key: string): Promise<boolean>;
    /** Read a text file, or null when absent. */
    readText(key: string): Promise<string | null>;
    /** Read a binary file, or null when absent. */
    readBytes(key: string): Promise<Uint8Array | null>;
    /** True when a binary file can be read back in full. */
    hasBinary?(key: string): Promise<boolean>;
    /** Size of one stored key in bytes, when the backend reports it. */
    size?(key: string): Promise<number>;
    /** Write a text file. */
    writeText(options: { key: string; text: string }): Promise<void>;
    /** Write a binary file from a stream. */
    writeStream(options: {
        key: string;
        stream: ReadableStream<Uint8Array>;
        onProgress?: (loaded: number) => void;
    }): Promise<void>;
    /** Remove a key and every chunk stored with it. */
    remove(key: string): Promise<void>;
    /** Remove a raw key without touching its chunks, when the backend can. */
    removeRaw?(key: string): Promise<void>;
    /** Chunk access for resumable binary downloads. */
    parts?: ModelPartStore;
}

/** One fetchable model definition. */
export interface ModelSpec {
    /** Selection this spec describes. */
    id: ModelId;
    /** Cache key of this concrete checkpoint. */
    checkpoint: CheckpointId;
    /** Repository that holds the weight file, owner and name. */
    repo: string;
    /** Repository for the text files, defaults to repo. */
    textRepo?: string;
    /** Architecture family that loads the weights. */
    family: ModelFamily;
    /** Weight file name. */
    weightFile: string;
    /** Expected weight size in bytes, zero when unknown. */
    weightBytes: number;
    /** Every file fetched for this model. */
    files: string[];
}

/** Phase reported by the load progress. */
export enum ProgressPhase {
    /** Bytes are moving from the network into the store. */
    download = "download",
    /** The stored file is checked against its expected size. */
    verify = "verify",
    /** The stored weights are parsed and loaded. */
    decode = "decode",
}

/** Progress report across every file of one checkpoint. */
export interface DownloadProgress {
    /** Model being downloaded. */
    model: ModelId;
    /** File currently being downloaded. */
    file: string;
    /** Step the load is in. */
    phase: ProgressPhase;
    /** Bytes received so far, across every file. */
    loaded: number;
    /** Expected total bytes across every file. */
    total: number;
    /** Fraction of the total, zero when the total is unknown. */
    ratio: number;
}

/** Download progress callback. */
export type ProgressHandler = (progress: DownloadProgress) => void;

/** Model selection shared by every load entry point. */
export interface ModelOptionsBase {
    /** Model to use, defaults to the default model. */
    model?: ModelId;
    /** Language that chooses a checkpoint variant. */
    language?: Language;
    /** Override for the weights host, defaults to Hugging Face. */
    weightsBaseUrl?: string;
    /** F32 weight cache cap in MB, defaults to the checkpoint. */
    weightCacheMB?: number;
    /** Worker threads for the decode, defaults to 1. */
    threads?: number;
    /** Called across every file as it loads. */
    onProgress?: ProgressHandler;
}

/** Model selection under a cache cap. */
export interface DiskBudgetOptions extends ModelOptionsBase {
    /** Cache cap in MB, defaults to 512. */
    allowedDiskSpaceMB?: number;
}

/** Options for downloading a model into the store. */
export interface PrepareModelOptions extends ModelOptionsBase {}


/** Options for deleting a cached model. */
export interface DeleteModelOptions {
    model: ModelId;
}

/** Options for the Node filesystem store. */
export interface NodeStoreOptions {
    /** Cache folder, defaults under the user home. */
    cacheDir?: string;
}

/** Runtime the package is executing in. */
export enum Platform {
    Node = "node",
    Browser = "browser",
}

/** One cached model usage record. */
export interface ModelUsageEntry {
    bytes: number;
    lastUsed: number;
}

/** Persisted disk usage for the package. */
export interface ModelUsage {
    models: Partial<Record<ModelId, ModelUsageEntry>>;
}

/** Options for reading the persisted usage. */
export interface UsageStoreOptions {
    store: ModelStore;
}

/** Options for the disk budget check. */
export interface EnsureSpaceOptions {
    store: ModelStore;
    model: ModelId;
    /** Language that chooses a checkpoint variant. */
    language?: Language;
    allowedDiskSpaceMB: number;
}

/** Options for touching a cached model. */
export interface TouchModelOptions {
    store: ModelStore;
    model: ModelId;
}

/** Decoded mono audio and its sample rate. */
export interface DecodedAudio {
    pcm: Float32Array;
    samplingRate: number;
}

/** Options for parsing one audio buffer. */
export interface DecodeBytesOptions {
    bytes: Uint8Array;
    platform: Platform;
}

/** Options for decoding one audio source. */
export interface DecodeSourceOptions {
    audioFilePath: string | Blob;
    platform: Platform;
}

/** Options for resampling mono audio. */
export interface ResampleAudioOptions {
    input: Float32Array;
    inputRate: number;
    outputRate: number;
}

/** Options for downmixing interleaved channels. */
export interface DownmixOptions {
    input: Float32Array;
    channels: number;
}

/** Options for reading a RIFF WAVE buffer. */
export interface ReadWavOptions {
    bytes: Uint8Array;
}

/** Options for reading a FLAC buffer. */
export interface ReadFlacOptions {
    bytes: Uint8Array;
}

/** FLAC stream information. */
export interface FlacStreamInfo {
    sampleRate: number;
    channels: number;
    bitsPerSample: number;
    totalSamples: number;
}

/** Options for reading an MP3 buffer. */
export interface ReadMp3Options {
    bytes: Uint8Array;
}

/** Exported surface of the inline minimp3 wasm module. */
export interface Minimp3Exports {
    memory: WebAssembly.Memory;
    decoder_init(): void;
    decoder_set_mp3_data_size(size: number): void;
    decoder_mp3_data_offset(): number;
    decoder_mp3_data_size(): number;
    decoder_pcm_data_offset(): number;
    decoder_pcm_data_size(): number;
    decode_results_sampling_rate(): number;
    decode_results_num_channels(): number;
    decode_results_num_samples(): number;
    decoder_current_time(): number;
    decoder_seek(position: number): void;
    decoder_decode(duration: number): void;
}

/** One microphone capture handle. */
export interface MicrophoneCapture {
    /** Frames at the context sample rate. */
    onChunk: (samples: Float32Array) => void;
    /** First capture failure. */
    onError: (error: Error) => void;
}

/** Options for a streaming transcription session. */
export interface SessionOptions {
    options: TranscribeOptions;
    voice: Voice;
}

/** Bound voice functions for one store. */
export interface Voice {
    /** Download every file of a model into the store. */
    prepareModel(options?: PrepareModelOptions): Promise<ModelSpec>;
    /** Load a model, downloading it first when missing. */
    loadModel(options?: LoadModelOptions): Promise<Model>;
    /** Transcribe 16 kHz mono audio. */
    transcribe(options: TranscribeAudioOptions): Transcript;
    /** Remove a model and its files from the store. */
    deleteModel(options: DeleteModelOptions): Promise<void>;
    /** Apply the disk budget, download and load one model. */
    ensureModel(options?: EnsureModelOptions): Promise<Model>;
    /** True when every file of a model is already stored. */
    isModelCached(options?: ModelOptionsBase): Promise<boolean>;
}

/** Functions behind one model family. */
export interface VoiceFamily {
    /** Parse the stored weights of this family. */
    load(options: StoreModelOptions): Promise<Model>;
    /** Decode one buffer with this family. */
    transcribe(options: TranscribeAudioOptions): Transcript;
}

/** Model families bundled into one entry. */
export interface VoiceFamilies {
    [ModelFamily.SenseVoice]?: VoiceFamily;
    [ModelFamily.Whisper]?: VoiceFamily;
}

/** Options for building a voice over a family set. */
export interface CreateVoiceEngineOptions {
    families: VoiceFamilies;
    store?: ModelStore;
}

/** Public functions built over one voice. */
export interface VoiceApi {
    /** True when every file of a model is already stored. */
    isModelCached(options?: ModelOptionsBase): Promise<boolean>;
    /** Download and load a model ahead of time. */
    loadModel(options?: LoadModelOptions): Promise<Model>;
    /** Download and load a model ahead of time. */
    preloadVoice(options?: PreloadVoiceOptions): Promise<void>;
    /** Start a streaming transcription. */
    transcribe(options?: TranscribeOptions): TranscribeSession;
}

/** Tokenizer with the pieces needed for decoding. */
export interface Tokenizer {
    decode(ids: number[]): string;
    tokenText(id: number): string;
    vocabSize: number;
}

/** One word with times in seconds. */
export interface Word {
    /** Word text, leading and trailing space trimmed. */
    word: string;
    /** Start time in seconds from the audio start. */
    start: number;
    /** End time in seconds from the audio start. */
    end: number;
}

/** One sentence level segment with times in seconds. */
export interface Segment {
    /** Start time in seconds from the audio start. */
    start: number;
    /** End time in seconds from the audio start. */
    end: number;
    /** Transcribed text of the segment. */
    text: string;
    /** Words of the segment, empty when not requested. */
    words: Word[];
}

/** Transcript metadata. */
export interface TranscriptMeta {
    /** Model that produced the transcript. */
    model: string;
    /** Real time factor, decode seconds per audio second. */
    rtf: number;
    /** Worker threads used for the decode. */
    threads: number;
}

/** Output contract for transcriptions. */
export interface Transcript {
    /** Full transcript text. */
    text: string;
    /** Language of the audio, detected when not requested. */
    language: Language;
    /** Audio duration in seconds. */
    duration: number;
    /** Sentence level segments, empty for a silent input. */
    segments: Segment[];
    /** Model and timing metadata. */
    meta: TranscriptMeta;
}

/** Timestamp detail: text only, sentence times or per word times. */
export enum TimestampMode {
    /** Plain text with no timing detail. */
    none = "none",
    /** Sentence level start and end times. */
    segment = "segment",
    /** Per word start and end times. */
    word = "word",
}

/** One live text reading. */
export interface TextSnapshot {
    /** Transcript text available so far. */
    text: string;
    /** Decode milliseconds, without download or idle waits. */
    msElapsed: number;
}

/** Decode parameters shared by every transcription request. */
export interface TranscribeBase {
    /** 16 kHz mono samples, highest priority source. */
    pcm?: Float32Array;
    /** Language of the audio, detects when omitted. */
    language?: Language;
    /** Timestamp detail to request. */
    timestamps?: TimestampMode;
    /** Decode at the true audio length, without 30 second padding. */
    lowLatency?: boolean;
    /** Request inverse text normalization output. */
    itn?: boolean;
}

/** Options for a streaming transcription. */
export interface TranscribeOptions extends TranscribeBase, DiskBudgetOptions {
    /** Browser URL or blob, or a Node filesystem path. */
    audioFilePath?: string | Blob;
    /** Capture the microphone, defaults to true. */
    microphone?: boolean;
    /** Update cadence for live sessions, defaults to 1000 ms. */
    updateIntervalMs?: number;
}

/** Live handle over one streaming transcription. */
export interface TranscribeSession {
    /** Latest full text, cheap and synchronous. */
    fullText(): TextSnapshot;
    /** Append 16 kHz mono samples, manual source only. */
    push(pcm: Float32Array): void;
    /** Stop capture, flush the tail and resolve the transcript. */
    stop(): Promise<Transcript>;
    /** Drop pending audio and resolve without a final decode. */
    cancel(): void;
    /** Resolves when a finite source ends, stop or cancel is called. */
    done: Promise<Transcript>;
    /** First asynchronous failure, or null. */
    error: Error | null;
}

/** Options for preparing a model ahead of time. */
export interface PreloadVoiceOptions extends DiskBudgetOptions {}

/** Options for resolving a model inside the voice. */
export interface EnsureModelOptions extends DiskBudgetOptions {}

/** Options for a synchronous buffer transcription. */
export interface TranscribeAudioOptions extends TranscribeBase {
    /** Loaded model that runs the decode. */
    model: Model;
    /** 16 kHz mono samples to transcribe. */
    pcm: Float32Array;
}

/** Loaded model bundle passed to the graph. */
export interface LoadedModel {
    kind: ModelFamily.Whisper;
    name: ModelId;
    config: ModelConfig;
    constants: Constants;
    mel: MelConfig;
    melFilters: Float32Array;
    tokenizer: Tokenizer;
    tensors: TensorStore;
}

/** GGUF scalar, string or array metadata value. */
export type GgufValue =
    | boolean
    | number
    | bigint
    | string
    | GgufValue[];

/** One GGUF tensor header entry. */
export interface GgufTensorInfo {
    name: string;
    /** ggml dimensions, innermost first. */
    dims: number[];
    /** ggml type code, 0 F32, 1 F16, 8 Q8_0. */
    type: number;
    /** Byte offset from the tensor data section. */
    offset: number;
}

/** Parsed GGUF container. */
export interface GgufFile {
    version: number;
    metadata: Map<string, GgufValue>;
    tensors: Map<string, GgufTensorInfo>;
    /** Byte offset of the tensor data section. */
    dataOffset: number;
    buffer: Uint8Array;
}

/** Options for parsing a GGUF file. */
export interface ParseGgufOptions {
    bytes: Uint8Array;
}

/** Options for reading one GGUF tensor into f32. */
export interface ReadGgufTensorOptions {
    file: GgufFile;
    info: GgufTensorInfo;
}

/** Options for building a weight store over a GGUF file. */
export interface CreateGgufWeightsOptions {
    file: GgufFile;
    /** F32 cache cap in bytes, defaults to the documented budget. */
    cacheBytes?: number;
}

/** SenseVoice SAN-M encoder shape. */
export interface SenseVoiceConfig {
    dInput: number;
    dModel: number;
    heads: number;
    headDim: number;
    ffnDim: number;
    kernelSize: number;
    /** Main tier blocks, the projection block included. */
    mainBlocks: number;
    /** Second tier blocks after the inter-tier norm. */
    tpBlocks: number;
    vocabSize: number;
    /** LayerNorm epsilon, 1e-12 for this family. */
    eps: number;
    /** Prefix embedding row per language, auto included. */
    languageIds: Map<string, number>;
    /** Prefix embedding rows for the event and emotion tokens. */
    eventEmotion: [number, number];
    withItn: number;
    withoutItn: number;
}

/** SenseVoice kaldi fbank frontend settings. */
export interface SenseVoiceFrontend {
    samplingRate: number;
    numMels: number;
    nFft: number;
    winLength: number;
    hopLength: number;
    lfrM: number;
    lfrN: number;
    upscale: number;
    /** Per feature shift, dInput wide. */
    shift: Float32Array;
    /** Per feature scale, dInput wide. */
    scale: Float32Array;
}

/** Read only f32 view over GGUF weights. */
export interface SenseVoiceWeights {
    names(): string[];
    has(name: string): boolean;
    /** PyTorch style row-major shape, outermost first. */
    shape(name: string): number[];
    f32(name: string): Float32Array;
}

/** Loaded SenseVoice bundle. */
export interface LoadedSenseVoice {
    kind: ModelFamily.SenseVoice;
    name: ModelId;
    config: SenseVoiceConfig;
    frontend: SenseVoiceFrontend;
    tokenizer: Tokenizer;
    vocabulary: SenseVoiceVocabulary;
    weights: SenseVoiceWeights;
}

/** Loaded model of any family. */
export type Model = LoadedModel | LoadedSenseVoice;

/** Options for loading a model from a store. */
export interface StoreModelOptions {
    store: ModelStore;
    model: ModelId;
    /** Concrete checkpoint that holds the files. */
    checkpoint: CheckpointId;
    /** F32 weight cache cap in MB, defaults to the checkpoint. */
    weightCacheMB?: number;
}

/** Options for loading a SenseVoice model from a store. */
export interface LoadSenseVoiceOptions extends StoreModelOptions {}

/** Options for loading a model from the store. */
export interface LoadModelOptions extends ModelOptionsBase {}

/** Options for the mel frontend. */
export interface ComputeMelOptions {
    pcm: Float32Array;
    mel: MelConfig;
    filters: Float32Array;
    /** Pad to the 30 second chunk. Defaults to true. */
    padToChunk?: boolean;
}

/** Result of the mel frontend. */
export interface MelResult {
    values: Float32Array;
    frames: number;
    melBins: number;
    /** True frames before zero padding to the chunk. */
    trueFrames: number;
}

/** Options for a single greedy decode. */
export interface GenerateOptions {
    model: LoadedModel;
    encoderOutput: Float32Array;
    encoderSeq: number;
    /** Initial token ids fed before sampling. */
    prompt: number[];
    /** Apply the timestamp logits processor. */
    timestamps: boolean;
    /** Pairs to capture cross attention for. */
    capture?: CapturePair[];
    /** Hard cap on generated tokens. */
    maxTokens?: number;
    /** Stop when the tail falls into a repetition loop. */
    guardRepeat?: boolean;
}

/** Result of one greedy decode. */
export interface GenerateOutput {
    /** Generated token ids, prompt excluded. */
    ids: number[];
    /** All fed token ids, prompt first. */
    allIds: number[];
    /** Per fed token, per capture pair, weights over encoder frames. */
    attention: Float32Array[][];
    eos: boolean;
}

/** Options for a single audio window. */
export interface WhisperWindowOptions extends TranscribeBase {
    model: LoadedModel;
    pcm: Float32Array;
    timestamps: TimestampMode;
    /** Start of the window in seconds, added to every time. */
    timeOffset: number;
}

/** One decoded audio window. */
export interface WindowResult {
    ids: number[];
    text: string;
    words: Word[];
    segments: Segment[];
    language: Language;
}

/** One detected speech chunk. */
export interface SpeechChunk {
    start: number;
    end: number;
}

/** Options for the phrase level chunker. */
export interface SegmentOptions {
    pcm: Float32Array;
    samplingRate: number;
    maxChunkSeconds?: number;
    minPauseSeconds?: number;
    minChunkSeconds?: number;
    dbFloor?: number;
}

/** Options for a row-major linear layer. */
export interface LinearOptions {
    input: Float32Array;
    weight: Float32Array;
    bias?: Float32Array;
    rows: number;
    inDim: number;
    outDim: number;
    out?: Float32Array;
}

/** Options for layer normalization. */
export interface LayerNormOptions {
    input: Float32Array;
    weight: Float32Array;
    bias: Float32Array;
    rows: number;
    dims: number;
    eps?: number;
    out?: Float32Array;
    meanOut?: Float32Array;
    rstdOut?: Float32Array;
}

/** Options for single head attention. */
export interface AttentionOptions {
    q: Float32Array;
    k: Float32Array;
    v: Float32Array;
    seqQ: number;
    seqK: number;
    headDim: number;
    heads: number;
    causal?: boolean;
    scale?: number;
    out?: Float32Array;
    /** Per head additive bias, heads by seqQ by seqK. */
    bias?: Float32Array;
}

/** Options for a direct 1d convolution, channels first. */
export interface Conv1dOptions {
    input: Float32Array;
    weight: Float32Array;
    bias: Float32Array;
    inChannels: number;
    outChannels: number;
    inLength: number;
    kernel: number;
    stride: number;
    padding: number;
}

/** Options for a direct 2d convolution, channels first. */
export interface Conv2dOptions {
    input: Float32Array;
    weight: Float32Array;
    bias?: Float32Array;
    inChannels: number;
    inHeight: number;
    inWidth: number;
    outChannels: number;
    kernel: number;
    stride: number;
    groups: number;
}

/** Options for an in place SiLU over part of a buffer. */
export interface SiluOptions {
    values: Float32Array;
    offset?: number;
    length?: number;
}

/** Options for a 1x1 stride one convolution. */
export interface PointwiseOptions {
    /** Channels first input, channels by spatial. */
    input: Float32Array;
    /** Out channels by in channels weight. */
    weight: Float32Array;
    bias?: Float32Array;
    channels: number;
    outChannels: number;
    spatial: number;
}

/** Options for a Shaw style relative position bias. */
export interface RelativeBiasOptions {
    /** Query plus bias v, seq by hidden. */
    q: Float32Array;
    /** Projected positions, 2 seq minus 1 by hidden. */
    relK: Float32Array;
    seq: number;
    heads: number;
    headDim: number;
    scale: number;
}

/** Options for creating a worker pool over the shared kernel memory. */
export interface CreatePoolOptions {
    memory: WebAssembly.Memory;
    /** Requested worker count, defaults to every core. */
    threads?: number;
}

/** One parallel matrix multiplication request. */
export interface PoolMatmulOptions {
    input: number;
    weight: number;
    bias: number;
    out: number;
    rows: number;
    inDim: number;
    outDim: number;
}

/** Worker pool that splits matrix rows across threads. */
export interface TensorPool {
    threads: number;
    /** Run one split matmul, false when a worker failed. */
    matmul(options: PoolMatmulOptions): boolean;
    close(): void;
}

/** Instantiated SIMD kernel and its linear memory arena. */
export interface WasmKernel {
    memory: WebAssembly.Memory;
    matmul: (
        input: number,
        weight: number,
        bias: number,
        out: number,
        rows: number,
        inDim: number,
        outDim: number,
    ) => void;
    attention: (
        q: number,
        k: number,
        v: number,
        bias: number,
        out: number,
        scores: number,
        seqQ: number,
        seqK: number,
        heads: number,
        headDim: number,
        scale: number,
        causal: number,
    ) => void;
    conv1d: (
        input: number,
        weight: number,
        bias: number,
        out: number,
        inChannels: number,
        outChannels: number,
        inLength: number,
        kernel: number,
        stride: number,
        padding: number,
    ) => void;
    conv2d: (
        input: number,
        weight: number,
        bias: number,
        out: number,
        inChannels: number,
        inHeight: number,
        inWidth: number,
        outChannels: number,
        kernel: number,
        stride: number,
        groups: number,
    ) => void;
    silu: (values: number, length: number) => void;
    gelu: (values: number, length: number) => void;
    pointwise: (
        input: number,
        weight: number,
        bias: number,
        out: number,
        tmpIn: number,
        tmpOut: number,
        channels: number,
        outChannels: number,
        spatial: number,
    ) => void;
    relativeBias: (
        q: number,
        relK: number,
        out: number,
        bd: number,
        seq: number,
        heads: number,
        headDim: number,
        scale: number,
    ) => void;
    next: number;
    tensors: WeakMap<Float32Array, number>;
    /** Released arena regions grouped by byte size. */
    free: Map<number, number[]>;
    input: number;
    inputSize: number;
    output: number;
    outputSize: number;
    scores: number;
    scoresSize: number;
    bias: number;
    biasSize: number;
    bd: number;
    bdSize: number;
    shared: boolean;
    pool: TensorPool | null | undefined;
    poolTried: boolean;
}

/** Options for the DTW alignment. */
export interface DtwOptions {
    /** Per attention row, per head, weights over encoder frames. */
    attention: Float32Array[][];
    tokenCount: number;
    frameCount: number;
    heads: number;
    medianWidth?: number;
}

/** Options for word assembly. */
export interface BuildWordsOptions {
    ids: number[];
    tokenTimes: Float32Array;
    tokenizer: Tokenizer;
    timeOffset: number;
}

/** SentencePiece vocabulary built from GGUF metadata. */
export interface SenseVoiceVocabulary {
    tokenizer: Tokenizer;
    isControl(id: number): boolean;
    /** Language for a control token, or null. */
    languageOf(id: number): Language | null;
}

/** Options for the SenseVoice feature frontend. */
export interface SenseVoiceFeatureOptions {
    pcm: Float32Array;
    frontend: SenseVoiceFrontend;
}

/** SenseVoice feature output, frames by dInput. */
export interface SenseVoiceFeatureResult {
    values: Float32Array;
    frames: number;
}

/** Options for the SenseVoice encoder. */
export interface SenseVoiceEncodeOptions {
    config: SenseVoiceConfig;
    weights: SenseVoiceWeights;
    /** Frames by dInput, row major. */
    features: Float32Array;
    frames: number;
    /** Four prefix embedding rows, lid, event, emotion, itn. */
    prefix: number[];
}

/** SenseVoice encoder output. */
export interface SenseVoiceEncodeResult {
    hidden: Float32Array;
    seq: number;
}

/** Options for the SenseVoice CTC head. */
export interface SenseVoiceLogitsOptions {
    config: SenseVoiceConfig;
    weights: SenseVoiceWeights;
    hidden: Float32Array;
    seq: number;
}

/** Options for a greedy CTC decode. */
export interface CtcDecodeOptions {
    /** Frames by vocabulary, row major. */
    logits: Float32Array;
    frames: number;
    vocabSize: number;
    blankId: number;
}

/** Result of a greedy CTC decode. */
export interface CtcDecodeResult {
    ids: number[];
    frames: number[];
}

/** Precomputed Bluestein FFT plan for one transform size. */
export interface FftPlan {
    n: number;
    m: number;
    chirpRe: Float32Array;
    chirpIm: Float32Array;
    kernelRe: Float32Array;
    kernelIm: Float32Array;
    cosTable: Float32Array;
    sinTable: Float32Array;
    bitrev: Uint32Array;
}

/** Raw byte level BPE tokenizer file. */
export interface RawTokenizer {
    model: { vocab: Record<string, number> };
    added_tokens?: Array<{ id: number; content: string; special?: boolean }>;
}

/** LSTM hidden and cell state. */
export interface LstmState {
    hidden: Float32Array;
    cell: Float32Array;
}

/** One safetensors header entry. */
export interface SafetensorsHeaderEntry {
    dtype: string;
    shape: number[];
    data_offsets: [number, number];
}

/** Raw safetensors header, tensor names to entries. */
export interface SafetensorsHeader {
    __metadata__?: Record<string, string>;
    [name: string]: SafetensorsHeaderEntry | Record<string, string> | undefined;
}

/** One cross attention capture target. */
export interface CapturePair {
    layer: number;
    head: number;
}

/** Persistent decoder state for one audio window. */
export interface DecoderState {
    model: LoadedModel;
    seq: number;
    dModel: number;
    layers: number;
    heads: number;
    headDim: number;
    ffnDim: number;
    selfKeys: Float32Array[];
    selfValues: Float32Array[];
    crossKeys: Float32Array[];
    crossValues: Float32Array[];
    position: number;
}

/** Result of one decoder step. */
export interface StepResult {
    logits: Float32Array;
    captures: Float32Array[];
}

/** Options for the encoder pass. */
export interface EncodeOptions {
    model: LoadedModel;
    /** Log-mel values, mel major, [bins, frames]. */
    mel: Float32Array;
    /** Collect each layer output for parity tests. */
    collect?: boolean;
}

/** Encoder output plus optional intermediate states. */
export interface EncodeResult {
    output: Float32Array;
    seq: number;
    perLayer: Float32Array[];
}

/** Result of the DTW alignment. */
export interface DtwResult {
    textIndices: Int32Array;
    timeIndices: Int32Array;
    /** One time in seconds per attention row. */
    times: Float64Array;
}
