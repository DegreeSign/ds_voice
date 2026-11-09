// MOSS audio tokenizer encoder for voice cloning.
import { attention } from "../../tensor/kernel/kernels";
import { geluErfInPlace, layerNorm, linear } from "./ops";
const
    /** Transformer width of the encoder. */
    HIDDEN = 256,
    /** Attention heads of every transformer block. */
    HEADS = 4,
    /** Width of one attention head. */
    HEAD_DIM = 64,
    /** Feed forward width of every transformer block. */
    INTERMEDIATE = 1024,
    /** Residual codebooks produced per frame. */
    CODEBOOKS = 16,
    /** Number of entries in one codebook. */
    CODEBOOK_SIZE = 1024,
    /** Latent width of the residual quantizer. */
    CODE_DIM = 768,
    /** Projected width of one quantizer stage. */
    RVQ_DIM = 512,
    /** Width of one codebook vector. */
    CODEBOOK_DIM = 8,
    /** Interleaved stereo samples in one patch. */
    PATCH = 240,
    /** Stereo frames in one patch. */
    SAMPLES_PER_PATCH = 120,
    /** Input samples per code frame. */
    DOWNSAMPLE = 3840,
    /** Largest reference kept for one clone. */
    MAX_REFERENCE_SAMPLES = 4 * 48000,
    /** Smallest reference accepted for one clone. */
    MIN_REFERENCE_SAMPLES = 48000,
    /** Size of the published encoder data file. */
    ENCODER_BYTES = 44_507_136,
    /** Normalization and RoPE constants. */
    NORM_EPS = 1e-5,
    ROPE_BASE = 10000,
    /** Rotated pairs per head. */
    PAIRS = HEAD_DIM / 2,
    /** Stage shape and sliding window of every transformer stage. */
    STAGES: Array<{
        index: number;
        layers: number;
        input: number;
        output: number;
        context: number;
    }> = [
        { index: 1, layers: 4, input: PATCH, output: 384, context: 1600 },
        { index: 3, layers: 2, input: 768, output: 384, context: 1200 },
        { index: 5, layers: 2, input: 768, output: 384, context: 800 },
        { index: 7, layers: 4, input: 768, output: 192, context: 500 },
    ],
    /** RoPE frequencies of one head. */
    FREQUENCIES = Float32Array.from({ length: PAIRS }, (_, index) =>
        Math.exp((-Math.log(ROPE_BASE) * 2 * index) / HEAD_DIM),
    ),
    /** Quantizer input biases embedded in the ONNX graph. */
    QUANTIZER_BIASES: number[][] = [
        [
            -0.05210450291633606, -0.39299750328063965,
            -0.17262807488441467, 0.3030686676502228,
            0.3708377480506897, -0.22469322383403778,
            0.4127640426158905, 0.31836459040641785,
        ],
        [
            -0.016022227704524994, 0.13183528184890747,
            0.4306739866733551, -0.039186377078294754,
            -0.7015690207481384, -0.033869098871946335,
            -0.4187164902687073, -0.08875171095132828,
        ],
        [
            -0.15809592604637146, 0.014311866834759712,
            0.1774987131357193, 0.16086357831954956,
            0.6836158037185669, 0.439736008644104,
            -0.323175311088562, -0.18822523951530457,
        ],
        [
            0.37358155846595764, 0.11303484439849854,
            -0.18871447443962097, -0.021973522379994392,
            -0.4877983331680298, -0.589205801486969,
            0.028411684557795525, 0.06623940914869308,
        ],
        [
            -0.12588590383529663, -0.48017624020576477,
            0.004186512436717749, -0.23528897762298584,
            0.07721872627735138, 0.5996802449226379,
            -0.02517935447394848, 0.2507273554801941,
        ],
        [
            0.2577722370624542, -0.8969453573226929,
            -0.00127889693249017, -0.03637214004993439,
            0.0013230020413175225, -0.3023323714733124,
            0.23191148042678833, -0.04276416450738907,
        ],
        [
            0.16772086918354034, 0.5255719423294067,
            -0.5048452615737915, -0.0668092593550682,
            0.04707559943199158, -0.015946652740240097,
            -0.10536403208971024, -0.10836982727050781,
        ],
        [
            0.027662616223096848, -0.04897528514266014,
            0.07361172884702682, 0.4148974120616913,
            -0.1628134399652481, -0.3419361114501953,
            0.4493037164211273, 0.18348972499370575,
        ],
        [
            -0.0750976949930191, 0.03662341460585594,
            -0.09031876921653748, -0.22935402393341064,
            -0.010090729221701622, 0.5850993990898132,
            0.20050153136253357, -0.06906522065401077,
        ],
        [
            0.12097462266683578, 0.00012975989375263453,
            0.08773968368768692, 0.09734805673360825,
            0.3053659200668335, 0.07140238583087921,
            -0.020068373531103134, -0.6386568546295166,
        ],
        [
            0.3439544141292572, -0.3911440968513489,
            0.011705471202731133, -0.31977352499961853,
            -0.17778629064559937, 0.09164314717054367,
            -0.08739724010229111, -0.049476344138383865,
        ],
        [
            0.2246658205986023, -0.03519893065094948,
            -0.33994486927986145, 0.29893869161605835,
            -0.12780529260635376, 0.02495509386062622,
            -0.059228118509054184, -0.3431158661842346,
        ],
        [
            0.16344806551933289, -0.1592581421136856,
            -0.030575821176171303, 0.10716202110052109,
            -0.1858930140733719, -0.030512560158967972,
            0.21243008971214294, -0.4567442536354065,
        ],
        [
            0.10579255223274231, -0.03890235349535942,
            0.5027333498001099, -0.00231143063865602,
            0.01942877098917961, 0.1359902173280716,
            0.07363083958625793, 0.010577771812677383,
        ],
        [
            -0.48616087436676025, -0.17703881859779358,
            0.15982207655906677, 0.03171921521425247,
            -0.06179104372859001, 0.005565858446061611,
            -0.28120657801628113, -0.186770960688591,
        ],
        [
            0.021861586719751358, -0.03057532198727131,
            0.3443891704082489, 0.15361559391021729,
            0.012129705399274826, 0.029608074575662613,
            -0.1598145067691803, -0.0019524919334799051,
        ],
    ],
    /** Offset and byte length of every encoder tensor. */
    ENCODER_TENSORS: Record<string, [number, number]> = {
        "encoder.1.transformer.layers.0.norm1.weight": [0, 1024],
        "encoder.1.transformer.layers.0.norm1.bias": [1024, 1024],
        "encoder.1.transformer.layers.0.norm2.weight": [2048, 1024],
        "encoder.1.transformer.layers.0.norm2.bias": [3072, 1024],
        "encoder.1.transformer.layers.0.layer_scale_1.scale": [4096, 1024],
        "encoder.1.transformer.layers.0.layer_scale_2.scale": [5120, 1024],
        "encoder.1.transformer.layers.1.norm1.weight": [6144, 1024],
        "encoder.1.transformer.layers.1.norm1.bias": [7168, 1024],
        "encoder.1.transformer.layers.1.norm2.weight": [8192, 1024],
        "encoder.1.transformer.layers.1.norm2.bias": [9216, 1024],
        "encoder.1.transformer.layers.1.layer_scale_1.scale": [10240, 1024],
        "encoder.1.transformer.layers.1.layer_scale_2.scale": [11264, 1024],
        "encoder.1.transformer.layers.2.norm1.weight": [12288, 1024],
        "encoder.1.transformer.layers.2.norm1.bias": [13312, 1024],
        "encoder.1.transformer.layers.2.norm2.weight": [14336, 1024],
        "encoder.1.transformer.layers.2.norm2.bias": [15360, 1024],
        "encoder.1.transformer.layers.2.layer_scale_1.scale": [16384, 1024],
        "encoder.1.transformer.layers.2.layer_scale_2.scale": [17408, 1024],
        "encoder.1.transformer.layers.3.norm1.weight": [18432, 1024],
        "encoder.1.transformer.layers.3.norm1.bias": [19456, 1024],
        "encoder.1.transformer.layers.3.norm2.weight": [20480, 1024],
        "encoder.1.transformer.layers.3.norm2.bias": [21504, 1024],
        "encoder.1.transformer.layers.3.layer_scale_1.scale": [22528, 1024],
        "encoder.1.transformer.layers.3.layer_scale_2.scale": [23552, 1024],
        "encoder.3.transformer.layers.0.norm1.weight": [24576, 1024],
        "encoder.3.transformer.layers.0.norm1.bias": [25600, 1024],
        "encoder.3.transformer.layers.0.norm2.weight": [26624, 1024],
        "encoder.3.transformer.layers.0.norm2.bias": [27648, 1024],
        "encoder.3.transformer.layers.0.layer_scale_1.scale": [28672, 1024],
        "encoder.3.transformer.layers.0.layer_scale_2.scale": [29696, 1024],
        "encoder.3.transformer.layers.1.norm1.weight": [30720, 1024],
        "encoder.3.transformer.layers.1.norm1.bias": [31744, 1024],
        "encoder.3.transformer.layers.1.norm2.weight": [32768, 1024],
        "encoder.3.transformer.layers.1.norm2.bias": [33792, 1024],
        "encoder.3.transformer.layers.1.layer_scale_1.scale": [34816, 1024],
        "encoder.3.transformer.layers.1.layer_scale_2.scale": [35840, 1024],
        "encoder.5.transformer.layers.0.norm1.weight": [36864, 1024],
        "encoder.5.transformer.layers.0.norm1.bias": [37888, 1024],
        "encoder.5.transformer.layers.0.norm2.weight": [38912, 1024],
        "encoder.5.transformer.layers.0.norm2.bias": [39936, 1024],
        "encoder.5.transformer.layers.0.layer_scale_1.scale": [40960, 1024],
        "encoder.5.transformer.layers.0.layer_scale_2.scale": [41984, 1024],
        "encoder.5.transformer.layers.1.norm1.weight": [43008, 1024],
        "encoder.5.transformer.layers.1.norm1.bias": [44032, 1024],
        "encoder.5.transformer.layers.1.norm2.weight": [45056, 1024],
        "encoder.5.transformer.layers.1.norm2.bias": [46080, 1024],
        "encoder.5.transformer.layers.1.layer_scale_1.scale": [47104, 1024],
        "encoder.5.transformer.layers.1.layer_scale_2.scale": [48128, 1024],
        "encoder.7.transformer.layers.0.norm1.weight": [49152, 1024],
        "encoder.7.transformer.layers.0.norm1.bias": [50176, 1024],
        "encoder.7.transformer.layers.0.norm2.weight": [51200, 1024],
        "encoder.7.transformer.layers.0.norm2.bias": [52224, 1024],
        "encoder.7.transformer.layers.0.layer_scale_1.scale": [53248, 1024],
        "encoder.7.transformer.layers.0.layer_scale_2.scale": [54272, 1024],
        "encoder.7.transformer.layers.1.norm1.weight": [55296, 1024],
        "encoder.7.transformer.layers.1.norm1.bias": [56320, 1024],
        "encoder.7.transformer.layers.1.norm2.weight": [57344, 1024],
        "encoder.7.transformer.layers.1.norm2.bias": [58368, 1024],
        "encoder.7.transformer.layers.1.layer_scale_1.scale": [59392, 1024],
        "encoder.7.transformer.layers.1.layer_scale_2.scale": [60416, 1024],
        "encoder.7.transformer.layers.2.norm1.weight": [61440, 1024],
        "encoder.7.transformer.layers.2.norm1.bias": [62464, 1024],
        "encoder.7.transformer.layers.2.norm2.weight": [63488, 1024],
        "encoder.7.transformer.layers.2.norm2.bias": [64512, 1024],
        "encoder.7.transformer.layers.2.layer_scale_1.scale": [65536, 1024],
        "encoder.7.transformer.layers.2.layer_scale_2.scale": [66560, 1024],
        "encoder.7.transformer.layers.3.norm1.weight": [67584, 1024],
        "encoder.7.transformer.layers.3.norm1.bias": [68608, 1024],
        "encoder.7.transformer.layers.3.norm2.weight": [69632, 1024],
        "encoder.7.transformer.layers.3.norm2.bias": [70656, 1024],
        "encoder.7.transformer.layers.3.layer_scale_1.scale": [71680, 1024],
        "encoder.7.transformer.layers.3.layer_scale_2.scale": [72704, 1024],
        "quantizer.input_proj.bias": [73728, 2048],
        "quantizer.quantizers.0.out_proj.bias": [75776, 2048],
        "quantizer.quantizers.1.out_proj.bias": [77824, 2048],
        "quantizer.quantizers.2.out_proj.bias": [79872, 2048],
        "quantizer.quantizers.3.out_proj.bias": [81920, 2048],
        "quantizer.quantizers.4.out_proj.bias": [83968, 2048],
        "quantizer.quantizers.5.out_proj.bias": [86016, 2048],
        "quantizer.quantizers.6.out_proj.bias": [88064, 2048],
        "quantizer.quantizers.7.out_proj.bias": [90112, 2048],
        "quantizer.quantizers.8.out_proj.bias": [92160, 2048],
        "quantizer.quantizers.9.out_proj.bias": [94208, 2048],
        "quantizer.quantizers.10.out_proj.bias": [96256, 2048],
        "quantizer.quantizers.11.out_proj.bias": [98304, 2048],
        "quantizer.quantizers.12.out_proj.bias": [100352, 2048],
        "quantizer.quantizers.13.out_proj.bias": [102400, 2048],
        "quantizer.quantizers.14.out_proj.bias": [104448, 2048],
        "encoder.1.input_proj.weight": [106496, 245760],
        "encoder.1.transformer.layers.0.self_attn.in_proj.weight": [
            352256, 786432,
        ],
        "encoder.1.transformer.layers.0.self_attn.out_proj.weight": [
            1138688, 262144,
        ],
        "encoder.1.transformer.layers.0.ffn.ffn.0.weight": [1400832, 1048576],
        "encoder.1.transformer.layers.0.ffn.ffn.2.weight": [2449408, 1048576],
        "encoder.1.transformer.layers.1.self_attn.in_proj.weight": [
            3497984, 786432,
        ],
        "encoder.1.transformer.layers.1.self_attn.out_proj.weight": [
            4284416, 262144,
        ],
        "encoder.1.transformer.layers.1.ffn.ffn.0.weight": [4546560, 1048576],
        "encoder.1.transformer.layers.1.ffn.ffn.2.weight": [5595136, 1048576],
        "encoder.1.transformer.layers.2.self_attn.in_proj.weight": [
            6643712, 786432,
        ],
        "encoder.1.transformer.layers.2.self_attn.out_proj.weight": [
            7430144, 262144,
        ],
        "encoder.1.transformer.layers.2.ffn.ffn.0.weight": [7692288, 1048576],
        "encoder.1.transformer.layers.2.ffn.ffn.2.weight": [8740864, 1048576],
        "encoder.1.transformer.layers.3.self_attn.in_proj.weight": [
            9789440, 786432,
        ],
        "encoder.1.transformer.layers.3.self_attn.out_proj.weight": [
            10575872, 262144,
        ],
        "encoder.1.transformer.layers.3.ffn.ffn.0.weight": [10838016, 1048576],
        "encoder.1.transformer.layers.3.ffn.ffn.2.weight": [11886592, 1048576],
        "encoder.1.output_proj.weight": [12935168, 393216],
        "encoder.3.input_proj.weight": [13328384, 786432],
        "encoder.3.transformer.layers.0.self_attn.in_proj.weight": [
            14114816, 786432,
        ],
        "encoder.3.transformer.layers.0.self_attn.out_proj.weight": [
            14901248, 262144,
        ],
        "encoder.3.transformer.layers.0.ffn.ffn.0.weight": [15163392, 1048576],
        "encoder.3.transformer.layers.0.ffn.ffn.2.weight": [16211968, 1048576],
        "encoder.3.transformer.layers.1.self_attn.in_proj.weight": [
            17260544, 786432,
        ],
        "encoder.3.transformer.layers.1.self_attn.out_proj.weight": [
            18046976, 262144,
        ],
        "encoder.3.transformer.layers.1.ffn.ffn.0.weight": [18309120, 1048576],
        "encoder.3.transformer.layers.1.ffn.ffn.2.weight": [19357696, 1048576],
        "encoder.3.output_proj.weight": [20406272, 393216],
        "encoder.5.input_proj.weight": [20799488, 786432],
        "encoder.5.transformer.layers.0.self_attn.in_proj.weight": [
            21585920, 786432,
        ],
        "encoder.5.transformer.layers.0.self_attn.out_proj.weight": [
            22372352, 262144,
        ],
        "encoder.5.transformer.layers.0.ffn.ffn.0.weight": [22634496, 1048576],
        "encoder.5.transformer.layers.0.ffn.ffn.2.weight": [23683072, 1048576],
        "encoder.5.transformer.layers.1.self_attn.in_proj.weight": [
            24731648, 786432,
        ],
        "encoder.5.transformer.layers.1.self_attn.out_proj.weight": [
            25518080, 262144,
        ],
        "encoder.5.transformer.layers.1.ffn.ffn.0.weight": [25780224, 1048576],
        "encoder.5.transformer.layers.1.ffn.ffn.2.weight": [26828800, 1048576],
        "encoder.5.output_proj.weight": [27877376, 393216],
        "encoder.7.input_proj.weight": [28270592, 786432],
        "encoder.7.transformer.layers.0.self_attn.in_proj.weight": [
            29057024, 786432,
        ],
        "encoder.7.transformer.layers.0.self_attn.out_proj.weight": [
            29843456, 262144,
        ],
        "encoder.7.transformer.layers.0.ffn.ffn.0.weight": [30105600, 1048576],
        "encoder.7.transformer.layers.0.ffn.ffn.2.weight": [31154176, 1048576],
        "encoder.7.transformer.layers.1.self_attn.in_proj.weight": [
            32202752, 786432,
        ],
        "encoder.7.transformer.layers.1.self_attn.out_proj.weight": [
            32989184, 262144,
        ],
        "encoder.7.transformer.layers.1.ffn.ffn.0.weight": [33251328, 1048576],
        "encoder.7.transformer.layers.1.ffn.ffn.2.weight": [34299904, 1048576],
        "encoder.7.transformer.layers.2.self_attn.in_proj.weight": [
            35348480, 786432,
        ],
        "encoder.7.transformer.layers.2.self_attn.out_proj.weight": [
            36134912, 262144,
        ],
        "encoder.7.transformer.layers.2.ffn.ffn.0.weight": [36397056, 1048576],
        "encoder.7.transformer.layers.2.ffn.ffn.2.weight": [37445632, 1048576],
        "encoder.7.transformer.layers.3.self_attn.in_proj.weight": [
            38494208, 786432,
        ],
        "encoder.7.transformer.layers.3.self_attn.out_proj.weight": [
            39280640, 262144,
        ],
        "encoder.7.transformer.layers.3.ffn.ffn.0.weight": [39542784, 1048576],
        "encoder.7.transformer.layers.3.ffn.ffn.2.weight": [40591360, 1048576],
        "encoder.7.output_proj.weight": [41639936, 196608],
        "quantizer.input_proj.weight": [41836544, 1572864],
        "quantizer.quantizers.0.in_proj.weight": [43409408, 16384],
        "quantizer.quantizers.0.codebook": [43425792, 32768],
        "quantizer.quantizers.0.out_proj.weight": [43462656, 16384],
        "quantizer.quantizers.1.in_proj.weight": [43479040, 16384],
        "quantizer.quantizers.1.codebook": [43495424, 32768],
        "quantizer.quantizers.1.out_proj.weight": [43532288, 16384],
        "quantizer.quantizers.2.in_proj.weight": [43548672, 16384],
        "quantizer.quantizers.2.codebook": [43565056, 32768],
        "quantizer.quantizers.2.out_proj.weight": [43601920, 16384],
        "quantizer.quantizers.3.in_proj.weight": [43618304, 16384],
        "quantizer.quantizers.3.codebook": [43634688, 32768],
        "quantizer.quantizers.3.out_proj.weight": [43671552, 16384],
        "quantizer.quantizers.4.in_proj.weight": [43687936, 16384],
        "quantizer.quantizers.4.codebook": [43704320, 32768],
        "quantizer.quantizers.4.out_proj.weight": [43741184, 16384],
        "quantizer.quantizers.5.in_proj.weight": [43757568, 16384],
        "quantizer.quantizers.5.codebook": [43773952, 32768],
        "quantizer.quantizers.5.out_proj.weight": [43810816, 16384],
        "quantizer.quantizers.6.in_proj.weight": [43827200, 16384],
        "quantizer.quantizers.6.codebook": [43843584, 32768],
        "quantizer.quantizers.6.out_proj.weight": [43880448, 16384],
        "quantizer.quantizers.7.in_proj.weight": [43896832, 16384],
        "quantizer.quantizers.7.codebook": [43913216, 32768],
        "quantizer.quantizers.7.out_proj.weight": [43950080, 16384],
        "quantizer.quantizers.8.in_proj.weight": [43966464, 16384],
        "quantizer.quantizers.8.codebook": [43982848, 32768],
        "quantizer.quantizers.8.out_proj.weight": [44019712, 16384],
        "quantizer.quantizers.9.in_proj.weight": [44036096, 16384],
        "quantizer.quantizers.9.codebook": [44052480, 32768],
        "quantizer.quantizers.9.out_proj.weight": [44089344, 16384],
        "quantizer.quantizers.10.in_proj.weight": [44105728, 16384],
        "quantizer.quantizers.10.codebook": [44122112, 32768],
        "quantizer.quantizers.10.out_proj.weight": [44158976, 16384],
        "quantizer.quantizers.11.in_proj.weight": [44175360, 16384],
        "quantizer.quantizers.11.codebook": [44191744, 32768],
        "quantizer.quantizers.11.out_proj.weight": [44228608, 16384],
        "quantizer.quantizers.12.in_proj.weight": [44244992, 16384],
        "quantizer.quantizers.12.codebook": [44261376, 32768],
        "quantizer.quantizers.12.out_proj.weight": [44298240, 16384],
        "quantizer.quantizers.13.in_proj.weight": [44314624, 16384],
        "quantizer.quantizers.13.codebook": [44331008, 32768],
        "quantizer.quantizers.13.out_proj.weight": [44367872, 16384],
        "quantizer.quantizers.14.in_proj.weight": [44384256, 16384],
        "quantizer.quantizers.14.codebook": [44400640, 32768],
        "quantizer.quantizers.14.out_proj.weight": [44437504, 16384],
        "quantizer.quantizers.15.in_proj.weight": [44453888, 16384],
        "quantizer.quantizers.15.codebook": [44470272, 32768],
    };

/** One transformer block of the encoder. */
interface MossEncoderLayer {
    norm1Weight: Float32Array;
    norm1Bias: Float32Array;
    norm2Weight: Float32Array;
    norm2Bias: Float32Array;
    scale1: Float32Array;
    scale2: Float32Array;
    qkv: Float32Array;
    out: Float32Array;
    up: Float32Array;
    down: Float32Array;
}

/** One encoder transformer stage plus its projections. */
interface MossEncoderStage {
    inputDim: number;
    outputDim: number;
    context: number;
    input: Float32Array;
    output: Float32Array;
    layers: MossEncoderLayer[];
}

/** One residual vector quantizer of the encoder. */
interface MossEncoderQuantizer {
    inWeight: Float32Array;
    inBias: Float32Array;
    codebook: Float32Array;
    outWeight?: Float32Array;
    outBias?: Float32Array;
}

/** Every weight of the MOSS audio tokenizer encoder. */
interface MossEncoder {
    input: Float32Array;
    stages: MossEncoderStage[];
    quantizerInput: Float32Array;
    quantizerBias: Float32Array;
    quantizers: MossEncoderQuantizer[];
}

/** Read a little endian f32 buffer without an unaligned view. */
const floatsOf = (bytes: Uint8Array): Float32Array => {
    const aligned = bytes.byteOffset % 4 === 0;
    if (!aligned) {
        bytes = bytes.slice();
    }
    return new Float32Array(
        bytes.buffer,
        bytes.byteOffset,
        Math.floor(bytes.byteLength / 4),
    );
};

/** Read one tensor by name, copying it away from the file buffer. */
const readTensor = (options: {
    floats: Float32Array;
    name: string;
    values: number;
}): Float32Array => {
    const entry = ENCODER_TENSORS[options.name];
    if (!entry) {
        throw new Error(`missing encoder tensor ${options.name}`);
    }
    const [offset, length] = entry;
    if (length !== options.values * 4) {
        throw new Error(`encoder tensor ${options.name} has the wrong size`);
    }
    return options.floats.slice(offset / 4, (offset + length) / 4);
};

/** Read one matrix and transpose it for the shared linear kernel. */
const readMatrix = (options: {
    floats: Float32Array;
    name: string;
    inDim: number;
    outDim: number;
}): Float32Array => {
    const
        { floats, name, inDim, outDim } = options,
        weight = readTensor({ floats, name, values: inDim * outDim }),
        out = new Float32Array(weight.length);
    for (let row = 0; row < inDim; row += 1) {
        for (let unit = 0; unit < outDim; unit += 1) {
            out[unit * inDim + row] = weight[row * outDim + unit];
        }
    }
    return out;
};

/** Read one transformer block. */
const readLayer = (options: {
    floats: Float32Array;
    stage: number;
    layer: number;
}): MossEncoderLayer => {
    const
        { floats, stage, layer } = options,
        prefix = `encoder.${stage}.transformer.layers.${layer}`,
        vector = (name: string): Float32Array =>
            readTensor({
                floats,
                name: `${prefix}.${name}`,
                values: HIDDEN,
            }),
        matrix = (name: string, inDim: number, outDim: number): Float32Array =>
            readMatrix({ floats, name: `${prefix}.${name}`, inDim, outDim });
    return {
        norm1Weight: vector("norm1.weight"),
        norm1Bias: vector("norm1.bias"),
        norm2Weight: vector("norm2.weight"),
        norm2Bias: vector("norm2.bias"),
        scale1: vector("layer_scale_1.scale"),
        scale2: vector("layer_scale_2.scale"),
        qkv: matrix("self_attn.in_proj.weight", HIDDEN, 3 * HIDDEN),
        out: matrix("self_attn.out_proj.weight", HIDDEN, HIDDEN),
        up: matrix("ffn.ffn.0.weight", HIDDEN, INTERMEDIATE),
        down: matrix("ffn.ffn.2.weight", INTERMEDIATE, HIDDEN),
    };
};

/** Read one encoder stage. */
const readStage = (options: {
    floats: Float32Array;
    index: number;
    layers: number;
    inputDim: number;
    outputDim: number;
    context: number;
}): MossEncoderStage => {
    const
        { floats, index, layers: count } = options,
        layers: MossEncoderLayer[] = [];
    for (let layer = 0; layer < count; layer += 1) {
        layers.push(readLayer({ floats, stage: index, layer }));
    }
    return {
        inputDim: options.inputDim,
        outputDim: options.outputDim,
        context: options.context,
        input: readMatrix({
            floats,
            name: `encoder.${index}.input_proj.weight`,
            inDim: options.inputDim,
            outDim: HIDDEN,
        }),
        output: readMatrix({
            floats,
            name: `encoder.${index}.output_proj.weight`,
            inDim: HIDDEN,
            outDim: options.outputDim,
        }),
        layers,
    };
};

/** Read one residual quantizer. */
const readQuantizer = (options: {
    floats: Float32Array;
    index: number;
}): MossEncoderQuantizer => {
    const
        { floats, index } = options,
        prefix = `quantizer.quantizers.${index}`,
        tensor = (name: string, values: number): Float32Array =>
            readTensor({ floats, name: `${prefix}.${name}`, values }),
        last = index + 1 === CODEBOOKS;
    return {
        inWeight: tensor("in_proj.weight", CODEBOOK_DIM * RVQ_DIM),
        inBias: Float32Array.from(QUANTIZER_BIASES[index]),
        codebook: tensor("codebook", CODEBOOK_SIZE * CODEBOOK_DIM),
        outWeight: last
            ? undefined
            : tensor("out_proj.weight", RVQ_DIM * CODEBOOK_DIM),
        outBias: last ? undefined : tensor("out_proj.bias", RVQ_DIM),
    };
};

/** Load the MOSS audio tokenizer encoder from its data file. */
const loadMossEncoder = (bytes: Uint8Array): MossEncoder => {
    if (bytes.length < ENCODER_BYTES) {
        throw new Error(`encoder data is too small`);
    }
    const
        floats = floatsOf(bytes),
        stages: MossEncoderStage[] = [],
        quantizers: MossEncoderQuantizer[] = [];
    for (const spec of STAGES) {
        stages.push(readStage({
            floats,
            index: spec.index,
            layers: spec.layers,
            inputDim: spec.input,
            outputDim: spec.output,
            context: spec.context,
        }));
    }
    for (let index = 0; index < CODEBOOKS; index += 1) {
        quantizers.push(readQuantizer({ floats, index }));
    }
    return {
        input: readMatrix({
            floats,
            name: "encoder.1.input_proj.weight",
            inDim: PATCH,
            outDim: HIDDEN,
        }),
        stages,
        quantizerInput: readTensor({
            floats,
            name: "quantizer.input_proj.weight",
            values: CODE_DIM * RVQ_DIM,
        }),
        quantizerBias: readTensor({
            floats,
            name: "quantizer.input_proj.bias",
            values: RVQ_DIM,
        }),
        quantizers,
    };
};

/** Apply interleaved RoPE to one query or key tensor. */
const rope = (values: Float32Array, rows: number): void => {
    const
        cos = new Float32Array(rows * PAIRS),
        sin = new Float32Array(rows * PAIRS);
    for (let row = 0; row < rows; row += 1) {
        for (let pair = 0; pair < PAIRS; pair += 1) {
            const angle = row * FREQUENCIES[pair];
            cos[row * PAIRS + pair] = Math.cos(angle);
            sin[row * PAIRS + pair] = Math.sin(angle);
        }
    }
    for (let row = 0; row < rows; row += 1) {
        for (let head = 0; head < HEADS; head += 1) {
            const base = row * HIDDEN + head * HEAD_DIM;
            for (let pair = 0; pair < PAIRS; pair += 1) {
                const
                    even = base + pair * 2,
                    odd = even + 1,
                    real = values[even],
                    imag = values[odd],
                    cosine = cos[row * PAIRS + pair],
                    sine = sin[row * PAIRS + pair];
                values[even] = real * cosine - imag * sine;
                values[odd] = real * sine + imag * cosine;
            }
        }
    }
};

/** Run one transformer stage over one frame prefix. */
const runStage = (options: {
    input: Float32Array;
    rows: number;
    stage: MossEncoderStage;
}): Float32Array => {
    const { rows, stage } = options;
    let x = options.input;
    for (const layer of stage.layers) {
        const normed = layerNorm({
            input: x,
            frames: rows,
            channels: HIDDEN,
            weight: layer.norm1Weight,
            bias: layer.norm1Bias,
            eps: NORM_EPS,
        });
        const
            qkv = linear({
                input: normed,
                rows,
                inDim: HIDDEN,
                weight: layer.qkv,
                outDim: 3 * HIDDEN,
            }),
            q = new Float32Array(rows * HIDDEN),
            k = new Float32Array(rows * HIDDEN),
            v = new Float32Array(rows * HIDDEN);
        for (let row = 0; row < rows; row += 1) {
            const base = row * 3 * HIDDEN;
            q.set(qkv.subarray(base, base + HIDDEN), row * HIDDEN);
            k.set(
                qkv.subarray(base + HIDDEN, base + 2 * HIDDEN),
                row * HIDDEN,
            );
            v.set(
                qkv.subarray(base + 2 * HIDDEN, base + 3 * HIDDEN),
                row * HIDDEN,
            );
        }
        rope(q, rows);
        rope(k, rows);
        const
            attended = attention({
                q,
                k,
                v,
                seqQ: rows,
                seqK: rows,
                heads: HEADS,
                headDim: HEAD_DIM,
                causal: true,
            }),
            projected = linear({
                input: attended,
                rows,
                inDim: HIDDEN,
                weight: layer.out,
                outDim: HIDDEN,
            });
        for (let i = 0; i < x.length; i += 1) {
            x[i] += layer.scale1[i % HIDDEN] * projected[i];
        }
        const normed2 = layerNorm({
            input: x,
            frames: rows,
            channels: HIDDEN,
            weight: layer.norm2Weight,
            bias: layer.norm2Bias,
            eps: NORM_EPS,
        });
        const
            up = linear({
                input: normed2,
                rows,
                inDim: HIDDEN,
                weight: layer.up,
                outDim: INTERMEDIATE,
            });
        geluErfInPlace(up);
        const forward = linear({
            input: up,
            rows,
            inDim: INTERMEDIATE,
            weight: layer.down,
            outDim: HIDDEN,
        });
        for (let i = 0; i < x.length; i += 1) {
            x[i] += layer.scale2[i % HIDDEN] * forward[i];
        }
    }
    return x;
};

/** Interleave adjacent frames into a wider channel axis. */
const interleave = (options: {
    input: Float32Array;
    rows: number;
    dim: number;
    groups: number;
}): Float32Array => {
    const
        { input, rows, dim, groups } = options,
        outRows = rows / groups,
        out = new Float32Array(outRows * dim * groups);
    for (let row = 0; row < outRows; row += 1) {
        for (let channel = 0; channel < dim; channel += 1) {
            for (let group = 0; group < groups; group += 1) {
                out[row * dim * groups + channel * groups + group] =
                    input[(row * groups + group) * dim + channel];
            }
        }
    }
    return out;
};

/** Quantize one latent run into residual codebook codes. */
const quantize = (options: {
    encoder: MossEncoder;
    input: Float32Array;
    rows: number;
}): Uint16Array => {
    const
        { encoder, rows } = options,
        codes = new Uint16Array(rows * CODEBOOKS),
        selected = new Float32Array(rows * CODEBOOK_DIM),
        normalized = new Float32Array(CODEBOOK_SIZE * CODEBOOK_DIM),
        codebookNorm = new Float32Array(CODEBOOK_SIZE);
    let residual = linear({
        input: options.input,
        rows,
        inDim: CODE_DIM,
        weight: encoder.quantizerInput,
        outDim: RVQ_DIM,
        bias: encoder.quantizerBias,
    });
    for (let level = 0; level < CODEBOOKS; level += 1) {
        const
            quantizer = encoder.quantizers[level],
            projected = linear({
                input: residual,
                rows,
                inDim: RVQ_DIM,
                weight: quantizer.inWeight,
                outDim: CODEBOOK_DIM,
                bias: quantizer.inBias,
            });
        for (let code = 0; code < CODEBOOK_SIZE; code += 1) {
            const base = code * CODEBOOK_DIM;
            let sum = 0;
            for (let d = 0; d < CODEBOOK_DIM; d += 1) {
                const value = quantizer.codebook[base + d];
                sum += value * value;
            }
            const norm = Math.max(Math.sqrt(sum), NORM_EPS);
            let squared = 0;
            for (let d = 0; d < CODEBOOK_DIM; d += 1) {
                const value = quantizer.codebook[base + d] / norm;
                normalized[base + d] = value;
                squared += value * value;
            }
            codebookNorm[code] = squared;
        }
        for (let row = 0; row < rows; row += 1) {
            const base = row * CODEBOOK_DIM;
            let sum = 0;
            for (let d = 0; d < CODEBOOK_DIM; d += 1) {
                sum += projected[base + d] * projected[base + d];
            }
            const scale = 1 / Math.max(Math.sqrt(sum), NORM_EPS);
            let best = 0;
            let bestDistance = Infinity;
            for (let code = 0; code < CODEBOOK_SIZE; code += 1) {
                const codeBase = code * CODEBOOK_DIM;
                let dot = 0;
                for (let d = 0; d < CODEBOOK_DIM; d += 1) {
                    dot += projected[base + d] * scale *
                        normalized[codeBase + d];
                }
                const distance = 1 - 2 * dot + codebookNorm[code];
                if (distance < bestDistance) {
                    bestDistance = distance;
                    best = code;
                }
            }
            codes[row * CODEBOOKS + level] = best;
            if (level + 1 === CODEBOOKS) {
                continue;
            }
            const codeBase = best * CODEBOOK_DIM;
            for (let d = 0; d < CODEBOOK_DIM; d += 1) {
                selected[base + d] = quantizer.codebook[codeBase + d];
            }
        }
        if (!quantizer.outWeight || !quantizer.outBias) {
            break;
        }
        const part = linear({
            input: selected,
            rows,
            inDim: CODEBOOK_DIM,
            weight: quantizer.outWeight,
            outDim: RVQ_DIM,
            bias: quantizer.outBias,
        });
        for (let i = 0; i < residual.length; i += 1) {
            residual[i] -= part[i];
        }
    }
    return codes;
};

/** Encode mono 48 kHz PCM into MOSS reference codes. */
const encodeMossAudio = (options: {
    encoder: MossEncoder;
    pcm: Float32Array;
}): Uint16Array => {
    const
        { encoder } = options,
        pcm = options.pcm.length > MAX_REFERENCE_SAMPLES
            ? options.pcm.subarray(0, MAX_REFERENCE_SAMPLES)
            : options.pcm;
    if (pcm.length < MIN_REFERENCE_SAMPLES) {
        throw new Error(`reference audio is too short`);
    }
    const
        validCodes = Math.floor(pcm.length / DOWNSAMPLE),
        pad = DOWNSAMPLE - (pcm.length % DOWNSAMPLE),
        length = pcm.length + pad,
        totalRows = length / SAMPLES_PER_PATCH,
        firstRows = Math.min(totalRows, validCodes * 32),
        patches = new Float32Array(firstRows * PATCH);
    for (let row = 0; row < firstRows; row += 1) {
        for (let sample = 0; sample < SAMPLES_PER_PATCH; sample += 1) {
            const
                index = row * SAMPLES_PER_PATCH + sample,
                value = index < pcm.length ? pcm[index] : 0;
            patches[row * PATCH + sample * 2] = value;
            patches[row * PATCH + sample * 2 + 1] = value;
        }
    }
    let
        x = linear({
            input: patches,
            rows: firstRows,
            inDim: PATCH,
            weight: encoder.input,
            outDim: HIDDEN,
        }),
        rows = firstRows;
    for (let index = 0; index < encoder.stages.length; index += 1) {
        const
            stage = encoder.stages[index],
            required = validCodes * (32 >> index);
        rows = Math.min(rows, required);
        if (index > 0) {
            x = linear({
                input: x,
                rows,
                inDim: stage.inputDim,
                weight: stage.input,
                outDim: HIDDEN,
            });
        }
        x = runStage({ input: x, rows, stage });
        const
            projected = linear({
                input: x,
                rows,
                inDim: HIDDEN,
                weight: stage.output,
                outDim: stage.outputDim,
            }),
            groups = index + 1 < encoder.stages.length ? 2 : 4;
        x = interleave({
            input: projected,
            rows,
            dim: stage.outputDim,
            groups,
        });
        rows /= groups;
    }
    return quantize({
        encoder,
        input: x,
        rows: Math.min(rows, validCodes),
    });
};

export {
    encodeMossAudio,
    loadMossEncoder,
    MAX_REFERENCE_SAMPLES,
    MIN_REFERENCE_SAMPLES,
};
export type { MossEncoder, MossEncoderLayer, MossEncoderStage };
