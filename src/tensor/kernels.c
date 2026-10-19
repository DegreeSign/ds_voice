// SIMD kernels for the voice graph, compiled to WebAssembly.
// Build: see the comment in src/tensor/kernelsWasm.ts.
#include <wasm_simd128.h>

#define MATMUL_ROWS 16

static inline float dot4(const float *a, const float *b, int length) {
    v128_t a0 = wasm_f32x4_splat(0.0f);
    v128_t a1 = wasm_f32x4_splat(0.0f);
    v128_t a2 = wasm_f32x4_splat(0.0f);
    v128_t a3 = wasm_f32x4_splat(0.0f);
    int i = 0;
    for (; i + 15 < length; i += 16) {
        a0 = wasm_f32x4_add(a0, wasm_f32x4_mul(
            wasm_v128_load(a + i), wasm_v128_load(b + i)));
        a1 = wasm_f32x4_add(a1, wasm_f32x4_mul(
            wasm_v128_load(a + i + 4), wasm_v128_load(b + i + 4)));
        a2 = wasm_f32x4_add(a2, wasm_f32x4_mul(
            wasm_v128_load(a + i + 8), wasm_v128_load(b + i + 8)));
        a3 = wasm_f32x4_add(a3, wasm_f32x4_mul(
            wasm_v128_load(a + i + 12), wasm_v128_load(b + i + 12)));
    }
    for (; i + 3 < length; i += 4) {
        a0 = wasm_f32x4_add(a0, wasm_f32x4_mul(wasm_v128_load(a + i), wasm_v128_load(b + i)));
    }
    a0 = wasm_f32x4_add(wasm_f32x4_add(a0, a1), wasm_f32x4_add(a2, a3));
    float sum = wasm_f32x4_extract_lane(a0, 0) + wasm_f32x4_extract_lane(a0, 1)
              + wasm_f32x4_extract_lane(a0, 2) + wasm_f32x4_extract_lane(a0, 3);
    for (; i < length; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

static inline float expfFast(float x);

/** Fast four lane exp. */
static inline v128_t expf4(v128_t x) {
    const v128_t lo = wasm_f32x4_splat(-87.0f);
    const v128_t hi = wasm_f32x4_splat(88.0f);
    x = wasm_f32x4_max(x, lo);
    x = wasm_f32x4_min(x, hi);
    const v128_t z = wasm_f32x4_mul(x, wasm_f32x4_splat(1.4426950408889634f));
    const v128_t kf = wasm_f32x4_floor(
        wasm_f32x4_add(z, wasm_f32x4_splat(0.5f))
    );
    const v128_t r = wasm_f32x4_sub(
        x,
        wasm_f32x4_mul(kf, wasm_f32x4_splat(0.6931471805599453f))
    );
    v128_t p = wasm_f32x4_splat(1.0f / 720.0f);
    p = wasm_f32x4_add(wasm_f32x4_splat(1.0f / 120.0f), wasm_f32x4_mul(r, p));
    p = wasm_f32x4_add(wasm_f32x4_splat(1.0f / 24.0f), wasm_f32x4_mul(r, p));
    p = wasm_f32x4_add(wasm_f32x4_splat(1.0f / 6.0f), wasm_f32x4_mul(r, p));
    p = wasm_f32x4_add(wasm_f32x4_splat(0.5f), wasm_f32x4_mul(r, p));
    p = wasm_f32x4_add(wasm_f32x4_splat(1.0f), wasm_f32x4_mul(r, p));
    p = wasm_f32x4_add(wasm_f32x4_splat(1.0f), wasm_f32x4_mul(r, p));
    const v128_t ki = wasm_i32x4_trunc_sat_f32x4(kf);
    const v128_t bits = wasm_i32x4_shl(
        wasm_i32x4_add(ki, wasm_i32x4_splat(127)),
        23
    );
    return wasm_f32x4_mul(p, bits);
}

/** SiLU over a contiguous range. */
__attribute__((visibility("default")))
void silu(float *values, int length) {
    int i = 0;
    const v128_t one = wasm_f32x4_splat(1.0f);
    for (; i + 3 < length; i += 4) {
        const v128_t x = wasm_v128_load(values + i);
        const v128_t e = expf4(wasm_f32x4_neg(x));
        wasm_v128_store(
            values + i,
            wasm_f32x4_div(x, wasm_f32x4_add(one, e))
        );
    }
    for (; i < length; i++) {
        const float x = values[i];
        values[i] = x / (1.0f + expfFast(-x));
    }
}

/** Fast exp with about 1e-7 relative error for non-positive inputs. */
static inline float expfFast(float x) {
    if (x < -87.0f) {
        return 0.0f;
    }
    if (x > 88.0f) {
        x = 88.0f;
    }
    const float z = x * 1.4426950408889634f;
    const float kf = __builtin_floorf(z + 0.5f);
    const int k = (int)kf;
    const float r = x - kf * 0.6931471805599453f;
    float p = 1.0f / 720.0f;
    p = 1.0f / 120.0f + r * p;
    p = 1.0f / 24.0f + r * p;
    p = 1.0f / 6.0f + r * p;
    p = 0.5f + r * p;
    p = 1.0f + r * p;
    p = 1.0f + r * p;
    union {
        float f;
        int i;
    } scale;
    scale.i = (k + 127) << 23;
    return p * scale.f;
}

__attribute__((visibility("default")))
void matmul(const float *input, const float *weight, const float *bias, float *out,
            int rows, int inDim, int outDim) {
    int i = 0;
    for (; i + MATMUL_ROWS <= rows; i += MATMUL_ROWS) {
        for (int j = 0; j < outDim; j++) {
            const float *w = weight + (long)j * inDim;
            const float b = bias ? bias[j] : 0.0f;
            for (int k = 0; k < MATMUL_ROWS; k++) {
                out[(long)(i + k) * outDim + j] =
                    dot4(input + (long)(i + k) * inDim, w, inDim) + b;
            }
        }
    }
    for (; i < rows; i++) {
        for (int j = 0; j < outDim; j++) {
            out[(long)i * outDim + j] =
                dot4(input + (long)i * inDim, weight + (long)j * inDim, inDim) +
                (bias ? bias[j] : 0.0f);
        }
    }
}

/** One dimensional convolution, channels first. */
__attribute__((visibility("default")))
void conv1d(const float *input, const float *weight, const float *bias, float *out,
            int inChannels, int outChannels, int inLength, int kernel,
            int stride, int padding) {
    const int outLength = (inLength + 2 * padding - kernel) / stride + 1;
    for (int oc = 0; oc < outChannels; oc++) {
        const float b = bias ? bias[oc] : 0.0f;
        float *outRow = out + (long)oc * outLength;
        for (int t = 0; t < outLength; t++) {
            float sum = b;
            const int base = t * stride - padding;
            if (base >= 0 && base + kernel <= inLength) {
                for (int ic = 0; ic < inChannels; ic++) {
                    const float *inRow = input + (long)ic * inLength + base;
                    const float *wRow =
                        weight + ((long)oc * inChannels + ic) * kernel;
                    for (int k = 0; k < kernel; k++) {
                        sum += inRow[k] * wRow[k];
                    }
                }
            } else {
                for (int ic = 0; ic < inChannels; ic++) {
                    const float *inRow = input + (long)ic * inLength;
                    const float *wRow =
                        weight + ((long)oc * inChannels + ic) * kernel;
                    for (int k = 0; k < kernel; k++) {
                        const int index = base + k;
                        if (index >= 0 && index < inLength) {
                            sum += inRow[index] * wRow[k];
                        }
                    }
                }
            }
            outRow[t] = sum;
        }
    }
}

__attribute__((visibility("default")))
void conv2d(const float *input, const float *weight, const float *bias, float *out,
            int inChannels, int inHeight, int inWidth, int outChannels,
            int kernel, int stride, int groups) {
    const int pad = (kernel - 1) / 2;
    const int inPerGroup = inChannels / groups;
    const int outPerGroup = outChannels / groups;
    const int height = (inHeight + 2 * pad - kernel) / stride + 1;
    const int width = (inWidth + 2 * pad - kernel) / stride + 1;
    for (int oc = 0; oc < outChannels; oc++) {
        const int group = groups > 1 ? oc / outPerGroup : 0;
        const float b = bias ? bias[oc] : 0.0f;
        float *outBase = out + (long)oc * height * width;
        for (int oh = 0; oh < height; oh++) {
            for (int ow = 0; ow < width; ow++) {
                float sum = b;
                for (int icg = 0; icg < inPerGroup; icg++) {
                    const int ic = group * inPerGroup + icg;
                    const float *inBase = input + (long)ic * inHeight * inWidth;
                    const float *wBase = weight +
                        ((long)oc * inPerGroup + icg) * kernel * kernel;
                    for (int kh = 0; kh < kernel; kh++) {
                        const int ih = oh * stride + kh - pad;
                        if (ih < 0 || ih >= inHeight) {
                            continue;
                        }
                        const float *inRow = inBase + (long)ih * inWidth;
                        for (int kw = 0; kw < kernel; kw++) {
                            const int iw = ow * stride + kw - pad;
                            if (iw < 0 || iw >= inWidth) {
                                continue;
                            }
                            sum += inRow[iw] * wBase[kh * kernel + kw];
                        }
                    }
                }
                outBase[(long)oh * width + ow] = sum;
            }
        }
    }
}

/** First byte after the static data and stack of the module. */
__attribute__((visibility("default")))
int arenaBase(void) {
    extern unsigned char __heap_base;
    return (int)(unsigned long)&__heap_base;
}

__attribute__((visibility("default")))
void permute(const float *src, float *dst, int outer, int inner) {
    for (int i = 0; i < outer; i++) {
        const float *row = src + (long)i * inner;
        float *column = dst + i;
        for (int j = 0; j < inner; j++) {
            column[(long)j * outer] = row[j];
        }
    }
}

/** 1x1 stride one convolution as a transpose plus matmul. */
__attribute__((visibility("default")))
void pointwise(const float *input, const float *weight, const float *bias,
               float *out, float *tmpIn, float *tmpOut, int channels,
               int outChannels, int spatial) {
    permute(input, tmpIn, channels, spatial);
    matmul(tmpIn, weight, bias, tmpOut, spatial, channels, outChannels);
    permute(tmpOut, out, spatial, outChannels);
}

__attribute__((visibility("default")))
void relativeBias(const float *q, const float *relK, float *out, float *bd,
                  int seq, int heads, int headDim, float scale) {
    const int p = 2 * seq - 1;
    const int hidden = heads * headDim;
    const int width = 2 * seq;
    for (int h = 0; h < heads; h++) {
        const int offset = h * headDim;
        for (int i = 0; i < seq; i++) {
            const float *qRow = q + (long)i * hidden + offset;
            for (int j = 0; j < p; j++) {
                const float *rRow = relK + (long)j * hidden + offset;
                bd[(long)i * p + j] = dot4(qRow, rRow, headDim) * scale;
            }
        }
        float *dst = out + (long)h * seq * seq;
        for (int i = 0; i < seq; i++) {
            for (int j = 0; j < seq; j++) {
                const long m = (long)i * p + j;
                const long index = (m / seq + 1) * (long)seq + (m % seq);
                const long row = index / width;
                const long col = index % width;
                dst[(long)i * seq + j] =
                    col == 0 ? 0.0f : bd[row * p + col - 1];
            }
        }
    }
}

__attribute__((visibility("default")))
void attention(const float *q, const float *k, const float *v, const float *bias,
               float *out, float *scores, int seqQ, int seqK, int heads,
               int headDim, float scale, int causal) {
    const int dim = heads * headDim;
    const int block = 16;
    for (int h = 0; h < heads; h++) {
        const int offset = h * headDim;
        for (int start = 0; start < seqQ; start += block) {
            const int rows = seqQ - start < block ? seqQ - start : block;
            for (int i = 0; i < rows; i++) {
                const int position = start + i;
                const float *qRow = q + (long)position * dim + offset;
                float *score = scores + (long)i * seqK;
                float max = -3.402823466e+38f;
                for (int j = 0; j < seqK; j++) {
                    const float *kRow = k + (long)j * dim + offset;
                    float value = dot4(qRow, kRow, headDim) * scale;
                    if (bias) {
                        value += bias[((long)h * seqQ + position) * seqK + j];
                    }
                    if (causal && j > position) {
                        value = -3.402823466e+38f;
                    }
                    score[j] = value;
                    if (value > max) {
                        max = value;
                    }
                }
                float sum = 0.0f;
                for (int j = 0; j < seqK; j++) {
                    const float e = expfFast(score[j] - max);
                    score[j] = e;
                    sum += e;
                }
                const float inv = 1.0f / sum;
                float *outRow = out + (long)position * dim + offset;
                for (int d = 0; d < headDim; d++) {
                    outRow[d] = 0.0f;
                }
                for (int j = 0; j < seqK; j++) {
                    const float weight = score[j] * inv;
                    if (weight == 0.0f) {
                        continue;
                    }
                    const float *vRow = v + (long)j * dim + offset;
                    int d = 0;
                    const v128_t w = wasm_f32x4_splat(weight);
                    for (; d + 3 < headDim; d += 4) {
                        const v128_t current = wasm_v128_load(outRow + d);
                        const v128_t add = wasm_v128_load(vRow + d);
                        wasm_v128_store(
                            outRow + d,
                            wasm_f32x4_add(current, wasm_f32x4_mul(w, add))
                        );
                    }
                    for (; d < headDim; d++) {
                        outRow[d] += weight * vRow[d];
                    }
                }
            }
        }
    }
}
