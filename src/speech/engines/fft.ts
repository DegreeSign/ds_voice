// Complex FFT and inverse short time Fourier transform helpers.

/** In place radix two complex FFT. */
const fft = (options: {
    re: Float32Array;
    im: Float32Array;
    inverse: boolean;
}): void => {
    const
        { re, im, inverse } = options,
        size = re.length;
    for (let i = 1, j = 0; i < size; i += 1) {
        let bit = size >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            const
                tr = re[i],
                ti = im[i];
            re[i] = re[j];
            im[i] = im[j];
            re[j] = tr;
            im[j] = ti;
        }
    }
    for (let len = 2; len <= size; len <<= 1) {
        const
            angle = (inverse ? 2 : -2) * Math.PI / len,
            wRe = Math.cos(angle),
            wIm = Math.sin(angle);
        for (let start = 0; start < size; start += len) {
            let
                curRe = 1,
                curIm = 0;
            for (let k = 0; k < len / 2; k += 1) {
                const
                    a = start + k,
                    b = a + len / 2,
                    vRe = re[b] * curRe - im[b] * curIm,
                    vIm = re[b] * curIm + im[b] * curRe;
                re[b] = re[a] - vRe;
                im[b] = im[a] - vIm;
                re[a] += vRe;
                im[a] += vIm;
                const nextRe = curRe * wRe - curIm * wIm;
                curIm = curRe * wIm + curIm * wRe;
                curRe = nextRe;
            }
        }
    }
    if (inverse) {
        for (let i = 0; i < size; i += 1) {
            re[i] /= size;
            im[i] /= size;
        }
    }
};

/** Periodic Hann window. */
const hannWindow = (size: number): Float32Array => {
    const window = new Float32Array(size);
    for (let i = 0; i < size; i += 1) {
        window[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / size);
    }
    return window;
};

/** Inverse STFT with centered frames, matching torch.istft. */
const inverseStft = (options: {
    magnitude: Float32Array;
    phase: Float32Array;
    bins: number;
    frames: number;
    nFft: number;
    hop: number;
}): Float32Array => {
    const
        { magnitude, phase, bins, frames, nFft, hop } = options,
        window = hannWindow(nFft),
        pad = nFft >> 1,
        length = (frames - 1) * hop + nFft,
        output = new Float32Array(length),
        envelope = new Float32Array(length),
        re = new Float32Array(nFft),
        im = new Float32Array(nFft);
    for (let frame = 0; frame < frames; frame += 1) {
        const offset = frame * hop;
        re.fill(0);
        im.fill(0);
        // torch.istft with center padding clears the DC and Nyquist bins.
        for (let bin = 1; bin < bins - 1; bin += 1) {
            const
                index = frame * bins + bin,
                mag = magnitude[index],
                angle = phase[index];
            re[bin] = mag * Math.cos(angle);
            im[bin] = mag * Math.sin(angle);
        }
        for (let bin = 1; bin < bins - 1; bin += 1) {
            re[nFft - bin] = re[bin];
            im[nFft - bin] = -im[bin];
        }
        fft({ re, im, inverse: true });
        for (let i = 0; i < nFft; i += 1) {
            const w = window[i];
            output[offset + i] += re[i] * w;
            envelope[offset + i] += w * w;
        }
    }
    const result = output.subarray(pad, length - pad);
    for (let i = 0; i < result.length; i += 1) {
        const value = envelope[pad + i];
        result[i] = value > 1e-11 ? result[i] / value : 0;
    }
    return result;
};

export { inverseStft };
