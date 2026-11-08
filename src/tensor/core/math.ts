// Small numeric helpers.
const
    /** Next power of two at or above x. */
    nextPow2 = (x: number): number => {
        let p = 1;
        while (p < x) {
            p <<= 1;
        }
        return p;
    },
    /** Reflect index into a finite range. */
    reflect = (options: { index: number; length: number }): number => {
        const { length } = options;
        let i = options.index;
        while (i < 0 || i >= length) {
            i = i < 0 ? -i : 2 * (length - 1) - i;
        }
        return i;
    };

export { nextPow2, reflect };
