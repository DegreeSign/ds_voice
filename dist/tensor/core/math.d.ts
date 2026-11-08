declare const 
/** Next power of two at or above x. */
nextPow2: (x: number) => number, 
/** Reflect index into a finite range. */
reflect: (options: {
    index: number;
    length: number;
}) => number;
export { nextPow2, reflect };
