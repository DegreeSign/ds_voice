// Bundled reference codes for the MOSS male voice.
const
    /** Little endian uint16 codes, 42 frames by 16 codebooks. */
    REFERENCE_BASE64 =
        "4gHCAJ4C8QCNA5YBoQFyAqsATgGbAxEBZgOxAhABMQDiAcIAngLxAI0DlgGhAXICqwBOAZsDEQFm" +
        "A7ECEAExAFMBrwIDAvEAgQBwAJQDegOrAPABMgCvAkwAsAEDA6sA8AEwAe0C8QDAAeQDxgP3AqcA" +
        "WAASAhoCegBPAYIBmgNEA5ECeQIVACsC+gEPAZ4DOADEAsUBZgNXAiQBbQLHAocCOAIoAkEAvwKi" +
        "AxoA/AIFAHYBLQCZAzIBVALxA84D/wDlA8gDOwNUAWMCYQO2AcoC7ANQA3oAkwLNABsBhQGNAwEC" +
        "EwICA44D0AHUAL0BjQO2ASUCtALjAFQB/wK3AsQAfQAgAQgC8APcAmEDBwIOAkoBHQD9AT0CmgBb" +
        "AJoBXQH9AAUBTgMAAo4AagLgABYCSgHTAGwDnAGgAaYD2wPlArkAcgN2AKcBnACSAJgDzADcAHkC" +
        "6wAKANMA/QJbAg0CWwMMAF0DTQMuAqMC+wLxApkBrgMOAi8B2QEKATkBZAAmALECnwIhAAsANAJ9" +
        "ApkDlwG8AkgA0wALAJgCFgA0AlUDcAOlAcoD5gI8AIgD/QAdA/ADxAMAAzAB1QL4ACoBPQOHAaQC" +
        "QQFjAugDwwEUA48B8wGLAawBpQNUAF0DNgGNAA8BZgK5AvICFwI4ArsA+QMqAhsCngNyAGkCLwED" +
        "Aw0ACwBDAgQDbAN8A10BRwFLAKQCpAHXApIDsQDxA1wBCQBMA78BiwM+AJwC8QE0AnEAmQOkAboA" +
        "nACVAEABZwLyAJwAMwAFAXgDmwL3A9ACDgBmAS0BpwBsA1MD2ALiAcsCfwP7ANcBlwCzAOAAiQIc" +
        "AJkA8QCuAnAAowHEAVwB+QEpAHAAVQEKAKwCRgGyAogC7wDmASoACgPIAREBEwDDAoADSANxACkB" +
        "pAAoAbMADgBFA8QDJQKGArcBLABeA0kDawAaA68AcAGqAykBlQDBAnMDFwAfAsEDOQGvARMAmgI7" +
        "AwQBagNgAQ0DWAJOABcCIAKYAQcBuwPpAuEBswPuAcsBlAJXAQQCcgOQAU8D0AMjAGcA/wMNA2UD" +
        "cgH7ADoCggDpAUQCuwCZA54AEwIcAc0A2wGHAvUCegIhA6sAwQDJAF8C+ADeAOgA5wK3AOoCrALh" +
        "Ae0BFgKwAXcCuwLVAYgC7AJhA8EAYwH7ARoARwBVAJAC3wGeA80CFQCgAHID3QPRAVYDlgI5AloA" +
        "SwCaAicAbAIyArUCawIkAD0CvAP6ARgDswFyA+YDUAGtA5UA9QMBAREAcQAbAugDcwAzAQ8B4gFk" +
        "ApgCNAH7AXUC/gL5AbIDMwLHA/QAZgCJAqQBxQJ4AnIA5wHlA4sCrQPNAXoCygBrASsBjgFxAxMD" +
        "1gLjAIcAVwJcArkAMAARARACJgGjA6ACHQNwAkUDEwCxA08BIgAOAVUCMALRA9wCWgC9A58CvQAX" +
        "An0AQgATAOIAjgDlAC0DWADhAzkAYgBrA+EB8wNzA4cBdACxA1oC7gFuASsCjQOMA+wDSANDA/AB" +
        "9AGbAMMBlwFAA9kAWgJJAxUADAKNA5gBzQMFAY0B6AEOAxQCvAO+AWQALwJaAkkDsANVAY0DlgCp" +
        "A0wDMANfAIwCrwKaADgByAEhA1oCYwHVApQAjQOWANwAVgM0A9UBRADpAGYDWwESATEAUwENANoB" +
        "LgH6AKYDfAAwA2sB1QEyAN4BcAA6AiAAYQPiAcIAngIjAI0D/wAvAZQBlwGMAXoCPAMVArEC9APL" +
        "AuIBwgCeAvEAjQOWAaEBlAGXAU4B9QEVAnAAsQIQATEA",
    /** Decode the base64 reference codes. */
    decodeReference = (): Uint16Array => {
        const
            binary = atob(REFERENCE_BASE64),
            bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i += 1) {
            bytes[i] = binary.charCodeAt(i);
        }
        const out = new Uint16Array(bytes.length / 2);
        for (let i = 0; i < out.length; i += 1) {
            out[i] = bytes[i * 2] | (bytes[i * 2 + 1] << 8);
        }
        return out;
    },
    /** Fixed reference codes of the MOSS voice. */
    MOSS_REFERENCE = decodeReference();

export { MOSS_REFERENCE };
