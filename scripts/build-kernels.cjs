// Rebuild the embedded SIMD kernels from src/tensor/kernels.c.
// Usage: node scripts/build-kernels.cjs
const
    { execFileSync } = require("node:child_process"),
    { readFileSync, writeFileSync, mkdtempSync, rmSync } = require("node:fs"),
    { tmpdir } = require("node:os"),
    path = require("node:path"),
    root = path.resolve(__dirname, ".."),
    source = path.join(root, "src", "tensor", "kernels.c"),
    target = path.join(root, "src", "tensor", "kernelsWasm.ts"),
    EXPORTS = [
        "matmul",
        "attention",
        "conv1d",
        "conv2d",
        "pointwise",
        "relativeBias",
        "silu",
        "gelu",
        "arenaBase",
    ],
    VARIANTS = [
        {
            name: "KERNEL_BASE64",
            flags: [],
        },
        {
            name: "KERNEL_SHARED_BASE64",
            flags: [
                "-matomics",
                "-mbulk-memory",
                "-Wl,--shared-memory",
                "-Wl,--max-memory=4294967296",
            ],
        },
    ],
    /** Compile one variant into bytes. */
    compile = (options) => {
        const
            folder = mkdtempSync(path.join(tmpdir(), "ds-voice-kernels-")),
            out = path.join(folder, "kernels.wasm"),
            args = [
                "--target=wasm32",
                "-O3",
                "-msimd128",
                "-nostdlib",
                "-Wl,--no-entry",
                "-Wl,--import-memory",
                ...EXPORTS.map((name) => `-Wl,--export=${name}`),
                ...options.flags,
                "-Wl,--allow-undefined",
                "-o",
                out,
                source,
            ];
        try {
            execFileSync("clang-21", args, { stdio: "inherit" });
            return readFileSync(out);
        } finally {
            rmSync(folder, { recursive: true, force: true });
        }
    },
    /** Split base64 into 100 character lines. */
    lines = (text) => {
        const rows = [];
        for (let index = 0; index < text.length; index += 100) {
            rows.push(`    "${text.slice(index, index + 100)}",`);
        }
        return rows.join("\n");
    },
    /** Rewrite one embedded constant. */
    embed = (text, name, bytes) => {
        const
            start = text.indexOf(`const ${name} = [`),
            end = text.indexOf("].join", start),
            head = text.slice(0, start),
            tail = text.slice(end);
        return `${head}const ${name} = [\n${
            lines(bytes.toString("base64"))
        }\n${tail}`;
    };

let text = readFileSync(target, "utf8");
for (const variant of VARIANTS) {
    text = embed(text, variant.name, compile(variant));
}
writeFileSync(target, text);
console.log("kernels embedded", target);
