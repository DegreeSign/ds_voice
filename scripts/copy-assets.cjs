// Copy the package asset tree into dist.
const fs = require("fs");
const path = require("path");

const
    source = path.resolve(__dirname, "../src/assets"),
    target = path.resolve(__dirname, "../dist/assets");

if (!fs.existsSync(source)) {
    process.exit(0);
}
fs.rmSync(target, { recursive: true, force: true });
fs.cpSync(source, target, { recursive: true });
console.log(`copied ${path.relative(process.cwd(), target)}`);
