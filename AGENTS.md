# ds_voice

Standalone TypeScript package for offline speech recognition with the SenseVoice
and Whisper model families, published as `@degreesign/voice`. This folder is
self contained and does not follow any host repository rules.

# Skill
Load `coding` skill of `@degreesign/skills` (probably already installed)

## Rules

- Use `yarn` for package management.
- Keep the package free of runtime dependencies. `@degreesign/cache` and
  `@degreesign/storage` are bundled at build time.
- Put every interface and type in `src/types.ts`.
- Pass an object parameter when a function takes more than one input.
- Keep lines short, at most 100 columns.
- Chain a run of adjacent `const` or `let` declarations into one
  statement: the keyword on its own line, each definition indented below
  it, separated by commas, with the last one terminated by a semicolon.
- Keep plain comments to three or four words and start them with `//`. Use a
  `/** */` block only to document a specific defined variable or function.
- Indent every file with four spaces per level and end it with a single newline.
- Never use an em dash in a file of this repository.
- Model weights keep `THIRD_PARTIES.md` current (Apache-2.0 and the FunASR
  Model Open Source License v1.1).
- Model weights are cached through `@degreesign/cache` on Node and through
  `@degreesign/storage` on the browser. Never write dumps into the repository.
- Measure `TextSnapshot.msElapsed` and `TranscriptMeta.rtf` over decode work
  alone. Model download, model load, audio decoding and waiting for audio stay
  outside the measurement in the package and in the tests.
- Verify with `yarn typecheck` and `yarn build`, capped at 60 seconds per run.
- Never run release, publish or deploy commands.
- Node >= 24. The source is TypeScript built with webpack and `tsc`, imports
  have no file extension, and both bundles build from the shared entry
  `src/index.ts`.

## Verification

Run the package checks, then the headless browser suite.

Package checks, each capped at 60 seconds:

1. `yarn typecheck`.
2. `yarn build`.
3. `node -e "require('./dist/node/degreesign.node.min.js')"` to catch
   top level init errors.
4. When a store changes, round trip a temp folder through
   `createNodeStore` with `ts-node`.

Headless browser suite, for any runtime change:

1. Keep the harness in `test/verify/harness.ts`, importing the modules from
   `../../src/...` and asserting every exported function of both entries.
2. Bundle it: `npx webpack --config test/verify/webpack.config.cjs`.
3. Run it: `node test/verify/drive.cjs "$PWD/build"`.
4. The driver exits 0 only on `"fail": 0`.
5. Delete `build/` after the run. Never commit build output.

Harness rules:

- Put the JSON report in the existing `#results` element and log
  `TESTS_RESULT`. Never append a second element with that id.
- Defer every top level test into a function. A `const` builder placed
  below a test sits in the temporal dead zone at module load.
- A failing test records its error and the suite keeps running.

Coverage rules:

- Cover both entries and every exported function of every module.
- Use correctly shaped synthetic weights. Real checkpoints need minutes
  of compute in JavaScript, so never load one in the suite.
- Assert exact values for pure helpers, and assert no throw plus output
  shape for the encoders, decoders and frontends.

Chromium notes:

- Snap Chromium cannot write a profile in `~` or `~/.cache`. Use
  `~/snap/chromium/common/ds_voice_chrome`.
- Chromium blocks many ports below 10000 with `ERR_UNSAFE_PORT`. Use a
  high port such as `17543`.
- Launch with `--headless=new --no-sandbox --disable-gpu`.
- Read targets from `http://127.0.0.1:<port>/json/list` and speak CDP
  over the page `webSocketDebuggerUrl` with the Node global `WebSocket`.
