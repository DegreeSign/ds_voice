import * as path from "path";
import * as webpack from "webpack";
import TerserPlugin from "terser-webpack-plugin";

const
    // Shared webpack configuration
    commonConfig: webpack.Configuration = {
        resolve: {
            extensions: [".ts", ".js"],
        },
        module: {
            rules: [
                {
                    test: /\.ts$/,
                    use: "ts-loader",
                    exclude: /node_modules/,
                },
            ],
        },
        plugins: [
            new webpack.BannerPlugin({
                banner: "/*! MIT License. DegreeSign Voice SDK */",
                raw: true,
            }),
        ],
        optimization: {
            minimize: true,
            minimizer: [new TerserPlugin()],
            usedExports: true,
            sideEffects: false,
        },
        mode: "production",
    },
    // Browser plugin set shared by the full and family entries
    browserPlugins: NonNullable<webpack.Configuration["plugins"]> = [
        ...(commonConfig.plugins ?? []),
        new webpack.NormalModuleReplacementPlugin(
            /^\.\/node$/,
            path.resolve(__dirname, "src/store/nodeStub.ts"),
        ),
        new webpack.NormalModuleReplacementPlugin(
            /^\.\/nodeFile$/,
            path.resolve(__dirname, "src/audio/nodeFileStub.ts"),
        ),
        new webpack.NormalModuleReplacementPlugin(
            /^\.\/poolNode$/,
            path.resolve(__dirname, "src/tensor/poolBrowser.ts"),
        ),
    ],
    // Browser webpack configuration
    browserConfig: webpack.Configuration = {
        ...commonConfig,
        entry: "./src/index.ts",
        target: "web",
        plugins: browserPlugins,
        output: {
            path: path.resolve(__dirname, "dist/browser"),
            filename: "degreesign.min.js",
            library: "dsVoice",
            libraryTarget: "umd",
            globalObject: "this",
        },
    },
    /** One browser bundle with a subset of the families. */
    familyConfig = (options: {
        entry: string;
        filename: string;
        library: string;
    }): webpack.Configuration => ({
        ...browserConfig,
        entry: options.entry,
        output: {
            path: path.resolve(__dirname, "dist/browser"),
            filename: options.filename,
            library: options.library,
            libraryTarget: "umd",
            globalObject: "this",
        },
    }),
    // Node webpack configuration
    nodeConfig: webpack.Configuration = {
        ...commonConfig,
        entry: "./src/index.ts",
        target: "node",
        resolve: {
            ...commonConfig.resolve,
            mainFields: ["module", "main"],
            conditionNames: ["import", "default"],
            aliasFields: [],
            preferRelative: true,
        },
        output: {
            path: path.resolve(__dirname, "dist/node"),
            filename: "degreesign.node.min.js",
            libraryTarget: "commonjs",
            globalObject:
                "typeof globalThis !== 'undefined' ? globalThis : this",
        },
        node: {
            __dirname: false,
            __filename: false,
        },
    };

module.exports = [
    browserConfig,
    familyConfig({
        entry: "./src/sensevoice.ts",
        filename: "sensevoice.min.js",
        library: "dsVoiceSenseVoice",
    }),
    familyConfig({
        entry: "./src/whisper.ts",
        filename: "whisper.min.js",
        library: "dsVoiceWhisper",
    }),
    nodeConfig,
];
