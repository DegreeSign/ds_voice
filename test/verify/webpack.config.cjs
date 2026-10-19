// Bundles the browser harness.
// Usage: npx webpack --config test/verify/webpack.config.cjs
const path = require("path");
const webpack = require("webpack");

module.exports = {
    entry: "./test/verify/harness.ts",
    target: "web",
    mode: "development",
    devtool: false,
    resolve: { extensions: [".ts", ".js"] },
    plugins: [
        new webpack.NormalModuleReplacementPlugin(
            /^\.\/node$/,
            path.resolve(__dirname, "../../src/store/nodeStub.ts"),
        ),
        new webpack.NormalModuleReplacementPlugin(
            /^\.\/nodeFile$/,
            path.resolve(__dirname, "../../src/audio/nodeFileStub.ts"),
        ),
        new webpack.NormalModuleReplacementPlugin(
            /^\.\/poolNode$/,
            path.resolve(__dirname, "../../src/tensor/poolStub.ts"),
        ),
    ],
    module: {
        rules: [
            {
                test: /\.ts$/,
                use: {
                    loader: "ts-loader",
                    options: { transpileOnly: true, configFile: "tsconfig.json" },
                },
                exclude: /node_modules/,
            },
        ],
    },
    output: {
        path: path.resolve(__dirname, "../../build"),
        filename: "harness.js",
    },
};
