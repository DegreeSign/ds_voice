// Static server for the manual recorder and demo pages.
// Usage: node test/example/serve.cjs [port]
const http = require("http");
const fs = require("fs");
const path = require("path");

const
    ROOT = path.resolve(__dirname, "../.."),
    PORT = Number(process.argv[2] || 17544),
    MIME = {
        ".html": "text/html",
        ".js": "text/javascript",
        ".css": "text/css",
        ".json": "application/json",
        ".wav": "audio/wav",
        ".bin": "application/octet-stream",
    },
    serve = () => {
        const server = http.createServer((request, response) => {
            const
                url = decodeURIComponent((request.url || "/").split("?")[0]),
                file = path.join(ROOT, url.replace(/^\/+/, ""));
            if (url === "/" || url === "/voice") {
                response.writeHead(302, {
                    location: url === "/voice"
                        ? "/docs/index.html"
                        : "/test/example/record.html",
                });
                response.end();
                return;
            }
            if (
                !file.startsWith(ROOT) ||
                !fs.existsSync(file) ||
                fs.statSync(file).isDirectory()
            ) {
                response.writeHead(404);
                response.end("not found");
                return;
            }
            response.writeHead(200, {
                "content-type":
                    MIME[path.extname(file)] || "application/octet-stream",
            });
            fs.createReadStream(file).pipe(response);
        });
        server.listen(PORT, "127.0.0.1", () => {
            console.log(
                `Recorder page: http://127.0.0.1:${PORT}/test/example/record.html`,
            );
            console.log(
                `Voice demo: http://127.0.0.1:${PORT}/docs/index.html`,
            );
        });
        server.on("error", (error) => {
            console.error("server error", error.message);
            process.exit(4);
        });
        return server;
    },
    server = serve();

process.on("SIGINT", () => {
    server.close();
});
