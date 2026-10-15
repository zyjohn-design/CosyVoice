#!/usr/bin/env python3
"""Serve the standalone test page and forward TTS requests using only stdlib."""

import argparse
import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Keep all traffic on the explicitly configured upstream.
        return None


class TestServer(ThreadingHTTPServer):
    daemon_threads = True


def make_handler(target, page):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, message):
            body = json.dumps({"code": 1, "msg": message}, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path in ("/", "/cosyvoice-test.html"):
                marker = '<meta name="cosyvoice-local-proxy" content="' + html.escape(target, quote=True) + '">'
                body = page.replace("</head>", marker + "\n</head>").encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif path in ("/api/health", "/api/speakers"):
                self.forward(path, None)
            else:
                self.send_json(404, "页面不存在")

        def do_POST(self):
            path = urlsplit(self.path).path
            if path != "/api/tts":
                self.send_json(404, "接口不存在")
                return
            # Cross-origin pages must not drive this local forwarding service.
            origin = self.headers.get("Origin")
            if origin and origin != "http://" + self.headers.get("Host", ""):
                self.send_json(403, "请从本地启动器提供的页面生成语音")
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.send_json(400, "请求长度无效")
                return
            if not 0 < size <= 1024 * 1024:
                self.send_json(413, "请求不能为空或超过 1 MB")
                return
            self.forward(path, self.rfile.read(size))

        def forward(self, path, body):
            headers = {"Accept": "*/*"}
            if body is not None:
                headers["Content-Type"] = self.headers.get("Content-Type", "application/x-www-form-urlencoded")
            request = Request(target + path.removeprefix("/api"), data=body, headers=headers, method=self.command)
            try:
                upstream = build_opener(ProxyHandler({}), NoRedirect()).open(request, timeout=300 if body is not None else 10)
            except HTTPError as error:
                upstream = error
            except (URLError, OSError) as error:
                self.send_json(502, f"本地转发器无法连接远端 {target}：{error}。请检查远端服务、网络/VPN和端口；此错误不是浏览器 CORS。")
                return
            with upstream:
                if 300 <= upstream.status < 400:
                    self.send_json(502, "远端返回重定向，请将 --target 设置为最终服务地址")
                    return
                self.send_response(upstream.status)
                for name in ("Content-Type", "Content-Length", "Content-Disposition"):
                    value = upstream.headers.get(name)
                    if value:
                        self.send_header(name, value)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    while True:
                        chunk = upstream.read(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                except (OSError, TimeoutError):
                    # The response has started, so a second HTTP response is invalid.
                    self.close_connection = True

    return Handler


def main():
    parser = argparse.ArgumentParser(description="CosyVoice 独立测试页本地启动器（无需第三方依赖）")
    parser.add_argument("--target", required=True, help="远端地址，例如 http://10.16.1.86:8080")
    parser.add_argument("--port", type=int, default=8088, help="本地端口（默认 8088）")
    args = parser.parse_args()
    target = args.target.strip()
    if "://" not in target:
        target = "http://" + target
    parsed = urlsplit(target)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        parser.error("--target 必须是 HTTP/HTTPS 服务根地址，不含账号、查询参数或片段")
    if not 1 <= args.port <= 65535:
        parser.error("--port 必须在 1 到 65535 之间")
    page_path = Path(__file__).resolve().parents[1] / "cosyvoice-test.html"
    if not page_path.is_file():
        parser.error(f"找不到页面：{page_path}，请保留 HTML 文件与 tools/ 的相对位置")
    try:
        server = TestServer(("127.0.0.1", args.port), make_handler(target.rstrip("/"), page_path.read_text(encoding="utf-8")))
    except OSError as error:
        parser.error(f"无法监听本地端口：{error}，可用 --port 指定其他端口")
    print(f"测试页面：http://127.0.0.1:{args.port}/", flush=True)
    print(f"远端服务：{target}（本地转发，无需远端 CORS）", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
