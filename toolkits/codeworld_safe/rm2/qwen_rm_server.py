"""HTTP scoring server for the Qwen3-VL success RM (one process per GPU, runs in /data_cf/qwen/venv).

  CUDA_VISIBLE_DEVICES=g python qwen_rm_server.py --port 29700+g [--lora ADAPTER_DIR] [--bs 32]
POST /score {"jpg": [base64 JPEG, ...], "instructions": [str, ...]} -> {"p": [float, ...]}; GET /health -> ok.
Requests are scored one at a time (one GPU per process).
"""
import argparse
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import qwen_rm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()
    model, proc = qwen_rm.load(a.lora)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(200 if self.path == "/health" else 404, {"ok": self.path == "/health"})

        def do_POST(self):
            try:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                imgs = [qwen_rm.decode(base64.b64decode(x)) for x in req["jpg"]]
                if len(imgs) != len(req["instructions"]):
                    raise ValueError("jpg and instructions differ in length")
                with lock:
                    p = qwen_rm.score_images(model, proc, imgs, req["instructions"], a.bs)
                self._send(200, {"p": [float(x) for x in p]})
            except Exception as exc:  # reported to the client, server keeps running
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, *args):
            pass

    print(f"qwen rm server on 127.0.0.1:{a.port} lora={a.lora}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
