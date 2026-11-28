"""A tiny OpenAI-compatible chat server (standard library only) for testing the evaluation and benchmark clients.

It answers each request with a canned text looked up by the SHA-256 of the last user message, so serving the
frozen test set's targets must score perfectly. It is NOT a model: it is used by the tests and by the notebook's
CPU smoke run, never to report a quality or speed number.

  python llm_finetune/stub_server.py --answers answers.jsonl --port 8011 [--mode targets|garbage] [--token-delay 0.002]

answers.jsonl lines: {"prompt_sha256": "...", "text": "..."}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UNKNOWN = '{"answer": "I could not find this in the records you are allowed to access.", "claims": []}'


def make_handler(answers: dict[str, str], model: str, mode: str, token_delay: float):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def _json(self, code: int, body: dict) -> None:
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._json(200, {"status": "ok"})
            elif self.path == "/v1/models":
                self._json(200, {"object": "list", "data": [{"id": model, "object": "model"}]})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self._json(404, {"error": "not found"})
                return
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            prompt = next((m["content"] for m in reversed(body["messages"]) if m["role"] == "user"), "")
            text = "this is not JSON" if mode == "garbage" else answers.get(
                hashlib.sha256(prompt.encode()).hexdigest(), UNKNOWN)
            pieces = [w + " " for w in text.split(" ")]
            pieces[-1] = pieces[-1][:-1]
            usage = {"prompt_tokens": len(prompt) // 4, "completion_tokens": len(pieces),
                     "total_tokens": len(prompt) // 4 + len(pieces)}
            if not body.get("stream"):
                time.sleep(token_delay * len(pieces))
                self._json(200, {"id": "stub", "object": "chat.completion", "model": model, "usage": usage,
                                 "choices": [{"index": 0, "finish_reason": "stop",
                                              "message": {"role": "assistant", "content": text}}]})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()

            def send(obj) -> None:
                self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
                self.wfile.flush()

            for p in pieces:
                time.sleep(token_delay)
                send({"id": "stub", "object": "chat.completion.chunk", "model": model,
                      "choices": [{"index": 0, "delta": {"content": p}, "finish_reason": None}]})
            send({"id": "stub", "object": "chat.completion.chunk", "model": model,
                  "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            if (body.get("stream_options") or {}).get("include_usage"):
                send({"id": "stub", "object": "chat.completion.chunk", "model": model, "choices": [], "usage": usage})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    return Handler


def serve(answers: dict[str, str], port: int = 0, model: str = "stub", mode: str = "targets",
          token_delay: float = 0.0) -> ThreadingHTTPServer:
    """Start in a background thread; the bound port is server.server_address[1]."""
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(answers, model, mode, token_delay))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def load_answers(path: str) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        return {r["prompt_sha256"]: r["text"] for r in map(json.loads, f) if r}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--model", default="stub")
    ap.add_argument("--mode", choices=["targets", "garbage"], default="targets")
    ap.add_argument("--token-delay", type=float, default=0.0)
    a = ap.parse_args()
    srv = serve(load_answers(a.answers), a.port, a.model, a.mode, a.token_delay)
    print(f"stub server on http://127.0.0.1:{srv.server_address[1]}/v1", flush=True)
    threading.Event().wait()
