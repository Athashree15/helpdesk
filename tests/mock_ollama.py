"""TEST-ONLY fake Ollama server used to check the plumbing without downloading models.

It is NOT a language model: answers are crude sentence picks from the prompt.
Results produced against it are meaningless, and the report generator stamps
them as a MOCK run. Use real Ollama for the assignment results.

    python tests/mock_ollama.py --port 11435
    set OLLAMA_BASE_URL=http://localhost:11435   (Windows)  /  export OLLAMA_BASE_URL=...  (macOS/Linux)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DIM = 384
WORD = re.compile(r"[a-z0-9]+")


def embed(text: str) -> list[float]:
    v = [0.0] * DIM
    for w in WORD.findall(text.lower()):
        if len(w) < 3:
            continue
        w = w[:6]  # crude stemming
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        v[h % DIM] += 1.0 if (h >> 9) & 1 else -1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def fake_chat(messages: list[dict]) -> str:
    user = messages[-1]["content"]
    if "Score the chatbot answer" in user:
        return json.dumps({"score": 3, "reason": "mock judge"})
    m = re.search(r"(?:CONTEXT:|Context:)(.*?)(?:STUDENT QUESTION:|Question:)(.*)", user, re.S)
    if not m:
        return "As per UGC norms, universities generally require 75% attendance. (mock answer)"
    ctx, q = m.group(1), m.group(2)
    qw = set(WORD.findall(q.lower()))
    best, best_s, best_i = "", -1, 1
    for block in re.split(r"\n(?=\[\d+\])", ctx):
        idx = re.match(r"\s*\[(\d+)\]", block)
        for s in re.split(r"(?<=[.!?])\s+|\n", block):
            sc = len(qw & set(WORD.findall(s.lower())))
            if sc > best_s and len(s) > 20 and not s.startswith("["):
                best, best_s, best_i = s.strip(), sc, int(idx.group(1)) if idx else 1
    return f"{best} [{best_i}]"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send({"models": [{"name": "mock-llm:latest"}, {"name": "mock-embed:latest"},
                                   {"name": "llama3.2:3b"}, {"name": "nomic-embed-text:latest"}]})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/embed":
            inp = data["input"] if isinstance(data["input"], list) else [data["input"]]
            self._send({"embeddings": [embed(t) for t in inp]})
        elif self.path == "/api/chat":
            text = fake_chat(data["messages"])
            self._send({"message": {"role": "assistant", "content": text},
                        "prompt_eval_count": sum(len(m["content"]) // 4 for m in data["messages"]),
                        "eval_count": len(text) // 4})
        else:
            self._send({"error": "not found"}, 404)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=11435)
    a = ap.parse_args()
    print(f"MOCK Ollama on http://localhost:{a.port} (test only)")
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
