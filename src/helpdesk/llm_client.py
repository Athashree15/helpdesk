"""Thin client for a locally running Ollama server (https://ollama.com).

Everything runs on the student's own machine: no data leaves the laptop and no
API key is needed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    model: str


class OllamaClient:
    def __init__(self, cfg: dict[str, Any]):
        o = cfg["ollama"]
        self.base_url = o["base_url"].rstrip("/")
        self.chat_model = o["chat_model"]
        self.embed_model = o["embed_model"]
        self.temperature = float(o.get("temperature", 0.1))
        self.num_ctx = int(o.get("num_ctx", 4096))
        self.timeout = int(o.get("timeout_s", 300))
        self.session = requests.Session()
        self.is_mock = False

    # ------------------------------------------------------------------ health
    def health(self) -> tuple[bool, str]:
        try:
            r = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            r.raise_for_status()
        except Exception as exc:  # noqa: BLE001
            return False, (f"Cannot reach Ollama at {self.base_url} ({exc}). "
                           "Install Ollama and run `ollama serve`.")
        names = {m.get("name", "") for m in r.json().get("models", [])}
        self.is_mock = any(n.startswith("mock-") for n in names)  # test server only

        def present(model: str) -> bool:
            if model in names or f"{model}:latest" in names:
                return True
            return ":" not in model and any(n.startswith(model + ":") for n in names)

        missing = [m for m in (self.chat_model, self.embed_model) if not present(m)]
        if missing:
            return False, "Missing models: " + ", ".join(missing) + \
                ". Run: " + " && ".join(f"ollama pull {m}" for m in missing)
        return True, f"Ollama OK - chat={self.chat_model}, embed={self.embed_model}"

    # -------------------------------------------------------------------- chat
    def chat(self, messages: list[dict[str, str]], temperature: float | None = None,
             model: str | None = None, json_mode: bool = False) -> LLMResult:
        payload: dict[str, Any] = {
            "model": model or self.chat_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "num_ctx": self.num_ctx,
                "seed": 42,
            },
        }
        if json_mode:
            payload["format"] = "json"
        t0 = time.perf_counter()
        r = self.session.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        latency = time.perf_counter() - t0
        return LLMResult(
            text=(data.get("message") or {}).get("content", "").strip(),
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
            latency_s=latency,
            model=payload["model"],
        )

    # -------------------------------------------------------------- embeddings
    def embed(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            r = self.session.post(f"{self.base_url}/api/embed",
                                  json={"model": self.embed_model, "input": batch},
                                  timeout=self.timeout)
            if r.status_code == 404:  # very old Ollama versions only have /api/embeddings
                for t in batch:
                    r2 = self.session.post(f"{self.base_url}/api/embeddings",
                                           json={"model": self.embed_model, "prompt": t},
                                           timeout=self.timeout)
                    r2.raise_for_status()
                    vectors.append(r2.json()["embedding"])
                continue
            r.raise_for_status()
            vectors.extend(r.json()["embeddings"])
        return vectors
