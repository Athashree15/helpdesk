"""Tracing, feedback capture and monitoring summaries (the "Ops" in LLMOps).

Every Model-3 request is written to a SQLite table with latency per stage, token
usage, retrieved sources, guardrail/cache/escalation flags and the prompt version.
User feedback (thumbs up/down) is stored against the same trace id.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
  trace_id TEXT PRIMARY KEY, ts REAL, session_id TEXT, model TEXT, prompt_version TEXT,
  question TEXT, answer TEXT, latency_s REAL, prompt_tokens INT, completion_tokens INT,
  blocked INT, block_reason TEXT, cache_hit INT, escalated INT, disclaimer INT,
  retrieval_confidence REAL, groundedness REAL, pii_masked TEXT, sources TEXT, spans TEXT
);
CREATE TABLE IF NOT EXISTS feedback (
  trace_id TEXT, ts REAL, rating INT, comment TEXT
);
"""


class Tracer:
    def __init__(self, cfg: dict[str, Any], db_name: str = "traces.db"):
        self.path = Path(cfg["paths"]["storage_dir"]) / db_name
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def log(self, trace_id: str, session_id: str, question: str, response, model: str) -> None:
        m = response.meta
        row = (trace_id, time.time(), session_id, model, m.get("prompt_version"), question,
               response.answer, response.latency_s, response.prompt_tokens, response.completion_tokens,
               int(m.get("blocked", False)), m.get("block_reason"), int(m.get("cache_hit", False)),
               int(m.get("escalated", False)), int(m.get("disclaimer", False)),
               m.get("retrieval_confidence"), m.get("groundedness"),
               json.dumps(m.get("pii_masked", [])),
               json.dumps([{k: s[k] for k in ("source", "section", "score")} for s in response.sources]),
               json.dumps({k: round(v, 4) for k, v in m.get("spans", {}).items()}))
        with self._conn() as c:
            c.execute(f"INSERT OR REPLACE INTO traces VALUES ({','.join('?' * len(row))})", row)

    def feedback(self, trace_id: str, rating: int, comment: str = "") -> None:
        with self._conn() as c:
            c.execute("INSERT INTO feedback VALUES (?,?,?,?)", (trace_id, time.time(), rating, comment))

    def rows(self, session_id: str | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            q = "SELECT * FROM traces"
            args: tuple = ()
            if session_id:
                q += " WHERE session_id = ?"
                args = (session_id,)
            q += " ORDER BY ts DESC LIMIT ?"
            return [dict(r) for r in c.execute(q, args + (limit,))]

    def summary(self, session_id: str | None = None) -> dict[str, Any]:
        rows = self.rows(session_id)
        if not rows:
            return {"requests": 0}
        lat = np.array([r["latency_s"] for r in rows])
        with self._conn() as c:
            fb = c.execute("SELECT AVG(rating), COUNT(*) FROM feedback").fetchone()
        g = [r["groundedness"] for r in rows if r["groundedness"] is not None]
        n = len(rows)
        return {
            "requests": n,
            "latency_p50_s": float(np.percentile(lat, 50)),
            "latency_p95_s": float(np.percentile(lat, 95)),
            "block_rate": sum(r["blocked"] for r in rows) / n,
            "cache_hit_rate": sum(r["cache_hit"] for r in rows) / n,
            "escalation_rate": sum(r["escalated"] for r in rows) / n,
            "disclaimer_rate": sum(r["disclaimer"] for r in rows) / n,
            "avg_groundedness": float(np.mean(g)) if g else None,
            "tokens_total": int(sum((r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0) for r in rows)),
            "feedback_avg": fb[0], "feedback_count": fb[1],
        }
