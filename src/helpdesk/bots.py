"""The three chatbot models compared in this assignment.

Model 1 - LLM only      : a local LLM answers from its pre-trained knowledge.
Model 2 - RAG           : the LLM answers from passages retrieved from the CHRIST knowledge base.
Model 3 - LLMOps (RAG+) : the RAG pipeline wrapped with production operations - guardrails,
                          versioned prompts, semantic cache, confidence-based escalation,
                          groundedness checks, tracing, feedback and experiment tracking.
"""
from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .guardrails import check_input, scrub_output
from .knowledge_base import Hit, KnowledgeBase, format_context
from .llm_client import OllamaClient
from .metrics import groundedness
from .observability import Tracer

ESCALATION_MSG = ("I couldn't find this in the helpdesk knowledge base, so I don't want to guess. "
                  "Please contact the relevant office (for example the Office of Student Affairs, "
                  "the Office of Examinations or your campus admissions office), or ask your class teacher.")


@dataclass
class BotResponse:
    answer: str
    model_name: str
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    sources: list[dict[str, Any]] = field(default_factory=list)
    context: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


def _hits_to_sources(hits: list[Hit]) -> list[dict[str, Any]]:
    return [{"rank": i + 1, "source": h.chunk.source, "section": h.chunk.section,
             "score": round(h.score, 4),
             "dense_score": None if math.isnan(h.dense_score) else round(h.dense_score, 4),
             "verified": h.chunk.meta.get("verified"), "text": h.chunk.text}
            for i, h in enumerate(hits)]


# =========================================================================== M1
class BaselineLLMBot:
    name = "Model 1: LLM only"
    key = "llm"

    def __init__(self, llm: OllamaClient, prompts: dict[str, Any]):
        self.llm = llm
        self.system = prompts["baseline_system"]["template"]

    def answer(self, question: str) -> BotResponse:
        t0 = time.perf_counter()
        res = self.llm.chat([{"role": "system", "content": self.system},
                             {"role": "user", "content": question}])
        return BotResponse(answer=res.text, model_name=self.name, latency_s=time.perf_counter() - t0,
                           prompt_tokens=res.prompt_tokens, completion_tokens=res.completion_tokens)


# =========================================================================== M2
class RAGBot:
    name = "Model 2: RAG"
    key = "rag"

    def __init__(self, llm: OllamaClient, kb: KnowledgeBase, prompts: dict[str, Any],
                 prompt_version: str = "v1", top_k: int | None = None, mode: str | None = None):
        self.llm, self.kb = llm, kb
        self.prompt = prompts["versions"][prompt_version]
        self.prompt_version = prompt_version
        self.top_k, self.mode = top_k, mode

    def generate(self, question: str, hits: list[Hit], prompt: dict[str, str] | None = None):
        p = prompt or self.prompt
        context = format_context(hits)
        msgs = [{"role": "system", "content": p["system"]},
                {"role": "user", "content": p["user"].format(context=context, question=question)}]
        return self.llm.chat(msgs), context

    def answer(self, question: str) -> BotResponse:
        t0 = time.perf_counter()
        t_r = time.perf_counter()
        hits = self.kb.search(question, k=self.top_k, mode=self.mode)
        retrieval_s = time.perf_counter() - t_r
        res, context = self.generate(question, hits)
        return BotResponse(answer=res.text, model_name=self.name, latency_s=time.perf_counter() - t0,
                           prompt_tokens=res.prompt_tokens, completion_tokens=res.completion_tokens,
                           sources=_hits_to_sources(hits), context=context,
                           meta={"retrieval_s": retrieval_s, "prompt_version": self.prompt_version})


# =========================================================================== M3
class SemanticCache:
    """Returns a stored answer when a new question is semantically near-identical."""

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.vecs: list[np.ndarray] = []
        self.items: list[dict[str, Any]] = []

    def lookup(self, vec: list[float]) -> tuple[dict[str, Any] | None, float]:
        if not self.vecs:
            return None, 0.0
        q = np.asarray(vec, dtype=float)
        q /= (np.linalg.norm(q) or 1.0)
        sims = np.stack(self.vecs) @ q
        i = int(np.argmax(sims))
        best = float(sims[i])
        return (self.items[i], best) if best >= self.threshold else (None, best)

    def store(self, vec: list[float], item: dict[str, Any]) -> None:
        v = np.asarray(vec, dtype=float)
        self.vecs.append(v / (np.linalg.norm(v) or 1.0))
        self.items.append(item)

    def clear(self) -> None:
        self.vecs.clear()
        self.items.clear()


class LLMOpsBot:
    name = "Model 3: LLMOps (RAG + Ops)"
    key = "llmops"

    def __init__(self, llm: OllamaClient, kb: KnowledgeBase, prompts: dict[str, Any],
                 cfg: dict[str, Any], tracer: Tracer | None = None,
                 prompt_version: str | None = None):
        ops = cfg["llmops"]
        self.cfg = cfg
        self.llm, self.kb = llm, kb
        self.prompts = prompts
        self.prompt_version = prompt_version or ops["prompt_version"]
        self.rag = RAGBot(llm, kb, prompts, prompt_version=self.prompt_version)
        self.guardrails_on = bool(ops.get("enable_guardrails", True))
        self.cache_on = bool(ops.get("enable_cache", True))
        self.cache = SemanticCache(float(ops.get("cache_similarity", 0.92)))
        self.min_retrieval = float(ops.get("min_retrieval_score", 0.45))
        self.min_grounded = float(ops.get("min_groundedness", 0.5))
        self.tracer = tracer or Tracer(cfg)
        self._kb_blob = "\n".join(kb.all_text())

    def answer(self, question: str, session_id: str = "eval") -> BotResponse:
        trace_id = uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        spans: dict[str, float] = {}
        meta: dict[str, Any] = {"trace_id": trace_id, "prompt_version": self.prompt_version,
                                "blocked": False, "cache_hit": False, "escalated": False,
                                "pii_masked": [], "disclaimer": False}

        def finish(answer: str, res=None, hits=None, context="") -> BotResponse:
            resp = BotResponse(answer=answer, model_name=self.name,
                               latency_s=time.perf_counter() - t0,
                               prompt_tokens=getattr(res, "prompt_tokens", 0) if res else 0,
                               completion_tokens=getattr(res, "completion_tokens", 0) if res else 0,
                               sources=_hits_to_sources(hits or []), context=context,
                               meta={**meta, "spans": spans})
            self.tracer.log(trace_id=trace_id, session_id=session_id, question=meta.get("clean_q", question),
                            response=resp, model=self.llm.chat_model)
            return resp

        # 1. input guardrails ------------------------------------------------
        t = time.perf_counter()
        q = question
        if self.guardrails_on:
            verdict = check_input(question)
            spans["guardrails_in_s"] = time.perf_counter() - t
            meta["pii_masked"] = verdict.pii_found
            q = verdict.clean_text
            meta["clean_q"] = q
            if verdict.blocked:
                meta["blocked"] = True
                meta["block_reason"] = verdict.reason
                return finish(verdict.message)

        # 2. semantic cache --------------------------------------------------
        t = time.perf_counter()
        qvec = self.kb.embed_query(q)
        spans["embed_s"] = time.perf_counter() - t
        if self.cache_on:
            item, sim = self.cache.lookup(qvec)
            meta["cache_similarity"] = round(sim, 4)
            if item:
                meta["cache_hit"] = True
                return finish(item["answer"], hits=item["hits"], context=item["context"])

        # 3. retrieval + confidence gate ---------------------------------------
        t = time.perf_counter()
        hits = self.kb.search(q, query_vec=qvec)
        spans["retrieval_s"] = time.perf_counter() - t
        dense = [h.dense_score for h in hits if not math.isnan(h.dense_score)]
        confidence = max(dense) if dense else 0.0
        meta["retrieval_confidence"] = round(confidence, 4)
        if confidence < self.min_retrieval:
            meta["escalated"] = True
            return finish(ESCALATION_MSG, hits=hits)

        # 4. generation with the versioned prompt ------------------------------
        t = time.perf_counter()
        res, context = self.rag.generate(q, hits)
        spans["generation_s"] = time.perf_counter() - t
        answer = res.text

        # 5. output guardrails: PII scrub + groundedness check ------------------
        if self.guardrails_on:
            answer = scrub_output(answer, self._kb_blob)
        g = groundedness(answer, [h.chunk.text for h in hits])
        meta["groundedness"] = round(g, 3)
        if g < self.min_grounded and not _is_refusal(answer):
            answer += ("\n\n_Note: parts of this answer could not be verified against the knowledge base. "
                       "Please confirm with the university office._")
            meta["disclaimer"] = True

        if self.cache_on and not _is_refusal(answer):
            self.cache.store(qvec, {"answer": answer, "hits": hits, "context": context})
        return finish(answer, res=res, hits=hits, context=context)

    def feedback(self, trace_id: str, rating: int, comment: str = "") -> None:
        self.tracer.feedback(trace_id, rating, comment)


def _is_refusal(text: str) -> bool:
    from .metrics import is_refusal
    return is_refusal(text)
