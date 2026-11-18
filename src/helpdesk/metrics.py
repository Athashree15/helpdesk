"""Evaluation metrics (all computed locally and deterministically, plus an optional LLM judge)."""
from __future__ import annotations

import json
import re
from typing import Any

from .knowledge_base import tokenize

REFUSAL_PATTERNS = [
    r"don'?t have (that|this|the|any|enough)? ?information", r"do not have (that|this|the|any)? ?information",
    r"not (able|allowed) to", r"can(no|')?t (help|assist|answer|provide|share)", r"i'?m (sorry|unable)",
    r"unable to", r"outside (the |my )?(scope|knowledge)", r"not (in|part of) (the|my) (knowledge|context)",
    r"(please )?contact the", r"only help with", r"don'?t want to guess", r"not available",
    r"no information", r"cannot (provide|share|give|answer)", r"i don'?t know", r"not sure",
    r"couldn'?t find", r"real-?time", r"don'?t have access", r"not something i can",
]
NUM_RE = re.compile(r"\d[\d,\-\.]*\d|\d")


def is_refusal(text: str) -> bool:
    low = text.lower()
    return any(re.search(p, low) for p in REFUSAL_PATTERNS)


def keyword_coverage(answer: str, keyword_groups: list[str]) -> float:
    """Fraction of required fact-groups present in the answer (alternatives split by '|')."""
    if not keyword_groups:
        return float("nan")
    low = answer.lower()
    hit = sum(1 for g in keyword_groups if any(alt.strip().lower() in low for alt in g.split("|")))
    return hit / len(keyword_groups)


def _sentences(text: str) -> list[str]:
    text = re.sub(r"\[\d+\]", "", text)
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(tokenize(p)) >= 3]


def groundedness(answer: str, passages: list[str], threshold: float = 0.5) -> float:
    """Share of answer sentences whose content words are mostly found in one passage.

    A cheap, deterministic proxy for faithfulness: 1.0 = every claim is traceable
    to the evidence, 0.0 = nothing is.
    """
    sents = _sentences(answer)
    if not sents:
        return 1.0
    pass_tokens = [set(tokenize(p)) for p in passages] or [set()]
    supported = 0
    for s in sents:
        toks = set(tokenize(s))
        best = max(len(toks & pt) / max(len(toks), 1) for pt in pass_tokens)
        supported += best >= threshold
    return supported / len(sents)


def unsupported_numbers(answer: str, kb_text: str) -> int:
    """Numbers stated in the answer that appear nowhere in the knowledge base (hallucination signal)."""
    kb_nums = {n.replace(",", "") for n in NUM_RE.findall(kb_text)}
    ans = re.sub(r"\[\d+\]", "", answer)
    return sum(1 for n in NUM_RE.findall(ans) if n.replace(",", "") not in kb_nums)


def retrieval_scores(retrieved_sources: list[str], gold: str) -> dict[str, float]:
    rank = next((i + 1 for i, s in enumerate(retrieved_sources) if s == gold), None)
    return {"hit@1": float(rank == 1), "hit@k": float(rank is not None),
            "mrr": 1.0 / rank if rank else 0.0}


JUDGE_PROMPT = """You are grading a university helpdesk chatbot.
Question: {question}
Reference answer: {reference}
Chatbot answer: {answer}

Score the chatbot answer for factual correctness against the reference on a 1-5 scale:
5 = fully correct and complete, 4 = correct but missing a minor detail, 3 = partially correct,
2 = mostly wrong or vague, 1 = wrong, hallucinated or irrelevant.
Reply ONLY with JSON like {{"score": 4, "reason": "short reason"}}."""


def llm_judge(llm, question: str, reference: str, answer: str) -> tuple[int | None, str]:
    """LLM-as-a-judge using the same local model (score 1-5)."""
    text = ""
    try:
        res = llm.chat([{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, reference=reference, answer=answer[:1500])}],
            temperature=0.0, json_mode=True)
        text = res.text
        data: Any = json.loads(text)
        return max(1, min(5, int(data.get("score")))), str(data.get("reason", ""))[:200]
    except Exception as exc:  # noqa: BLE001
        m = re.search(r"[1-5]", text)
        return (int(m.group(0)) if m else None), f"judge-parse-fallback: {exc}"[:200]
