"""CHRIST University student helpdesk chatbot - LLM vs RAG vs LLMOps."""
from __future__ import annotations

from typing import Any

from .bots import BaselineLLMBot, LLMOpsBot, RAGBot
from .config import load_config, load_prompts
from .knowledge_base import KnowledgeBase
from .llm_client import OllamaClient
from .observability import Tracer


def build_all(cfg: dict[str, Any] | None = None, verbose: bool = True):
    """Create the shared LLM client, the knowledge base and the three bots."""
    cfg = cfg or load_config()
    prompts = load_prompts(cfg)
    llm = OllamaClient(cfg)
    kb = KnowledgeBase(cfg, llm).build(verbose=verbose)
    tracer = Tracer(cfg)
    bots = {
        "llm": BaselineLLMBot(llm, prompts),
        "rag": RAGBot(llm, kb, prompts, prompt_version="v1"),
        "llmops": LLMOpsBot(llm, kb, prompts, cfg, tracer=tracer),
    }
    return cfg, prompts, llm, kb, tracer, bots


__all__ = ["build_all", "load_config", "load_prompts", "OllamaClient", "KnowledgeBase",
           "BaselineLLMBot", "RAGBot", "LLMOpsBot", "Tracer"]
