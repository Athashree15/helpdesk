"""Knowledge base: loading, chunking, indexing (ChromaDB + BM25) and hybrid retrieval."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import chromadb
from rank_bm25 import BM25Okapi

from .llm_client import OllamaClient

STOPWORDS = set("""a an the and or but if of to in on at for from by with without is are was were be been
being do does did have has had i me my we our you your he she it its they them their this that these
those what which who whom when where why how can could should would will shall may might must not no
yes so than then there here about into over under again further once all any both each few more most
other some such only own same too very s t just don am as up out off""".split())

TOKEN_RE = re.compile(r"[a-z0-9@.\-]+")


def tokenize(text: str) -> list[str]:
    toks = [t.strip(".-") for t in TOKEN_RE.findall(text.lower())]
    return [t for t in toks if t and t not in STOPWORDS]


@dataclass
class Document:
    source: str
    title: str
    verified: str
    url: str
    body: str


@dataclass
class Chunk:
    id: str
    source: str
    title: str
    section: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Hit:
    chunk: Chunk
    score: float          # fused score used for ranking
    dense_score: float    # cosine similarity (0..1) - used as a confidence signal
    bm25_score: float


# ---------------------------------------------------------------------- loading
def load_documents(kb_dir: str) -> list[Document]:
    docs = []
    for path in sorted(Path(kb_dir).glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        meta: dict[str, str] = {}
        body = raw
        if raw.startswith("---"):
            _, front, body = raw.split("---", 2)
            for line in front.strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
        docs.append(Document(source=path.name, title=meta.get("title", path.stem),
                             verified=meta.get("verified", "unknown"),
                             url=meta.get("source", ""), body=body.strip()))
    return docs


# --------------------------------------------------------------------- chunking
def chunk_document(doc: Document, chunk_size: int, overlap: int) -> list[Chunk]:
    """Heading-aware chunking.

    The document is split on '## ' headings; long sections are split further on
    sentence boundaries with a character overlap. Every chunk is prefixed with
    the document title and section heading (a "contextual chunk header"), which
    makes short chunks far easier to retrieve.
    """
    sections = re.split(r"^##\s+", doc.body, flags=re.M)
    chunks: list[Chunk] = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        heading, _, content = sec.partition("\n")
        content = content.strip() or heading
        sentences = re.split(r"(?<=[.!?])\s+", content)
        buf = ""
        pieces: list[str] = []
        for s in sentences:
            if buf and len(buf) + len(s) + 1 > chunk_size:
                pieces.append(buf.strip())
                buf = buf[-overlap:] if overlap else ""
                # start the overlap on a word boundary
                buf = buf[buf.find(" ") + 1:] if " " in buf else buf
            buf += " " + s
        if buf.strip():
            pieces.append(buf.strip())
        for i, p in enumerate(pieces):
            text = f"{doc.title} > {heading}\n{p}"
            cid = hashlib.md5(f"{doc.source}|{heading}|{i}|{chunk_size}".encode()).hexdigest()[:12]
            chunks.append(Chunk(id=cid, source=doc.source, title=doc.title, section=heading,
                                text=text, meta={"verified": doc.verified, "url": doc.url}))
    return chunks


def build_chunks(kb_dir: str, chunk_size: int, overlap: int) -> list[Chunk]:
    out: list[Chunk] = []
    for d in load_documents(kb_dir):
        out.extend(chunk_document(d, chunk_size, overlap))
    return out


# -------------------------------------------------------------------- the index
class KnowledgeBase:
    """Hybrid retriever: dense vectors in ChromaDB + sparse BM25, fused with RRF."""

    def __init__(self, cfg: dict[str, Any], llm: OllamaClient,
                 chunk_size: int | None = None, chunk_overlap: int | None = None):
        r = cfg["retrieval"]
        self.cfg = cfg
        self.llm = llm
        self.kb_dir = cfg["paths"]["kb_dir"]
        self.chunk_size = int(chunk_size or r["chunk_size"])
        self.chunk_overlap = int(r["chunk_overlap"] if chunk_overlap is None else chunk_overlap)
        self.default_mode = r.get("mode", "hybrid")
        self.default_k = int(r.get("top_k", 4))
        store = Path(cfg["paths"]["storage_dir"]) / "chroma"
        store.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(store))
        safe_model = re.sub(r"[^a-zA-Z0-9]", "_", llm.embed_model)
        self.collection_name = f"kb_{safe_model}_{self.chunk_size}"[:60]
        self.chunks: list[Chunk] = []
        self.bm25: BM25Okapi | None = None
        self._by_id: dict[str, Chunk] = {}

    # -- build / load -----------------------------------------------------------
    def _fingerprint(self) -> str:
        h = hashlib.md5()
        for p in sorted(Path(self.kb_dir).glob("*.md")):
            h.update(p.name.encode())
            h.update(p.read_bytes())
        h.update(f"{self.chunk_size}|{self.chunk_overlap}|{self.llm.embed_model}".encode())
        return h.hexdigest()

    def build(self, force: bool = False, verbose: bool = True) -> "KnowledgeBase":
        fp = self._fingerprint()
        manifest = Path(self.cfg["paths"]["storage_dir"]) / f"{self.collection_name}.json"
        self.chunks = build_chunks(self.kb_dir, self.chunk_size, self.chunk_overlap)
        self._by_id = {c.id: c for c in self.chunks}
        self.bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])

        if not force and manifest.exists():
            m = json.loads(manifest.read_text())
            if m.get("fingerprint") == fp:
                try:
                    col = self.client.get_collection(self.collection_name)
                    if col.count() == len(self.chunks):
                        if verbose:
                            print(f"[kb] reusing index '{self.collection_name}' ({col.count()} chunks)")
                        return self
                except Exception:  # noqa: BLE001
                    pass

        if verbose:
            print(f"[kb] embedding {len(self.chunks)} chunks with '{self.llm.embed_model}' ...")
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:  # noqa: BLE001
            pass
        col = self.client.create_collection(self.collection_name, metadata={"hnsw:space": "cosine"},
                                            embedding_function=None)
        vectors = self.llm.embed([self._embed_text(c.text, doc=True) for c in self.chunks])
        col.add(ids=[c.id for c in self.chunks], embeddings=vectors,
                documents=[c.text for c in self.chunks],
                metadatas=[{"source": c.source, "title": c.title, "section": c.section,
                            "verified": c.meta["verified"]} for c in self.chunks])
        manifest.write_text(json.dumps({"fingerprint": fp, "chunks": [asdict(c) for c in self.chunks]},
                                       indent=1))
        if verbose:
            print(f"[kb] index '{self.collection_name}' ready")
        return self

    def _embed_text(self, text: str, doc: bool) -> str:
        # nomic-embed-text is trained with task prefixes; harmless for other models
        if "nomic" in self.llm.embed_model:
            return ("search_document: " if doc else "search_query: ") + text
        return text

    # -- retrieval --------------------------------------------------------------
    def dense_search(self, query: str, k: int, query_vec: list[float] | None = None) -> list[tuple[str, float]]:
        col = self.client.get_collection(self.collection_name)
        qv = query_vec or self.llm.embed([self._embed_text(query, doc=False)])[0]
        res = col.query(query_embeddings=[qv], n_results=min(k, len(self.chunks)))
        return [(cid, 1.0 - float(d)) for cid, d in zip(res["ids"][0], res["distances"][0])]

    def bm25_search(self, query: str, k: int) -> list[tuple[str, float]]:
        assert self.bm25 is not None
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.chunks[i].id, float(scores[i])) for i in order]

    def search(self, query: str, k: int | None = None, mode: str | None = None,
               query_vec: list[float] | None = None) -> list[Hit]:
        k = k or self.default_k
        mode = mode or self.default_mode
        pool = max(k * 3, 10)
        dense = self.dense_search(query, pool, query_vec) if mode in ("dense", "hybrid") else []
        sparse = self.bm25_search(query, pool) if mode in ("bm25", "hybrid") else []
        d_map, s_map = dict(dense), dict(sparse)

        if mode == "dense":
            ranked = [(cid, s) for cid, s in dense]
        elif mode == "bm25":
            ranked = [(cid, s) for cid, s in sparse]
        else:  # Reciprocal Rank Fusion (Cormack et al., 2009)
            fused: dict[str, float] = {}
            for lst in (dense, sparse):
                for rank, (cid, _) in enumerate(lst):
                    fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + rank + 1)
            ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)

        hits = []
        for cid, score in ranked[:k]:
            hits.append(Hit(chunk=self._by_id[cid], score=score,
                            dense_score=d_map.get(cid, math.nan), bm25_score=s_map.get(cid, 0.0)))
        return hits

    def embed_query(self, query: str) -> list[float]:
        return self.llm.embed([self._embed_text(query, doc=False)])[0]

    def all_text(self) -> list[str]:
        return [c.text for c in self.chunks]


def format_context(hits: list[Hit]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        parts.append(f"[{i}] (source: {h.chunk.source})\n{h.chunk.text}")
    return "\n\n".join(parts)
