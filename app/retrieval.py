"""Hybrid retrieval: vector search (Qdrant) + keyword search (BM25), merged with
Reciprocal Rank Fusion.

Why hybrid: vectors catch meaning ("send it back" ~ "return"), keywords catch exact
terms (SKUs, "PC-1042", "30 days"). Each one misses things the other finds.
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from .ingest import Chunk
from .llm import LLM, tokenize

log = logging.getLogger("supportbot.retrieval")
COLLECTION = "docs"


@dataclass
class Hit:
    chunk: Chunk
    score: float            # fused score used for ranking
    similarity: float = 0.0 # cosine similarity from the vector side (0 if not found there)
    keyword: float = 0.0    # BM25 score (0 if not found there)


class HybridIndex:
    def __init__(self, llm: LLM, index_dir: Path):
        self.llm = llm
        self.index_dir = index_dir
        self.chunks: list[Chunk] = []
        self.by_id: dict[str, Chunk] = {}
        self.bm25: BM25Okapi | None = None
        self.qdrant: QdrantClient | None = None

    # ---------- build / load ----------
    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("No chunks to index. Is the docs folder empty?")
        if self.qdrant is not None:
            self.qdrant.close()
            self.qdrant = None
        if self.index_dir.exists():
            shutil.rmtree(self.index_dir)
        self.index_dir.mkdir(parents=True)

        vectors = self.llm.embed([c.text for c in chunks], task="document")
        self.qdrant = QdrantClient(path=str(self.index_dir / "qdrant"))
        self.qdrant.create_collection(
            COLLECTION, vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE))
        self.qdrant.upsert(COLLECTION, points=[
            PointStruct(id=i, vector=v, payload={"chunk_id": c.id}) for i, (c, v) in enumerate(zip(chunks, vectors))
        ])
        (self.index_dir / "chunks.json").write_text(json.dumps([c.to_dict() for c in chunks], indent=1))
        self._set_chunks(chunks)
        log.info("Indexed %d chunks", len(chunks))

    def load(self) -> bool:
        path = self.index_dir / "chunks.json"
        if not path.exists():
            return False
        self._set_chunks([Chunk(**d) for d in json.loads(path.read_text())])
        self.qdrant = QdrantClient(path=str(self.index_dir / "qdrant"))
        return True

    def _set_chunks(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.by_id = {c.id: c for c in chunks}
        self.bm25 = BM25Okapi([tokenize(c.text) or ["_"] for c in chunks])

    def close(self) -> None:
        if self.qdrant is not None:
            self.qdrant.close()
            self.qdrant = None

    # ---------- search ----------
    def vector_search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        qv = self.llm.embed([query], task="query")[0]
        res = self.qdrant.query_points(COLLECTION, query=qv, limit=k).points
        return [(self.by_id[p.payload["chunk_id"]], p.score) for p in res]

    def keyword_search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        terms = tokenize(query)
        if not terms:
            return []
        scores = self.bm25.get_scores(terms)
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.chunks[i], float(scores[i])) for i in ranked if scores[i] > 0]

    def search(self, query: str, k: int = 5, mode: str = "hybrid", pool: int = 20) -> list[Hit]:
        vec = self.vector_search(query, pool)
        if mode == "vector":
            return [Hit(c, s, similarity=s) for c, s in vec[:k]]

        kw = self.keyword_search(query, pool)
        hits: dict[str, Hit] = {}
        rrf_k = 60  # standard RRF constant: dampens the advantage of rank 1 vs rank 2
        for rank, (c, s) in enumerate(vec):
            h = hits.setdefault(c.id, Hit(c, 0.0))
            h.score += 1 / (rrf_k + rank + 1)
            h.similarity = s
        for rank, (c, s) in enumerate(kw):
            h = hits.setdefault(c.id, Hit(c, 0.0))
            h.score += 1 / (rrf_k + rank + 1)
            h.keyword = s
        return sorted(hits.values(), key=lambda h: -h.score)[:k]
