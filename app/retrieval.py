"""Hybrid retrieval: vector search (Qdrant) + keyword search (BM25), merged with
Reciprocal Rank Fusion.

Why hybrid: vectors catch meaning ("send it back" ~ "return"), keywords catch exact
terms (SKUs, "PC-1042", "30 days"). Each one misses things the other finds.

Re-indexing is safe while the bot is answering: a new index is built in its own folder
(data/index/v<timestamp>), and only when it is complete is it swapped in and the old one
removed. The CURRENT file names the live version, so a restart picks up the latest one.
"""
from __future__ import annotations

import json
import logging
import shutil
import threading
import time
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


class _Snapshot:
    """One complete, immutable index version."""

    def __init__(self, path: Path, chunks: list[Chunk], qdrant: QdrantClient):
        self.path = path
        self.chunks = chunks
        self.by_id = {c.id: c for c in chunks}
        self.bm25 = BM25Okapi([tokenize(c.text) or ["_"] for c in chunks])
        self.qdrant = qdrant


class HybridIndex:
    def __init__(self, llm: LLM, index_dir: Path):
        self.llm = llm
        self.index_dir = index_dir
        self._snap: _Snapshot | None = None
        # Local Qdrant is not built for concurrent access, and a swap must not happen mid-search.
        self._lock = threading.RLock()

    @property
    def chunks(self) -> list[Chunk]:
        return self._snap.chunks if self._snap else []

    # ---------- build / load ----------
    def build(self, chunks: list[Chunk]) -> None:
        """Build a new version next to the live one, then swap it in."""
        if not chunks:
            raise ValueError("No chunks to index. Is the docs folder empty?")
        vectors = self.llm.embed([c.text for c in chunks], task="document")  # slow part, outside the lock

        self.index_dir.mkdir(parents=True, exist_ok=True)
        path = self.index_dir / f"v{time.time_ns()}"
        path.mkdir()
        qdrant = QdrantClient(path=str(path / "qdrant"))
        try:
            qdrant.create_collection(
                COLLECTION, vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE))
            qdrant.upsert(COLLECTION, points=[
                PointStruct(id=i, vector=v, payload={"chunk_id": c.id}) for i, (c, v) in enumerate(zip(chunks, vectors))
            ])
            (path / "chunks.json").write_text(json.dumps([c.to_dict() for c in chunks], indent=1))
        except Exception:
            qdrant.close()
            shutil.rmtree(path, ignore_errors=True)
            raise
        new = _Snapshot(path, chunks, qdrant)

        with self._lock:
            old, self._snap = self._snap, new
            (self.index_dir / "CURRENT").write_text(path.name)
        if old:
            old.qdrant.close()
        self._remove_old_versions(keep=path.name)
        log.info("Indexed %d chunks into %s", len(chunks), path.name)

    def load(self) -> bool:
        current = self.index_dir / "CURRENT"
        if not current.exists():
            return False
        path = self.index_dir / current.read_text().strip()
        if not (path / "chunks.json").exists():
            return False
        chunks = [Chunk(**d) for d in json.loads((path / "chunks.json").read_text())]
        with self._lock:
            self._snap = _Snapshot(path, chunks, QdrantClient(path=str(path / "qdrant")))
        self._remove_old_versions(keep=path.name)
        return True

    def _remove_old_versions(self, keep: str) -> None:
        for p in self.index_dir.iterdir():
            if p.is_dir() and p.name != keep:
                shutil.rmtree(p, ignore_errors=True)

    def close(self) -> None:
        with self._lock:
            if self._snap:
                self._snap.qdrant.close()
                self._snap = None

    # ---------- search ----------
    def search(self, query: str, k: int = 5, mode: str = "hybrid", pool: int = 20) -> list[Hit]:
        qv = self.llm.embed([query], task="query")[0]  # network call, outside the lock
        with self._lock:
            snap = self._snap
            if snap is None:
                raise RuntimeError("Search index is not built yet.")
            vec = [(snap.by_id[p.payload["chunk_id"]], p.score)
                   for p in snap.qdrant.query_points(COLLECTION, query=qv, limit=pool).points]
            if mode == "vector":
                return [Hit(c, s, similarity=s) for c, s in vec[:k]]
            kw = self._keyword_search(snap, query, pool)

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

    @staticmethod
    def _keyword_search(snap: _Snapshot, query: str, k: int) -> list[tuple[Chunk, float]]:
        terms = tokenize(query)
        if not terms:
            return []
        scores = snap.bm25.get_scores(terms)
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(snap.chunks[i], float(scores[i])) for i in ranked if scores[i] > 0]
