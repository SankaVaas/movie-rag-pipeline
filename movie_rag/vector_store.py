"""
vector_store.py — Embedding, FAISS indexing, and semantic retrieval.

Main Design:
  - Embeddings: sentence-transformers all-MiniLM-L6-v2 (384-dim, CPU-first)
  - Index: FAISS IndexFlatIP on L2-normalised vectors = exact cosine similarity
  - Persistence: index saved as .faiss + .meta.json for instant cold-start
  - Safety: score floor filters low-relevance chunks before they reach the LLM
"""

import json
import logging
import os
from typing import List, Dict, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from .config import PipelineConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

FAISS_SUFFIX = ".faiss"
META_SUFFIX  = ".meta.json"


class VectorStore:
    """
    Wraps a FAISS index with metadata and provides embed + retrieve.

    Thread-safety note: FAISS reads are thread-safe; writes are not.
    For concurrent writes, add an external lock around build().
    """

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG):
        self.config  = config
        self._index:   Optional[faiss.Index] = None
        self._records: List[Dict]            = []
        self._model:   Optional[SentenceTransformer] = None

    # ── Lazy model load ───────────────────────────────────────────────────

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model '%s'...", self.config.embedding_model)
            self._model = self.load_model_fast(self.config.embedding_model)
        return self._model

    @staticmethod
    def load_model_fast(model_name: str) -> SentenceTransformer:
        """
        Load a cached SentenceTransformer without a Hugging Face Hub round-trip.

        force offline mode and only fall back to an online load if the model isn't cached yet.
        """
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        try:
            return SentenceTransformer(model_name)
        except Exception:
            logger.info("Model not cached locally yet — downloading '%s'...", model_name)
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            return SentenceTransformer(model_name)

    # ── Build ─────────────────────────────────────────────────────────────

    def build(self, records: List[Dict]) -> None:
        """
        Embed all records and build the FAISS index in memory.

        Uses L2-normalised vectors with IndexFlatIP so that inner product
        equals cosine similarity — exact search, no approximation.
        """
        if not records:
            raise ValueError("Cannot build index from empty record list.")

        texts      = [r["text"] for r in records]
        embeddings = self._embed(texts)

        dim        = embeddings.shape[1]
        index      = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        self._index   = index
        self._records = records

        logger.info(
            "FAISS index built: %d vectors, dim=%d.",
            index.ntotal, dim,
        )

    def _embed(self, texts: List[str]) -> np.ndarray:
        """Encode texts and L2-normalise for cosine similarity."""
        vecs = self.model.encode(
            texts,
            batch_size=self.config.embed_batch_size,
            show_progress_bar=len(texts) > 1,   # skip bar overhead for single-query encodes
            convert_to_numpy=True,
        ).astype(np.float32)
        faiss.normalize_L2(vecs)
        return vecs

    # ── Persist ───────────────────────────────────────────────────────────

    def save(self, path_prefix: str) -> None:
        """Save index and metadata to disk for instant cold-start."""
        if self._index is None:
            raise RuntimeError("No index to save. Call build() first.")
        faiss.write_index(self._index, path_prefix + FAISS_SUFFIX)
        with open(path_prefix +    META_SUFFIX, "w", encoding="utf-8") as f:
            json.dump(self._records, f, ensure_ascii=False)
        logger.info("Index saved → '%s'.", path_prefix)

    def load(self, path_prefix: str) -> None:
        """Load a previously saved index from disk."""
        faiss_path = path_prefix + FAISS_SUFFIX
        meta_path  = path_prefix + META_SUFFIX
        if not os.path.exists(faiss_path):
            raise FileNotFoundError(f"No FAISS index at '{faiss_path}'.")
        self._index = faiss.read_index(faiss_path)
        with open(meta_path, encoding="utf-8") as f:
            self._records = json.load(f)
        logger.info(
            "Index loaded ← '%s' (%d vectors).",
            path_prefix, self._index.ntotal,
        )

    @property
    def is_ready(self) -> bool:
        return self._index is not None and len(self._records) > 0

    # ── Retrieve ──────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Embed query and return the top_k most similar chunks.

        Applies a minimum cosine similarity floor (config.min_score) to
        prevent low-relevance context from reaching the LLM — a key safety
        and quality measure in production RAG systems.

        Returns:
            List of { title, chunk_id, text, score } dicts, descending by score.
            Empty list if nothing clears the similarity floor.
        """
        if not self.is_ready:
            raise RuntimeError("VectorStore not ready. Call build() or load() first.")

        k     = top_k or self.config.top_k
        q_vec = self._embed([query])

        scores, indices = self._index.search(q_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            if float(score) < self.config.min_score:
                logger.debug(
                    "Chunk %d score %.3f below floor %.2f — skipped.",
                    idx, score, self.config.min_score,
                )
                continue
            rec = self._records[idx]
            results.append({
                "title":    rec["title"],
                "chunk_id": rec["chunk_id"],
                "text":     rec["text"],
                "score":    round(float(score), 4),
            })

        logger.info(
            "Retrieved %d/%d chunks above score floor %.2f.",
            len(results), k, self.config.min_score,
        )
        return results