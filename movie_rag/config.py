"""
config.py — Central configuration for the Movie RAG pipeline.

All tunable parameters live here. No magic numbers scattered across the codebase.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class PipelineConfig:
    # ── Data ──────────────────────────────────────────────────────────────
    dataset_rows: int   = 10000       # rows loaded from CSV
    csv_path: str       = "wiki_movie_plots_deduped.csv"
    index_path: str     = "movie_index"   # prefix; .faiss + .meta.json appended

    # ── Chunking ──────────────────────────────────────────────────────────
    chunk_size: int     = 300       # words per chunk
    chunk_overlap: int  = 50        # word overlap between consecutive chunks

    # ── Embedding ─────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"   # 384-dim, CPU-friendly
    embed_batch_size: int = 64

    # ── Retrieval ─────────────────────────────────────────────────────────
    top_k: int          = 4         # chunks retrieved per query
    min_score: float    = 0.20      # cosine similarity floor — reject irrelevant chunks

    # ── Generation ────────────────────────────────────────────────────────
    llm_model: str      = "gpt-4o-mini"
    max_tokens: int     = 1024
    temperature: float  = 0.2       # low temp for factual, grounded answers

    # ── Safety ────────────────────────────────────────────────────────────
    max_query_length: int  = 500    # chars — reject suspiciously long inputs
    max_context_chars: int = 6000   # total chars sent to LLM as context


# Singleton used throughout the app
DEFAULT_CONFIG = PipelineConfig()
