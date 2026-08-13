"""
pipeline.py — Top-level RAG pipeline orchestrator.

Ties together: loader -> vector_store -> safety -> generator

This is the single public API of the package. External code (main.py, tests)
should only import Pipeline — never the individual modules directly.
"""

import logging
import os
from typing import Dict, Optional

from .config import PipelineConfig, DEFAULT_CONFIG
from .loader import load_movies, build_chunks
from .vector_store import VectorStore
from .generator import AnswerGenerator, RAGResponse
from .safety import validate_query, QueryValidationError

logger = logging.getLogger(__name__)


class Pipeline:
    """
    End-to-end Movie RAG pipeline.

    Lifecycle:
        pipe = Pipeline()
        pipe.build()          #loads data, embeds, saves index
        result = pipe.query("Which film has a robot uprising?")
        print(result.to_json())

    Subsequent runs:
        pipe = Pipeline()
        pipe.load()           # instant — reads saved FAISS index
        result = pipe.query(...)
    """

    def __init__(self, config: PipelineConfig = DEFAULT_CONFIG):
        self.config    = config
        self.store     = VectorStore(config)
        self.generator = AnswerGenerator(config)

    # ── Index management ──────────────────────────────────────────────────

    def build(self, force: bool = False) -> None:
        """
        Load data -> chunk -> embed -> save FAISS index.

        Args:
            force: rebuild even if a saved index already exists.
        """
        index_exists = os.path.exists(self.config.index_path + ".faiss")

        if index_exists and not force:
            logger.info(
                "Saved index found at '%s'. Loading instead of rebuilding. "
                "Pass force=True to rebuild.",
                self.config.index_path,
            )
            self.load()
            return

        logger.info("Building pipeline index...")
        df      = load_movies(self.config)
        records = build_chunks(df, self.config)
        self.store.build(records)
        self.store.save(self.config.index_path)
        logger.info("Pipeline ready.")

    def load(self) -> None:
        """Load a pre-built FAISS index from disk."""
        self.store.load(self.config.index_path)
        logger.info("Pipeline ready (loaded from disk).")

    @property
    def is_ready(self) -> bool:
        return self.store.is_ready

    # ── Query ─────────────────────────────────────────────────────────────

    def query(self, raw_query: str) -> RAGResponse:
        """
        Run the full RAG pipeline for a single query.

        Args:
            raw_query: raw user input (will be validated before processing)

        Returns:
            RAGResponse with answer, contexts, reasoning, and sources.

        Raises:
            QueryValidationError: if the query fails safety checks.
            RuntimeError: if the pipeline index isn't built yet.
        """
        if not self.is_ready:
            raise RuntimeError(
                "Pipeline not ready. Call build() or load() before querying."
            )

        # Safety gate — validates and sanitizes the query
        query = validate_query(raw_query, self.config)

        logger.info("Query: %r", query)

        chunks = self.store.retrieve(query)
        logger.info(
            "Top results: %s",
            [(c["title"], c["score"]) for c in chunks],
        )

        response = self.generator.generate(query, chunks)
        logger.info("Answer generated (%d chars).", len(response.answer))
        return response
