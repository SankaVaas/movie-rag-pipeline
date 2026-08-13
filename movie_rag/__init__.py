"""movie_rag — Minimal production-grade Movie Plot RAG system."""

from .pipeline import Pipeline
from .config import PipelineConfig, DEFAULT_CONFIG
from .safety import QueryValidationError

__all__ = ["Pipeline", "PipelineConfig", "DEFAULT_CONFIG", "QueryValidationError"]
