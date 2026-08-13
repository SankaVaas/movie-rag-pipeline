"""
safety.py — Input validation and query safety guards.

Industrial systems should never trust raw user input.
This module is the single validation gate before any query reaches
the embedding model or the LLM.

Checks:
  1. Type and length bounds
  2. Empty / whitespace-only input
  3. Injection-pattern detection (prompt injection attempts)
  4. Character allowlist (printable Unicode, no control chars)
"""

import re
import logging
from typing import Tuple

from .config import PipelineConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# Patterns that suggest prompt injection or jailbreak attempts
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"(system\s*prompt|forget\s+everything)",
    r"you\s+are\s+now\s+a",
    r"act\s+as\s+(if\s+you\s+are\s+)?a",
    r"(reveal|print|show|output)\s+(your\s+)?(system\s+)?prompt",
    r"<\s*(script|iframe|object|embed)",    # XSS-style injections
    r"\{.*\}",                              # template / JSON injection
]

INJECTION_RE = re.compile(
    "|".join(INJECTION_PATTERNS),
    flags=re.IGNORECASE | re.DOTALL,
)


class QueryValidationError(ValueError):
    """Raised when a query fails safety validation."""


def validate_query(
    query: str,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> str:
    """
    Validate and sanitize a user query.

    Args:
        query:  raw user input
        config: pipeline config for length limits

    Returns:
        Stripped, safe query string.

    Raises:
        QueryValidationError: on any failed check (with a user-safe message).
    """
    # 1. Type check
    if not isinstance(query, str):
        raise QueryValidationError("Query must be a string.")

    # 2. Strip and check emptiness
    query = query.strip()
    if not query:
        raise QueryValidationError("Query cannot be empty.")

    # 3. Length bounds
    if len(query) > config.max_query_length:
        raise QueryValidationError(
            f"Query too long ({len(query)} chars). "
            f"Please keep it under {config.max_query_length} characters."
        )

    # 4. Control-character check
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", query):
        raise QueryValidationError("Query contains invalid control characters.")

    # 5. Injection detection
    if INJECTION_RE.search(query):
        logger.warning("Potential prompt injection detected in query: %r", query[:80])
        raise QueryValidationError(
            "Query contains patterns not allowed in this system. "
            "Please ask a straightforward question about movies."
        )

    return query


def truncate_context(text: str, max_chars: int) -> str:
    """
    Hard-truncate context sent to the LLM.
    Prevents token-limit overruns and reduces cost in high-volume settings.
    Truncates at a word boundary where possible.
    """
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.8:
        truncated = truncated[:last_space]
    return truncated + " [truncated]"
