"""
loader.py — Data loading, validation, and chunking.

  - Load Wikipedia Movie Plots CSV (or auto-download sample)
  - Validate and sanitize inputs
  - Chunk long plot texts with configurable overlap
"""

import logging
import os
import re
from typing import List, Dict

import pandas as pd

from .config import PipelineConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load & validate
# ---------------------------------------------------------------------------

def load_movies(config: PipelineConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """
    Load movie data from CSV. Auto-downloads a sample if the file is absent.

    Returns a DataFrame with clean 'Title' and 'Plot' columns.
    Raises FileNotFoundError only if auto-download also fails.
    """
    path = config.csv_path

    if not os.path.exists(path):
        logger.info("CSV not found at '%s'. Attempting auto-download...", path)
        download_sample(path, config.dataset_rows)

    try:
        df = pd.read_csv(path, usecols=["Title", "Plot"])
    except ValueError:
        raise ValueError(
            f"CSV at '{path}' must contain 'Title' and 'Plot' columns. "
            "Download from: https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots"
        )

    df = (
        df.dropna(subset=["Title", "Plot"])
        .assign(
            Title=lambda d: d["Title"].str.strip(),
            Plot=lambda d:  d["Plot"].str.strip(),
        )
        .query("Plot.str.len() >= 500")   # skip trivially short plots
        .iloc[:config.dataset_rows]
        .reset_index(drop=True)
    )

    logger.info("Loaded %d movies from '%s'.", len(df), path)
    return df


def download_sample(path: str, n: int) -> None:
    """Fallback: download n rows from HuggingFace."""
    try:
        from datasets import load_dataset as hf_load
        ds = hf_load(
            "vishnupriyavr/wiki-movie-plots-with-summaries",
            split=f"train[:{n}]",
        )
        pd.DataFrame({"Title": ds["Title"], "Plot": ds["Plot"]}).to_csv(path, index=False)
        logger.info("Auto-downloaded %d rows → '%s'.", n, path)
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not download dataset: {exc}. "
            "Please place 'wiki_movie_plots_deduped.csv' in the project root."
        ) from exc


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> List[str]:
    """
    Split text into overlapping word-count windows.

    Args:
        text:       input string
        chunk_size: target words per chunk
        overlap:    words shared between adjacent chunks (preserves cross-boundary context)

    Returns:
        List of non-empty chunk strings. Always at least one chunk.
    """
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")

    words  = text.split()
    if not words:
        return []

    chunks = []
    start  = 0
    stride = chunk_size - overlap

    while start < len(words):
        chunk = " ".join(words[start : start + chunk_size])
        chunks.append(chunk)
        start += stride

    return chunks


def build_chunks(
    df: pd.DataFrame,
    config: PipelineConfig = DEFAULT_CONFIG,
) -> List[Dict]:
    """
    Chunk every movie plot and return a flat list of records.

    Each record: { title, chunk_id, text }
    chunk_id is zero-indexed per movie — useful for debugging retrieval.
    """
    records: List[Dict] = []

    for _, row in df.iterrows():
        chunks = chunk_text(row["Plot"], config.chunk_size, config.chunk_overlap)
        for i, chunk in enumerate(chunks):
            records.append({
                "title":    row["Title"],
                "chunk_id": i,
                "text":     chunk,
            })

    logger.info(
        "Built %d chunks from %d movies (avg %.1f chunks/movie).",
        len(records),
        len(df),
        len(records) / max(len(df), 1),
    )
    return records
