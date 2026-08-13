"""
loader.py — Data loading, validation, and chunking.

  - Load Wikipedia Movie Plots CSV (or auto-download sample)
  - Validate and sanitize inputs
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
        .query("Plot.str.len() >= 100")   # skip trivially short plots
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