"""
main.py — CLI entry point for the Movie Plot RAG pipeline.

Usage:
    # First run: build the index
    python main.py --build

    # Single query
    python main.py --query "Which movie has a robot uprising?"

    # Interactive REPL
    python main.py

    # Rebuild index with custom CSV
    python main.py --build --csv my_data.csv --rows 10000
"""

import argparse
import json
import logging
import os
import sys

from movie_rag import Pipeline, PipelineConfig, QueryValidationError


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        stream=sys.stderr,   # JSON answers go to stdout
    )
    # Suppress noisy third-party loggers unless verbose
    if not verbose:
        for noisy in ("sentence_transformers", "transformers", "huggingface_hub", "faiss", "httpx", "httpcore"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Movie Plot RAG — answer questions about movie plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python main.py --build
              python main.py --query "Which film features HAL 9000?"
              python main.py --query "Find a movie about a deserted island" --top-k 6
              python main.py                          # interactive REPL
              python main.py --build --force          # force index rebuild
        """),
    )

    p.add_argument("--build",   action="store_true",
                   help="Build (or rebuild) the FAISS index before querying")
    p.add_argument("--force",   action="store_true",
                   help="Force rebuild even if a saved index exists")
    p.add_argument("--query",   type=str, default=None,
                   help="Single query (non-interactive mode)")
    p.add_argument("--csv",     type=str, default=None,
                   help="Path to Wikipedia Movie Plots CSV")
    p.add_argument("--rows",    type=int, default=None,
                   help="Number of movie rows to load (default: 500)")
    p.add_argument("--top-k",   type=int, default=None, dest="top_k",
                   help="Number of chunks to retrieve per query (default: 4)")
    p.add_argument("--verbose", action="store_true",
                   help="Show DEBUG logs and full pipeline internals")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

import textwrap   # needed for epilog, imported here to keep top-level clean


def main() -> None:
    args = _parse_args()
    _setup_logging(args.verbose)

    # Build config, overriding defaults with any CLI arguments supplied
    config_kwargs = {}
    if args.csv:
        config_kwargs["csv_path"]     = args.csv
    if args.rows:
        config_kwargs["dataset_rows"] = args.rows
    if args.top_k:
        config_kwargs["top_k"]        = args.top_k

    config   = PipelineConfig(**config_kwargs) if config_kwargs else PipelineConfig()
    pipeline = Pipeline(config)

    # ---- Build / Load --------------------------------
    try:
        if args.build:
            pipeline.build(force=args.force)
        else:
            # Auto-load if index exists; auto-build if it doesn't
            if os.path.exists(config.index_path + ".faiss"):
                pipeline.load()
            else:
                print(
                    "No saved index found. Building now "
                    "(this takes ~30s on first run)...",
                    file=sys.stderr,
                )
                pipeline.build()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # ---- Single-query mode --------------------------------
    if args.query:
        _run_query(pipeline, args.query)
        return

    # --- Interactive REPL --------------------------------
    _repl(pipeline)


def _run_query(pipeline: Pipeline, raw_query: str) -> None:
    """Run a single query and print JSON to stdout."""
    try:
        result = pipeline.query(raw_query)
        print(result.to_json())
    except QueryValidationError as exc:
        print(json.dumps({"error": "invalid_query", "detail": str(exc)}, indent=2))
        sys.exit(1)
    except Exception as exc:
        logging.getLogger(__name__).exception("Unexpected error during query.")
        print(json.dumps({"error": "internal_error", "detail": str(exc)}, indent=2))
        sys.exit(1)


def _repl(pipeline: Pipeline) -> None:
    """Interactive query loop."""
    print("\n🎬  Movie RAG — Interactive Mode")
    print("   Ask any question about movie plots.")
    print("   Type 'quit' or press Ctrl-C to exit.\n")

    while True:
        try:
            raw = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not raw:
            continue
        if raw.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        try:
            result = pipeline.query(raw)
            print("\n" + result.to_json() + "\n")
        except QueryValidationError as exc:
            print(f"\n[Rejected] {exc}\n")
        except Exception as exc:
            logging.getLogger(__name__).exception("Unexpected error.")
            print(f"\n[Error] {exc}\n")


if __name__ == "__main__":
    main()
