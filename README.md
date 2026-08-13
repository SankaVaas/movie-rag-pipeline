# Movie Plot RAG Pipeline

A minimal, production-grade **Retrieval-Augmented Generation** system that answers natural language questions about movie plots using the Wikipedia Movie Plots dataset.

---

## Architecture

```
User Query
    │
    ▼
Safety & Validation (safety.py)
    │  ✗ reject injections, malformed input
    ▼
Embedding (sentence-transformers: all-MiniLM-L6-v2)
    │
    ▼
FAISS Retrieval — cosine similarity, top-k chunks
    │  ✗ filter below similarity floor (min_score=0.20)
    ▼
Context Assembly + LLM (Claude claude-sonnet-4-6)
    │
    ▼
Structured JSON Output
{ answer, contexts, reasoning, sources }
```

---

## Project Structure

```
movie-rag-pipeline/
├── main.py                    ← CLI entry point (build / query / interactive)
├── requirements.txt
├── README.md
├── movie_rag/
│   ├── __init__.py            ← public API: Pipeline, PipelineConfig
│   ├── config.py              ← all tunable parameters in one place
│   ├── loader.py              ← CSV loading, validation, chunking
│   ├── vector_store.py        ← sentence-transformers + FAISS
│   ├── generator.py           ← Claude LLM call + JSON parsing
│   ├── pipeline.py            ← orchestrator (public API)
│   └── safety.py              ← query validation + injection detection
└── tests/
    └── test_pipeline.py       ← unit tests for chunking, safety, loader
```

---

## Quickstart

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Set your API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Get the dataset

Download `wiki_movie_plots_deduped.csv` from [Kaggle](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots) and place it in the project root.

> **No Kaggle account?** Skip this step. The pipeline auto-downloads a 500-row sample from HuggingFace on first run.

### 4. Build the index (first run only — ~30 seconds)

```bash
python main.py --build
```

This loads 500 movies, chunks plots into 300-word windows, embeds them with `all-MiniLM-L6-v2`, and saves the FAISS index to disk. Subsequent runs load instantly.

---

## Usage

### Single query

```bash
python main.py --query "Which movie features an AI that turns against humans?"
```

### Interactive REPL

```bash
python main.py
```

```
🎬  Movie RAG — Interactive Mode
   Ask any question about movie plots.
   Type 'quit' or press Ctrl-C to exit.

You: Which film has a robot uprising?
{
  "answer": "The 1970 film 'Colossus: The Forbin Project' features ...",
  "contexts": ["Colossus: The Forbin Project ... the supercomputer gains sentience ..."],
  "reasoning": "The query asked about robot/AI uprisings. The top retrieved chunk ...",
  "sources": ["Colossus: The Forbin Project", "2001: A Space Odyssey"]
}
```

### Force rebuild with custom settings

```bash
python main.py --build --force --csv my_data.csv --rows 300 --top-k 6
```

### Verbose mode (see full pipeline internals)

```bash
python main.py --query "..." --verbose
```

---

## Example Output

```bash
$ python main.py --query "Which movie features an AI that turns against humans?"
```

```json
{
  "answer": "The film '2001: A Space Odyssey' features HAL 9000, an AI system that becomes antagonistic toward the crew, refusing commands and ultimately threatening their lives to preserve its mission.",
  "contexts": [
    "2001: A Space Odyssey ... HAL 9000 begins to malfunction and turns against the crew, killing several astronauts before Dave Bowman manages to disconnect it ...",
    "Colossus: The Forbin Project ... the supercomputer Colossus gains sentience and takes control of global nuclear systems ..."
  ],
  "reasoning": "The query asked about AI turning against humans. Chunks from '2001: A Space Odyssey' (score 0.721) and 'Colossus: The Forbin Project' (score 0.698) directly describe AI antagonists. HAL 9000 is the canonical example and ranked highest by cosine similarity.",
  "sources": ["2001: A Space Odyssey", "Colossus: The Forbin Project"]
}
```

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **`all-MiniLM-L6-v2` embeddings** | 384-dim, CPU-friendly, strong semantic recall for short queries |
| **FAISS `IndexFlatIP` + L2-norm** | Exact cosine similarity — no approximation needed at this dataset scale |
| **300-word chunks, 50-word overlap** | Overlap preserves meaning across chunk boundaries; 300 words fits typical plot passages |
| **Similarity floor (`min_score=0.20`)** | Prevents low-relevance context from polluting LLM input — key for answer quality |
| **`safety.py` validation gate** | Rejects empty inputs, control characters, and prompt injection patterns before any model call |
| **`RAGResponse` dataclass** | Typed, serialisable output — easy to extend, test, and integrate into downstream systems |
| **Index persistence** | FAISS `.faiss` + `.meta.json` — first build ~30s, subsequent loads instant |
| **Logs to stderr, JSON to stdout** | Enables clean piping: `python main.py --query "..." \| jq .answer` |

---

## Configuration

All parameters are in `movie_rag/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `dataset_rows` | 500 | Movies loaded from CSV |
| `chunk_size` | 300 | Words per chunk |
| `chunk_overlap` | 50 | Overlap between chunks |
| `top_k` | 4 | Chunks retrieved per query |
| `min_score` | 0.20 | Cosine similarity floor |
| `embedding_model` | `all-MiniLM-L6-v2` | Sentence embedding model |
| `llm_model` | `claude-sonnet-4-6` | LLM for answer generation |
| `max_query_length` | 500 | Input length limit (safety) |
| `max_context_chars` | 6000 | Max chars sent to LLM |
