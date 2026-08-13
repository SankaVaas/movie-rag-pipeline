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
    │  ✗ filter below similarity floor (min_score=0.35)
    ▼
Context Assembly + LLM (Groq API: llama-3.3-70b-versatile)
    │  hosted inference — fast, requires GROQ_API_KEY
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
│   ├── generator.py           ← Groq API call + JSON parsing
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
export GROQ_API_KEY=gsk_...
```

Get a free key at [console.groq.com](https://console.groq.com/keys) — Groq's free tier is generous and inference is very fast, so per-query latency is dominated by embedding/retrieval, not generation.

### 3. Get the dataset

Download `wiki_movie_plots_deduped.csv` from [Kaggle](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots) and place it in the project root.

> **No Kaggle account?** Skip this step. The pipeline auto-downloads a sample from HuggingFace on first run.

### 4. Build the index (first run only)

```bash
python main.py --build
```

This loads the movies, chunks plots into 300-word windows, embeds them with `all-MiniLM-L6-v2`, and saves the FAISS index to disk. The embedding model is downloaded once to the Hugging Face cache (`~/.cache/huggingface`); subsequent `--build`/`--load` runs reuse the cached weights. No local LLM weights are downloaded — generation runs through the Groq API.

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

> The REPL loads the embedding model once and reuses it for every question in the session — much faster per-query than invoking `main.py --query` repeatedly, since each CLI invocation is a fresh process.

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
| **Similarity floor (`min_score=0.35`)** | Prevents low-relevance context from polluting LLM input — key for answer quality |
| **`safety.py` validation gate** | Rejects empty inputs, control characters, and prompt injection patterns before any model call |
| **`RAGResponse` dataclass** | Typed, serialisable output — easy to extend, test, and integrate into downstream systems |
| **Index persistence** | FAISS `.faiss` + `.meta.json` — first build ~30s, subsequent loads instant |
| **Groq-hosted LLM** | `llama-3.3-70b-versatile` via the Groq API — much stronger grounding/instruction-following than a small local model, and Groq's inference speed keeps generation well under a second per query |
| **Logs to stderr, JSON to stdout** | Enables clean piping: `python main.py --query "..." \| jq .answer` |

---

## Configuration

All parameters are in `movie_rag/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `dataset_rows` | 1000 | Movies loaded from CSV |
| `chunk_size` | 300 | Words per chunk |
| `chunk_overlap` | 50 | Overlap between chunks |
| `top_k` | 4 | Chunks retrieved per query |
| `min_score` | 0.35 | Cosine similarity floor |
| `embedding_model` | `all-MiniLM-L6-v2` | Sentence embedding model |
| `llm_model` | `llama-3.3-70b-versatile` | Groq-hosted model for answer generation |
| `max_query_length` | 500 | Input length limit (safety) |
| `max_context_chars` | 6000 | Max chars sent to LLM |

---

## Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | Yes | Authenticates calls to the Groq API in `generator.py` |