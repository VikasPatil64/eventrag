# RAG Application

A production-grade **Retrieval-Augmented Generation** (RAG) application built on:

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | FastAPI |
| Workflow | Inngest (event-driven, step-based) |
| Embeddings | OpenAI `text-embedding-3-large` |
| LLM | OpenAI `gpt-4o-mini` |
| Vector DB | Qdrant |
| PDF parsing | LlamaIndex PDFReader + SentenceSplitter |
| Config | pydantic-settings |

---

## Architecture

```
PDF Upload
    ↓
Streamlit (streamlit_app.py)
    ↓  fires Inngest event "rag/ingest-pdf"
FastAPI (main.py)
    ↓
Inngest Dev Server
    ↓
Step 1: Load PDF + Chunk (app/ingestion/pdf_parser.py)
    ↓
Step 2: OpenAI Embeddings + Qdrant Upsert (app/ingestion/embedder.py)
    ↓
─── Query path ───
Streamlit fires "rag/query-pdf"
    ↓
Step 1: Embed query + Qdrant semantic search (app/retrieval/vector_store.py)
    ↓
Step 2: GPT-4o-mini answer generation (ctx.step.ai.infer)
    ↓
Streamlit displays answer + sources
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker Desktop (for Qdrant)
- [Inngest Dev Server](https://www.inngest.com/docs/local-development)
- OpenAI API key with billing enabled

### 1. Clone and install

```bash
git clone <repo>
cd rag-app
cp .env.example .env
# Fill in your OPENAI_API_KEY in .env
uv sync          # or: pip install -e .
```

### 2. Start Qdrant

```bash
docker compose up qdrant -d
```

Qdrant UI available at http://localhost:6333/dashboard

### 3. Start Inngest Dev Server

```bash
npx inngest-cli@latest dev -u http://localhost:8000/api/inngest
```

Inngest UI available at http://localhost:8288

### 4. Start FastAPI backend

```bash
uvicorn main:app --reload --port 8000
```

Verify: http://localhost:8000/health

### 5. Start Streamlit frontend

```bash
streamlit run streamlit_app.py
```

App at http://localhost:8501

---

## Project Structure

```
rag-app/
├── app/
│   ├── config/
│   │   └── settings.py          # pydantic-settings — single source of truth
│   ├── ingestion/
│   │   ├── pdf_parser.py        # PDF load + chunk
│   │   └── embedder.py          # OpenAI embedding wrapper
│   ├── models/
│   │   └── schemas.py           # Pydantic models (shared across layers)
│   ├── retrieval/
│   │   └── vector_store.py      # Qdrant wrapper (singleton)
│   └── utils/
│       └── logging_config.py    # Structured logging setup
├── main.py                      # FastAPI + Inngest functions
├── streamlit_app.py             # Streamlit UI
├── docker-compose.yml           # Qdrant service
├── .env.example                 # Environment template (never commit .env)
└── pyproject.toml
```

---

## Environment Variables

See [`.env.example`](.env.example) for all available options.

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *required* | OpenAI API key |
| `OPENAI_EMBED_MODEL` | `text-embedding-3-large` | Embedding model |
| `OPENAI_EMBED_DIM` | `1536` | Embedding dimensions |
| `OPENAI_LLM_MODEL` | `gpt-4o-mini` | LLM for answer generation |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant instance URL |
| `QDRANT_COLLECTION` | `docs` | Collection name |
| `QDRANT_SCORE_THRESHOLD` | `0.3` | Min similarity score |
| `INNGEST_APP_ID` | `rag-app` | Inngest app identifier |
| `CHUNK_SIZE` | `1024` | Chars per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `DEFAULT_TOP_K` | `5` | Chunks retrieved per query |

---

## Key Design Decisions

### Why Inngest?
Inngest provides step-level retries, memoisation, and observability. If OpenAI embedding fails mid-ingestion, only the embedding step is retried — the PDF doesn't get re-parsed.

### Why Qdrant?
Persistent, production-grade vector database with payload indexing, filtering, and Docker support. The `source_id` payload field is indexed for fast per-document filtering (V2 feature).

### Why `text-embedding-3-large`?
Best OpenAI embedding model for retrieval quality. The key fix in V1: always pass `dimensions=` to the API to guarantee the collection dimension and the actual vector dimension match.

---

## Roadmap

### V1 — Stabilisation (done ✅)
- Fixed app_id mismatch (silent event loss)
- Fixed embedding dimension mismatch (Qdrant rejection)
- Fixed asyncio safety in Streamlit
- Added error handling in all Inngest steps
- Added score threshold to prevent irrelevant chunk retrieval
- Added `/health` endpoint
- Structured logging across all modules
- Docker Compose for Qdrant
- pydantic-settings config management

### V2 — Production Features
- [ ] Multi-document management (list, delete documents)
- [ ] Upload history in Streamlit sidebar
- [ ] Conversation memory (multi-turn chat)
- [ ] Metadata filtering (query specific documents)
- [ ] Page-level citations with source passage display
- [ ] FastAPI `/upload` endpoint (decouple file handling from Streamlit)

### V3 — Resume-Worthy
- [ ] Hybrid search (BM25 + dense vectors)
- [ ] Cross-encoder reranking
- [ ] Streaming LLM responses (SSE)
- [ ] RAG evaluation with RAGAS (faithfulness, relevancy)
- [ ] Prometheus metrics + Grafana dashboard
- [ ] Redis query result caching

---

## Development

```bash
# Run tests
pytest tests/ -v

# Format
ruff format .

# Lint
ruff check .
```
