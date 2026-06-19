# EventRAG – RAG Application

<<<<<<< HEAD
A production-ready **Retrieval-Augmented Generation** app that lets you upload PDFs, chat with them, and get answers with citations. It’s built to be modular, observable, and easy to run locally.

![Streamlit UI](https://img.shields.io/badge/frontend-Streamlit-red) ![FastAPI](https://img.shields.io/badge/backend-FastAPI-green) ![Inngest](https://img.shields.io/badge/workflow-Inngest-blue) ![Qdrant](https://img.shields.io/badge/vector%20db-Qdrant-purple)

---
=======
A Retrieval-Augmented Generation (RAG) system that lets you upload PDFs and ask questions about them.

## Features

- PDF ingestion with chunking and semantic search
- Configurable embedding and LLM providers (free or paid)
- Event-driven pipeline with step-level retries via Inngest
- Qdrant for persistent vector storage

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | FastAPI |
| Workflow | Inngest |
| Embeddings | sentence-transformers (local) or OpenAI |
| LLM | Gemini 2.5 Flash (free) or GPT-4o-mini |
| Vector DB | Qdrant |
| PDF parsing | LlamaIndex PDFReader + SentenceSplitter |
>>>>>>> 4bb0fcb (Add provider-agnostic RAG with Gemini and local embeddings)

## What’s inside?

<<<<<<< HEAD
| Layer | Technology | Alternatives (if you’re on a budget) |
|-------|------------|--------------------------------------|
| Frontend | Streamlit | – |
| Backend | FastAPI | – |
| Workflow engine | Inngest (step‑based, retries, observability) | You can replace it with Celery or just call functions directly, but we like the visibility. |
| Embeddings | OpenAI `text-embedding-3-large` | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (local), or [Groq](https://groq.com) free API |
| LLM | OpenAI `gpt-4o-mini` | [Ollama](https://ollama.com) (llama3, mistral), [Groq](https://groq.com) (llama3‑70b free tier), or [Mistral AI](https://mistral.ai) |
| Vector DB | Qdrant | – |
| PDF parsing | LlamaIndex PDFReader + SentenceSplitter | – |
| Config | pydantic‑settings | – |

> **No OpenAI credits?** No problem. The code is built with `Config` that you can point to local embeddings and a local LLM. See the [Environment Variables](#environment-variables) section for how to switch.

---

## How it works (in plain English)

1. You upload a PDF via the Streamlit UI.
2. Streamlit triggers an Inngest event `"rag/ingest-pdf"`.
3. FastAPI picks it up and starts a **step‑based workflow**:
   - **Step 1**: Parse the PDF, split it into overlapping chunks.
   - **Step 2**: Embed each chunk (using your embedding model) and store the vectors + metadata in Qdrant.
4. When you ask a question, Streamlit fires `"rag/query-pdf"`:
   - **Step 1**: Embed your question and run a similarity search over Qdrant.
   - **Step 2**: Feed the top‑matching chunks to the LLM to generate a final answer (with source citations).
5. The answer appears in Streamlit, along with the chunks that were used.

All steps are retried automatically if something fails (e.g. the embedding API goes down) – only the failed step is retried, not the whole workflow. Nice, right?

---

## Quick Start (get it running in 5 minutes)

### What you need

- Python 3.11+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for Qdrant)
- [Inngest Dev Server](https://www.inngest.com/docs/local-development) – we use `npx` to run it
- An API key for your chosen LLM/embedding provider (or a local model)

### 1. Clone and prepare

```bash
git clone https://github.com/VikasPatil64/eventrag
cd eventrag
cp .env.example .env
# Edit .env – at minimum, fill in your provider keys (if using OpenAI, set OPENAI_API_KEY)
```

### 2. Install dependencies

We use `uv` (fast Python package manager), but you can also use `pip`:

```bash
uv sync          # or: pip install -e .
```

### 3. Fire up Qdrant

```bash
docker compose up qdrant -d
```

Qdrant dashboard: http://localhost:6333/dashboard

### 4. Start the Inngest Dev Server

```bash
npx inngest-cli@latest dev -u http://localhost:8000/api/inngest
```

You'll see the UI at http://localhost:8288 – handy for watching your workflows run.

### 5. Run the FastAPI backend

```bash
uvicorn main:app --reload --port 8000
```

Check it's alive: http://localhost:8000/health

### 6. Launch the Streamlit frontend

```bash
streamlit run streamlit_app.py
```

Open http://localhost:8501 and start uploading PDFs!

---

## Project Structure (what goes where)
=======
```
PDF Upload → Streamlit
  ↓ fires Inngest event "rag/ingest-pdf"
FastAPI → Inngest Dev Server
  ↓ Step 1: Load PDF + Chunk
  ↓ Step 2: Embed + Upsert to Qdrant

Query → Streamlit
  ↓ fires Inngest event "rag/query-pdf"
  ↓ Step 1: Embed query + Qdrant semantic search
  ↓ Step 2: LLM answer generation
  ↓ Streamlit displays answer + sources
```

## Project Structure
>>>>>>> 4bb0fcb (Add provider-agnostic RAG with Gemini and local embeddings)

```
eventrag/
├── app/
│   ├── config/
│   │   └── settings.py          # All configuration via pydantic-settings (single source of truth)
│   ├── ingestion/
<<<<<<< HEAD
│   │   ├── pdf_parser.py        # Loads PDF, splits into chunks
│   │   └── embedder.py          # Embedding logic (easily swappable)
│   ├── models/
│   │   └── schemas.py           # Pydantic models shared between API, DB, and UI
│   ├── retrieval/
│   │   └── vector_store.py      # Qdrant client (singleton, with connection handling)
│   └── utils/
│       └── logging_config.py    # Structured logging setup
├── main.py                      # FastAPI app + Inngest function definitions
├── streamlit_app.py             # Streamlit UI
├── docker-compose.yml           # Qdrant service definition
├── .env.example                 # Environment template (never commit .env)
└── pyproject.toml               # Dependencies and project metadata
=======
│   │   ├── pdf_parser.py        # PDF load + chunk
│   │   ├── embedder.py          # Embedding dispatcher (local | OpenAI)
│   │   └── local_embedder.py    # sentence-transformers implementation
│   ├── llm/
│   │   └── gemini_provider.py   # Gemini implementation
│   ├── models/
│   │   └── schemas.py           # Pydantic models
│   ├── retrieval/
│   │   └── vector_store.py      # Qdrant wrapper
│   └── utils/
│       └── logging_config.py    # Structured logging
├── main.py                      # FastAPI + Inngest functions
├── streamlit_app.py             # Streamlit UI
├── docker-compose.yml           # Qdrant service
├── .env.example
└── pyproject.toml
>>>>>>> 4bb0fcb (Add provider-agnostic RAG with Gemini and local embeddings)
```

## Setup

### Prerequisites

- Python 3.11+
- Docker Desktop (for local Qdrant)
- [Inngest Dev Server](https://www.inngest.com/docs/local-development)
- Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey) (free)

### 1. Install

```bash
git clone <repo>
cd rag-app
cp .env.example .env
# Set GEMINI_API_KEY in .env
uv sync
```

### 2. Start Qdrant

```bash
docker compose up qdrant -d
```

### 3. Start Inngest Dev Server

```bash
npx inngest-cli@latest dev -u http://localhost:8000/api/inngest
```

### 4. Start FastAPI

```bash
uvicorn main:app --reload --port 8000
```

Verify: http://localhost:8000/health

### 5. Start Streamlit

```bash
streamlit run streamlit_app.py
```

App: http://localhost:8501

## Environment Variables

<<<<<<< HEAD
The `.env.example` file lists everything you can tweak. Here are the most important ones:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | – | Your OpenAI key (required if using OpenAI). |
| `OPENAI_EMBED_MODEL` | `text-embedding-3-large` | Change to `sentence-transformers/all-MiniLM-L6-v2` or any local model (but you'll need to implement the embedding logic). |
| `OPENAI_EMBED_DIM` | `1536` | Must match the chosen model's output dimension. |
| `OPENAI_LLM_MODEL` | `gpt-4o-mini` | Swap for `ollama/llama3` or `groq/llama3-70b` if you adapt the LLM call. |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint. |
| `QDRANT_COLLECTION` | `docs` | Collection name inside Qdrant. |
| `QDRANT_SCORE_THRESHOLD` | `0.3` | Minimum similarity score; lower values return more (but maybe irrelevant) chunks. |
| `INNGEST_APP_ID` | `rag-app` | Must match what Inngest expects. |
| `CHUNK_SIZE` | `1024` | Characters per chunk. |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks. |
| `DEFAULT_TOP_K` | `5` | Number of chunks retrieved per query. |

**Switching to local/free models**:  
- Set `OPENAI_EMBED_MODEL` to a local model name and change the embedding call in `app/ingestion/embedder.py` to use `sentence-transformers`.  
- For the LLM, replace the `ctx.step.ai.infer` call in `main.py` with an HTTP call to Ollama or Groq.  
We deliberately kept the embedding and LLM logic isolated, so it's easy to swap.
=======
| Variable | Default | Description |
|---|---|---|
| `EMBED_PROVIDER` | `local` | `local` or `openai` |
| `LLM_PROVIDER` | `gemini` | `gemini` or `openai` |
| `GEMINI_API_KEY` | *required for gemini* | Google AI Studio key |
| `OPENAI_API_KEY` | *required for openai* | OpenAI key |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant instance |
| `QDRANT_COLLECTION` | `docs` | Collection name |
| `INNGEST_APP_ID` | `rag-app` | Must match in both main.py and streamlit |
| `CHUNK_SIZE` | `1024` | Characters per chunk |
| `DEFAULT_TOP_K` | `5` | Chunks retrieved per query |
>>>>>>> 4bb0fcb (Add provider-agnostic RAG with Gemini and local embeddings)

See `.env.example` for all options.

<<<<<<< HEAD
## Why these choices? (Key design decisions)

- **Inngest** – gives us step‑level retries, memoisation, and a beautiful UI. If the embedding API times out, only that step is retried – the PDF isn't reparsed. It also makes the whole flow auditable.
- **Qdrant** – persistent, production‑grade vector DB. It supports payload indexing, filtering, and runs smoothly in Docker. We index `source_id` so that later we can filter by document.
- **`text-embedding-3-large`** – we picked it for quality, but we **always** pass the `dimensions` parameter to the API so that the vector length matches what Qdrant expects – this solved a nasty mismatch bug.
- **pydantic‑settings** – keeps config clean, typed, and environment‑aware.

---

## Future Scope (what’s cooking)

- **Multi‑document management** – list, delete, rename uploaded documents.
- **Upload history** in the Streamlit sidebar so you can see past uploads.
- **Conversation memory** – multi‑turn chat with context.
- **Metadata filtering** – ask only about specific documents.
- **Page‑level citations** with source text display.
- **Dedicated `/upload` endpoint** (so you can upload without Streamlit).
- **Hybrid search** (BM25 + dense vectors) for better recall.
- **Cross‑encoder reranking** to improve the final answer quality.
- **Streaming LLM responses** (SSE) for a better UX.
- **RAG evaluation** with RAGAS (faithfulness, relevancy) to measure performance.
- **Prometheus metrics** + Grafana dashboard for monitoring.
- **Redis query cache** to speed up repeated questions.

---

## Development (tests, formatting, linting)

```bash
# Run tests (once we write some)
pytest tests/ -v

# Format code
ruff format .

# Lint
ruff check .
```

---

## Final words

This app handles the common RAG pitfalls (dimension mismatches, async issues, retries) so you can focus on retrieval quality or building a slick UI. Open an issue if you get stuck 

Happy building! 🚀
=======
> **Note on dimension change**: switching from `EMBED_PROVIDER=openai` (1536 dims) to `local` (384 dims) will automatically recreate the Qdrant collection. Re-ingest your PDFs after switching.

## Future Scope

- Multi-document management (list, delete)
- Conversation memory
- Metadata filtering
- Hybrid search (BM25 + dense)
- Reranking
- Streaming responses
- RAG evaluation
- Monitoring and caching

## License

MIT
>>>>>>> 4bb0fcb (Add provider-agnostic RAG with Gemini and local embeddings)
