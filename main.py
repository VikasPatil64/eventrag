"""
FastAPI application entry point + Inngest function definitions.

Run with:
    uvicorn main:app --reload --port 8000

Architecture:
    PDF Upload -> Streamlit -> FastAPI -> Inngest -> load -> chunk -> embed -> upsert
    -> query event -> semantic search -> LLM (Gemini or OpenAI) -> Streamlit response
"""

import logging
import uuid
from contextlib import asynccontextmanager

import inngest
import inngest.fast_api
from fastapi import FastAPI
from inngest.experimental import ai

from app.config.settings import get_settings
from app.ingestion.embedder import embed_texts
from app.ingestion.pdf_parser import load_and_chunk_pdf
from app.models.schemas import (
    RAGChunkAndSrc,
    RAGQueryResult,
    RAGSearchResult,
    RAGUpsertResult,
)
from app.retrieval.vector_store import get_vector_store
from app.utils.logging_config import configure_logging, get_logger

configure_logging(level=logging.INFO)
logger = get_logger(__name__)

settings = get_settings()

inngest_client = inngest.Inngest(
    app_id=settings.inngest_app_id,
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)


# ── Step helper functions ───────────────────────────────────────────────────


def _load_step(pdf_path: str, source_id: str) -> RAGChunkAndSrc:
    """Step 1 of ingest: load PDF and split into chunks."""
    logger.info("Step load-and-chunk: path=%s", pdf_path)
    try:
        chunks = load_and_chunk_pdf(pdf_path)
    except Exception as exc:
        logger.error("load_and_chunk_pdf failed for '%s': %s", pdf_path, exc)
        raise
    logger.info("Step load-and-chunk: produced %d chunk(s).", len(chunks))
    return RAGChunkAndSrc(chunks=chunks, source_id=source_id)


def _upsert_step(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
    """Step 2 of ingest: embed chunks and upsert to Qdrant."""
    if not chunks_and_src.chunks:
        logger.warning("No chunks to embed — skipping upsert (empty/image-only PDF?).")
        return RAGUpsertResult(ingested=0, source_id=chunks_and_src.source_id)

    source_id = chunks_and_src.source_id or "unknown"
    logger.info(
        "Step embed-and-upsert: embedding %d chunk(s) for source '%s'.",
        len(chunks_and_src.chunks),
        source_id,
    )

    try:
        vecs = embed_texts(chunks_and_src.chunks)
    except Exception as exc:
        logger.error("embed_texts failed: %s", exc)
        raise

    ids = [
        str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}"))
        for i in range(len(chunks_and_src.chunks))
    ]
    payloads = [
        {"source_id": source_id, "text": chunks_and_src.chunks[i]}
        for i in range(len(chunks_and_src.chunks))
    ]

    try:
        get_vector_store().upsert(ids, vecs, payloads)
    except Exception as exc:
        logger.error("Qdrant upsert failed: %s", exc)
        raise

    logger.info("Step embed-and-upsert: %d point(s) written to Qdrant.", len(ids))
    return RAGUpsertResult(ingested=len(ids), source_id=source_id)


def _search_step(question: str, top_k: int) -> RAGSearchResult:
    """Step 1 of query: embed the question and search Qdrant."""
    logger.info("Step embed-and-search: question=%r top_k=%d", question, top_k)

    try:
        query_vec = embed_texts([question])[0]
    except Exception as exc:
        logger.error("Query embedding failed: %s", exc)
        raise

    found = get_vector_store().search(
        query_vector=query_vec,
        top_k=top_k,
        score_threshold=settings.qdrant_score_threshold,
    )

    logger.info("Step embed-and-search: %d context(s) retrieved.", len(found["contexts"]))
    return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])


def _build_prompts(question: str, contexts: list[str]) -> tuple[str, str]:
    """Build the system and user prompts for the LLM."""
    context_block = "\n\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(contexts))
    system_prompt = (
        "You are a precise, factual document assistant. "
        "Answer ONLY using the provided context. "
        "Never fabricate information."
    )
    user_prompt = (
        "Use ONLY the following context passages to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Instructions:\n"
        "- Answer concisely and accurately using only the context above.\n"
        "- If the answer is not present in the context, say exactly: "
        "'I cannot find this information in the provided documents.'\n"
        "- Do NOT invent information not found in the context."
    )
    return system_prompt, user_prompt


def _gemini_answer_step(question: str, contexts: list[str]) -> dict:
    """Call Gemini to generate an answer. Returns dict for Inngest serialization."""
    from app.llm.gemini_provider import generate_answer

    system_prompt, user_prompt = _build_prompts(question, contexts)
    return {"answer": generate_answer(system_prompt, user_prompt)}


# ── Inngest functions ───────────────────────────────────────────────────────


@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest-pdf"),
    retries=2,
)
async def rag_ingest_pdf(ctx: inngest.Context) -> dict:
    pdf_path: str = ctx.event.data["pdf_path"]
    source_id: str = ctx.event.data.get("source_id", pdf_path)

    chunks_and_src: RAGChunkAndSrc = await ctx.step.run(
        "load-and-chunk",
        lambda: _load_step(pdf_path, source_id),
        output_type=RAGChunkAndSrc,
    )

    result: RAGUpsertResult = await ctx.step.run(
        "embed-and-upsert",
        lambda: _upsert_step(chunks_and_src),
        output_type=RAGUpsertResult,
    )

    logger.info("rag_ingest_pdf complete: %d chunk(s) ingested.", result.ingested)
    return result.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query-pdf"),
    retries=1,
)
async def rag_query_pdf_ai(ctx: inngest.Context) -> dict:
    question: str = ctx.event.data["question"]
    top_k: int = int(ctx.event.data.get("top_k", settings.default_top_k))

    found: RAGSearchResult = await ctx.step.run(
        "embed-and-search",
        lambda: _search_step(question, top_k),
        output_type=RAGSearchResult,
    )

    if not found.contexts:
        logger.warning("No relevant context found for question=%r", question)
        return RAGQueryResult(
            answer=(
                "I could not find relevant information in the uploaded documents "
                "for your question."
            ),
            sources=[],
            num_contexts=0,
        ).model_dump()

    system_prompt, user_prompt = _build_prompts(question, found.contexts)

    if settings.llm_provider == "gemini":
        result = await ctx.step.run(
            "llm-answer",
            lambda: _gemini_answer_step(question, found.contexts),
        )
        answer: str = result["answer"]
    else:
        # OpenAI path — uses Inngest's built-in AI adapter.
        adapter = ai.openai.Adapter(
            auth_keys=settings.openai_api_key,
            model=settings.openai_llm_model,
        )
        try:
            res = await ctx.step.ai.infer(
                "llm-answer",
                adapter=adapter,
                body={
                    "max_tokens": 1024,
                    "temperature": 0.1,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
            )
        except Exception as exc:
            logger.error("OpenAI LLM inference failed: %s", exc)
            raise
        answer = res["choices"][0]["message"]["content"].strip()

    logger.info("rag_query_pdf_ai complete: answer length=%d chars.", len(answer))
    return RAGQueryResult(
        answer=answer,
        sources=found.sources,
        num_contexts=len(found.contexts),
    ).model_dump()


# ── FastAPI application ─────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "RAG app starting — embed_provider=%s, embed_model=%s, embed_dim=%d, "
        "llm_provider=%s, llm_model=%s, qdrant=%s, collection=%s",
        settings.embed_provider,
        settings.active_embed_model,
        settings.active_embed_dim,
        settings.llm_provider,
        settings.active_llm_model,
        settings.qdrant_url,
        settings.qdrant_collection,
    )
    try:
        get_vector_store()
        logger.info("Qdrant connection verified at startup.")
    except Exception as exc:
        logger.error("Could not connect to Qdrant at startup: %s", exc)
    yield
    logger.info("RAG app shutting down.")


app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation backend — FastAPI + Inngest + Qdrant.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["ops"])
async def health() -> dict:
    """Liveness check. Returns Qdrant status and active provider config."""
    try:
        info = get_vector_store().client.get_collections()
        qdrant_status = f"ok ({len(info.collections)} collection(s))"
    except Exception as exc:
        qdrant_status = f"error: {exc}"

    return {
        "status": "ok",
        "qdrant": qdrant_status,
        "embed_provider": settings.embed_provider,
        "embed_model": settings.active_embed_model,
        "embed_dim": settings.active_embed_dim,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.active_llm_model,
        "collection": settings.qdrant_collection,
    }


inngest.fast_api.serve(
    app,
    inngest_client,
    functions=[rag_ingest_pdf, rag_query_pdf_ai],
)