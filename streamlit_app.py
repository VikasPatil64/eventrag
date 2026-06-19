"""
Streamlit frontend for the RAG application.
"""

import threading
import asyncio
import time
from pathlib import Path

import requests
import streamlit as st
import inngest
from dotenv import load_dotenv

from app.config.settings import get_settings

load_dotenv()

settings = get_settings()

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG — Document QA",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    top_k = st.slider(
        "Chunks to retrieve (top-k)",
        min_value=1,
        max_value=20,
        value=settings.default_top_k,
        step=1,
        help="How many document chunks are retrieved and sent to the LLM.",
    )
    st.caption(f"**Embed:** `{settings.embed_provider}` / `{settings.active_embed_model}`")
    st.caption(f"**LLM:** `{settings.llm_provider}` / `{settings.active_llm_model}`")
    st.caption(f"**Collection:** `{settings.qdrant_collection}`")
    st.caption(f"**Embed dim:** `{settings.active_embed_dim}`")

    st.divider()
    if st.button("🏥 Health check"):
        try:
            r = requests.get("http://localhost:8000/health", timeout=5)
            r.raise_for_status()
            data = r.json()
            st.success(f"API: {data.get('status')} | Qdrant: {data.get('qdrant')}")
        except Exception as exc:
            st.error(f"Health check failed: {exc}")


# ── Inngest client (cached so one instance per Streamlit session) ───────────
@st.cache_resource
def _inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id=settings.inngest_app_id, is_production=False)


# ── Async event helpers ─────────────────────────────────────────────────────

def _run_async(coro) -> object:
    """
    Run an async coroutine safely from Streamlit's sync context.

    Streamlit runs its own event loop; spinning a daemon thread with a fresh
    loop avoids "This event loop is already running" errors in all environments.
    """
    result_holder: list = []
    error_holder: list = []

    def _target():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result_holder.append(loop.run_until_complete(coro))
        except Exception as exc:
            error_holder.append(exc)
        finally:
            loop.close()

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join()

    if error_holder:
        raise error_holder[0]
    return result_holder[0] if result_holder else None


async def _send_ingest_event(pdf_path: Path) -> str:
    client = _inngest_client()
    events = await client.send(
        inngest.Event(
            name="rag/ingest-pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )
    return events[0] if events else ""


async def _send_query_event(question: str, top_k: int) -> str:
    client = _inngest_client()
    events = await client.send(
        inngest.Event(
            name="rag/query-pdf",
            data={"question": question, "top_k": top_k},
        )
    )
    return events[0] if events else ""


# ── Inngest run polling ─────────────────────────────────────────────────────

def _inngest_api_base() -> str:
    return f"{settings.inngest_dev_server_url}/v1"


def _fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json().get("data", [])


def _wait_for_run_output(
    event_id: str,
    timeout_s: float = 180.0,
    poll_interval_s: float = 0.75,
) -> dict:
    """
    Poll Inngest dev-server until the function run reaches a terminal state.

    Inngest run statuses: Running → Completed | Failed | Cancelled.
    """
    start = time.monotonic()
    last_status: str | None = None

    while True:
        runs = _fetch_runs(event_id)
        if runs:
            run = runs[0]
            status: str = run.get("status", "")
            last_status = status or last_status

            if status == "Completed":
                raw_output = run.get("output") or {}
                # Defensively handle Inngest's potential wrapping of output.
                if isinstance(raw_output, dict) and "body" in raw_output:
                    return raw_output["body"]
                return raw_output

            if status in ("Failed", "Cancelled"):
                raise RuntimeError(
                    f"Inngest function run ended with status '{status}'. "
                    "Check the Inngest Dev Server UI at http://localhost:8288 "
                    "for the full error trace."
                )

        if time.monotonic() - start > timeout_s:
            raise TimeoutError(
                f"Timed out after {timeout_s}s waiting for Inngest run "
                f"(event_id={event_id!r}, last_status={last_status!r}). "
                "Is the FastAPI server running on port 8000?"
            )

        time.sleep(poll_interval_s)


# ── File upload helper ──────────────────────────────────────────────────────

def _save_uploaded_pdf(file) -> Path:
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / file.name
    dest.write_bytes(file.getbuffer())
    return dest


# ── UI ──────────────────────────────────────────────────────────────────────

st.title("📚 Document QA — RAG Application")
st.caption(
    "Upload a PDF to ingest it into the knowledge base, "
    "then ask questions about its contents."
)

# ── Section 1: Upload ───────────────────────────────────────────────────────
st.subheader("1️⃣  Ingest a PDF")
uploaded = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"],
    accept_multiple_files=False,
    help="The PDF will be chunked, embedded, and stored in Qdrant.",
)

if uploaded is not None:
    with st.spinner(f"Saving and triggering ingestion of **{uploaded.name}**…"):
        pdf_path = _save_uploaded_pdf(uploaded)
        event_id = _run_async(_send_ingest_event(pdf_path))  # type: ignore[arg-type]

    if event_id:
        st.success(
            f"✅ Ingestion triggered for **{uploaded.name}**  \n"
            f"Event ID: `{event_id}`  \n"
            "Processing happens asynchronously — watch the "
            "[Inngest Dev Server](http://localhost:8288) for progress."
        )
    else:
        st.warning(
            "Event sent but no event ID returned. "
            "Check that the Inngest Dev Server is running at "
            f"`{settings.inngest_dev_server_url}`."
        )

st.divider()

# ── Section 2: Query ────────────────────────────────────────────────────────
st.subheader("2️⃣  Ask a Question")

with st.form("rag_query_form"):
    question = st.text_area(
        "Your question",
        placeholder="What is the main topic of the document?",
        height=80,
    )
    submitted = st.form_submit_button("🔍 Ask", use_container_width=True)

if submitted and question.strip():
    with st.spinner("Embedding query and searching knowledge base…"):
        try:
            event_id = _run_async(_send_query_event(question.strip(), int(top_k)))
            output = _wait_for_run_output(event_id)
        except TimeoutError as exc:
            st.error(f"⏱️ Timeout: {exc}")
            st.stop()
        except RuntimeError as exc:
            st.error(f"❌ Function failed: {exc}")
            st.stop()
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")
            st.stop()

    answer: str = output.get("answer", "")
    sources: list[str] = output.get("sources", [])
    num_contexts: int = output.get("num_contexts", 0)

    st.subheader("💬 Answer")
    st.write(answer or "*(No answer returned)*")

    st.caption(f"Retrieved **{num_contexts}** context chunk(s) from Qdrant.")

    if sources:
        with st.expander("📄 Sources"):
            for s in sources:
                st.write(f"- `{s}`")

elif submitted and not question.strip():
    st.warning("Please enter a question before submitting.")
