"""
DEPRECATED — replaced by app/ingestion/pdf_parser.py and app/ingestion/embedder.py

This file is kept temporarily for reference only.
Do NOT import from it — all imports should use the new module paths:

    from app.ingestion.pdf_parser import load_and_chunk_pdf
    from app.ingestion.embedder import embed_texts
"""
raise ImportError(
    "data_loader.py is deprecated. "
    "Import from app.ingestion.pdf_parser and app.ingestion.embedder instead."
)
