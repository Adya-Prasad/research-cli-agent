"""Stage 1 of the retrieval subsystem: files on disk -> ResearchDocument.

Ingestion's entire job is I/O and format normalization. It does not know
what a chunk is, what an embedding is, or what BM25 is -- it knows how to
read a file and hand back typed, validated text. That narrow boundary is
what lets you later swap the *source* (a database export, an S3 bucket,
a Notion API) without chunking.py or retrieval.py noticing, because they
only ever see ResearchDocument objects, never a filesystem path directly.
"""

from __future__ import annotations

import logging
from pathlib import Path

from research_agent.retrieval.models import DocumentType, ResearchDocument

logger = logging.getLogger(__name__)

# Phase 1 supports plain-text formats only. PDFs and HTML need their own
# extraction step (layout-aware text pulling, tag stripping) that belongs
# in a dedicated parser, not bolted onto this loop as more branches.
_SUPPORTED_SUFFIXES: dict[str, DocumentType] = {".md": "md", ".txt": "txt"}


def load_documents(root: Path) -> list[ResearchDocument]:
    """Walk `root`, read every supported file, return typed documents.

    Files are visited in sorted path order, not `os.walk` order (which
    is filesystem-dependent and not guaranteed stable). Sorted order is
    what makes doc_id sequences -- and therefore chunk_id sequences --
    reproducible between two runs over an unchanged directory, which is
    the property the whole recall@k evaluation depends on: a query's
    LabeledQuery.relevant_chunk_ids only stays valid if re-ingesting the
    same corpus keeps producing the same ids.

    Unreadable or empty files are skipped with a warning rather than
    raising -- one malformed file in a 500-document corpus shouldn't take
    down the whole ingestion run. Silent skips would be worse: you'd
    lose a document's worth of coverage from your index and only notice
    when recall@k mysteriously drops on unrelated queries.
    """
    root = Path(root)

    documents: list[ResearchDocument] = []
    if not root.is_dir():
        raise ValueError(f"Corpus directory does not exist: {root}")

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        doc_type = _SUPPORTED_SUFFIXES.get(path.suffix.lower())
        if doc_type is None:
            continue

        try:
            text = path.read_text(encoding="utf-8").strip()
        except (UnicodeDecodeError, OSError) as exc:
            logger.warning("Skipping unreadable file %s: %s", path, exc)
            continue

        if not text:
            logger.warning("Skipping empty file %s", path)
            continue
        # pass the corpus relative Path
        relative_path = path.relative_to(root)
        documents.append(
            ResearchDocument.from_text(source_path=relative_path, doc_type=doc_type, text=text)
        )

    logger.info("Ingested %d document(s) from %s", len(documents), root)
    return documents
