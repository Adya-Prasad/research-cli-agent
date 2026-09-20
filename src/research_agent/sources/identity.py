"""Canonical identity and deterministic duplicate resolution."""

import hashlib
import re

from research_agent.orchestration.models import (
    EvidenceScope,
    WorkerEvidence,
)
from research_agent.sources.text import normalize_inline

_SCOPE_RICHNESS: dict[EvidenceScope, int] = {
    "metadata": 1,
    "local_excerpt": 2,
    "web_excerpt": 2,
    "documentation_excerpt": 3,
    "code_excerpt": 3,
    "abstract": 4,
}


def _normalize_doi(value: str) -> str:
    normalized = normalize_inline(value).casefold()

    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "doi:",
    ):
        normalized = normalized.removeprefix(prefix)

    return normalized.strip()


def _normalize_arxiv_id(value: str) -> str:
    normalized = normalize_inline(value).casefold()

    for prefix in (
        "https://arxiv.org/abs/",
        "http://arxiv.org/abs/",
        "arxiv:",
    ):
        normalized = normalized.removeprefix(prefix)

    return re.sub(r"v\d+$", "", normalized).strip()


def canonical_evidence_identity(
    evidence: WorkerEvidence,
) -> str:
    """Resolve a provider-independent identity where possible."""

    identifiers = evidence.identifiers

    if identifiers.doi:
        doi = _normalize_doi(identifiers.doi)
        if doi:
            return f"doi:{doi}"

    if identifiers.arxiv_id:
        arxiv_id = _normalize_arxiv_id(identifiers.arxiv_id)
        if arxiv_id:
            return f"arxiv:{arxiv_id}"

    if identifiers.semantic_scholar_id:
        semantic_id = normalize_inline(identifiers.semantic_scholar_id).casefold()

        if semantic_id:
            return f"s2:{semantic_id}"

    # Title/year matching is suitable only for scholarly records.
    # Applying it to arbitrary excerpts can collapse distinct chunks
    # from the same document or provider.
    if evidence.source_type == "paper_abstract":
        normalized_title = re.sub(
            r"[^a-z0-9]+",
            " ",
            normalize_inline(evidence.title).casefold(),
        ).strip()

        year_match = re.search(
            r"\b(?:19|20)\d{2}\b",
            evidence.published_at or "",
        )
        year = year_match.group(0) if year_match else "unknown"

        if normalized_title:
            digest = hashlib.sha256((f"{normalized_title}\x1f{year}").encode()).hexdigest()[:24]

            return f"title-year:{digest}"

    # Non-scholarly excerpts are independent snapshots.
    return f"snapshot:{evidence.chunk_id}"


def with_canonical_identity(
    evidence: WorkerEvidence,
) -> WorkerEvidence:
    """Attach identity and initialize provider provenance."""

    chain = evidence.provider_chain if evidence.provider_chain else (evidence.provider,)

    return evidence.model_copy(
        update={
            "canonical_identity": (canonical_evidence_identity(evidence)),
            "provider_chain": chain,
        }
    )


def _richness_key(
    evidence: WorkerEvidence,
) -> tuple[int, int, int, str]:
    return (
        int(evidence.eligible_for_synthesis),
        _SCOPE_RICHNESS[evidence.scope],
        len(evidence.text),
        evidence.provider,
    )


def merge_duplicate_evidence(
    left: WorkerEvidence,
    right: WorkerEvidence,
) -> WorkerEvidence:
    """Keep the richer snapshot while retaining provider lineage."""

    prepared_left = with_canonical_identity(left)
    prepared_right = with_canonical_identity(right)

    if prepared_left.canonical_identity != prepared_right.canonical_identity:
        raise ValueError("cannot merge evidence with different identities")

    winner = max(
        (prepared_left, prepared_right),
        key=_richness_key,
    )

    providers = tuple(
        dict.fromkeys(
            (
                *prepared_left.provider_chain,
                *prepared_right.provider_chain,
            )
        )
    )

    return winner.model_copy(
        update={
            "canonical_identity": (prepared_left.canonical_identity),
            "provider_chain": providers,
        }
    )
