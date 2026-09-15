"""Construction of normalized external evidence snapshots."""

import hashlib

from research_agent.orchestration.models import (
    EvidenceScope,
    EvidenceSourceType,
    ScholarlyIdentifiers,
    WorkerEvidence,
)
from research_agent.sources.errors import SourceFormatInvalid
from research_agent.sources.text import normalize_inline


def evidence_id(
    provider: str,
    source_identifier: str,
    text: str,
) -> str:
    """Create a deterministic provider-specific snapshot ID."""

    digest = hashlib.sha256(
        (
            f"{provider}\x1f{source_identifier}\x1f{text}"
        ).encode()
    ).hexdigest()[:24]

    return f"{provider}:{digest}"


def make_external_evidence(
    *,
    provider: str,
    source_type: EvidenceSourceType,
    scope: EvidenceScope,
    source_identifier: str,
    title: str,
    canonical_url: str,
    text: str,
    rank: int,
    score: float,
    published_at: str | None = None,
    identifiers: ScholarlyIdentifiers | None = None,
) -> WorkerEvidence:
    """Build one validated external evidence snapshot."""

    normalized_provider = normalize_inline(provider)
    normalized_title = normalize_inline(title)
    normalized_text = text.strip()

    if not normalized_provider:
        raise SourceFormatInvalid("source provider is empty")
    if not normalized_title:
        raise SourceFormatInvalid("source title is empty")
    if not normalized_text:
        raise SourceFormatInvalid("source text is empty")

    return WorkerEvidence(
        chunk_id=evidence_id(
            normalized_provider,
            source_identifier,
            normalized_text,
        ),
        source=canonical_url,
        source_type=source_type,
        provider=normalized_provider,
        title=normalized_title,
        canonical_url=canonical_url,
        text=normalized_text,
        rank=rank,
        score=score,
        published_at=published_at,
        scope=scope,
        identifiers=(
            identifiers
            if identifiers is not None
            else ScholarlyIdentifiers()
        ),
        provider_chain=(normalized_provider,),
    )