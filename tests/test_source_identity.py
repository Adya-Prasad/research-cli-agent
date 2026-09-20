from research_agent.orchestration.models import (
    EvidenceScope,
    ScholarlyIdentifiers,
    WorkerEvidence,
)
from research_agent.sources.evidence import (
    make_external_evidence,
)
from research_agent.sources.identity import (
    canonical_evidence_identity,
    merge_duplicate_evidence,
)


def _paper(
    *,
    provider: str,
    scope: EvidenceScope,
    text: str,
    doi: str,
) -> WorkerEvidence:
    return make_external_evidence(
        provider=provider,
        source_type="paper_abstract",
        scope=scope,
        source_identifier=f"{provider}-record",
        title="Reliable Agent Runtimes",
        canonical_url="https://doi.org/10.1000/example",
        text=text,
        rank=1,
        score=1.0,
        published_at="2026",
        identifiers=ScholarlyIdentifiers(doi=doi),
    )


def test_identity_normalizes_doi() -> None:
    evidence = _paper(
        provider="crossref",
        scope="metadata",
        text="Reliable Agent Runtimes",
        doi="https://doi.org/10.1000/EXAMPLE",
    )

    assert canonical_evidence_identity(evidence) == ("doi:10.1000/example")


def test_richer_duplicate_wins_and_preserves_providers() -> None:
    metadata = _paper(
        provider="crossref",
        scope="metadata",
        text="Metadata only.",
        doi="10.1000/example",
    )
    abstract = _paper(
        provider="semantic-scholar",
        scope="abstract",
        text=("A complete abstract about bounded retries and durable checkpoints."),
        doi="10.1000/example",
    )

    merged = merge_duplicate_evidence(metadata, abstract)

    assert merged.provider == "semantic-scholar"
    assert merged.scope == "abstract"
    assert merged.provider_chain == (
        "crossref",
        "semantic-scholar",
    )
    assert merged.canonical_identity == ("doi:10.1000/example")
