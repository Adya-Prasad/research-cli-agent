from research_agent.orchestration.models import (
    ScholarlyIdentifiers,
)
from research_agent.sources.evidence import (
    make_external_evidence,
)


def test_external_evidence_normalizes_and_preserves_provenance() -> None:
    evidence = make_external_evidence(
        provider="crossref",
        source_type="paper_abstract",
        scope="metadata",
        source_identifier="10.1000/example",
        title="  Reliable   Agents  ",
        canonical_url="https://doi.org/10.1000/example",
        text="  Metadata about reliable agents.  ",
        rank=1,
        score=1.0,
        identifiers=ScholarlyIdentifiers(
            doi="10.1000/example"
        ),
    )

    assert evidence.title == "Reliable Agents"
    assert evidence.text == "Metadata about reliable agents."
    assert evidence.provider_chain == ("crossref",)
    assert evidence.identifiers.doi == "10.1000/example"
    assert evidence.chunk_id.startswith("crossref:")