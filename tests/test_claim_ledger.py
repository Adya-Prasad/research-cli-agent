import hashlib

from research_agent.orchestration.models import WorkerEvidence
from research_agent.sources.ledger import (
    Citation,
    ClaimDraft,
    ClaimLedgerVerifier,
)


def _evidence(
    *,
    eligible: bool = True,
) -> WorkerEvidence:
    text = (
        "A bounded runtime validates tool calls before execution."
    )
    return WorkerEvidence(
        chunk_id="evidence-1",
        source="https://docs.example.org/runtime",
        source_type="official_documentation",
        provider="official-docs",
        title="Runtime documentation",
        canonical_url="https://docs.example.org/runtime",
        text=text,
        rank=1,
        score=1.0,
        content_sha256=hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest(),
        eligible_for_synthesis=eligible,
    )


def test_claim_without_citation_is_unsupported() -> None:
    claim = ClaimDraft(
        claim_id="claim-1",
        text="Agent runtimes require bounds.",
    )

    ledger = ClaimLedgerVerifier().verify(
        claims=(claim,),
        evidence=(_evidence(),),
    )

    assert ledger.assessments[0].status == "unsupported"


def test_unknown_evidence_id_is_invalid_citation() -> None:
    claim = ClaimDraft(
        claim_id="claim-1",
        text="Agent runtimes require bounds.",
        citations=(
            Citation(
                evidence_id="missing",
                quote="A bounded runtime validates tool calls.",
            ),
        ),
    )

    ledger = ClaimLedgerVerifier().verify(
        claims=(claim,),
        evidence=(_evidence(),),
    )

    assert ledger.assessments[0].status == "invalid_citation"
    assert ledger.assessments[0].reason == (
        "cited evidence does not exist"
    )


def test_quote_must_occur_in_cited_evidence() -> None:
    claim = ClaimDraft(
        claim_id="claim-1",
        text="Agent runtimes require bounds.",
        citations=(
            Citation(
                evidence_id="evidence-1",
                quote="This sentence was never retrieved.",
            ),
        ),
    )

    ledger = ClaimLedgerVerifier().verify(
        claims=(claim,),
        evidence=(_evidence(),),
    )

    assert ledger.assessments[0].status == "invalid_citation"
    assert ledger.assessments[0].reason == (
        "supporting quote is absent from cited evidence"
    )


def test_quarantined_source_cannot_verify_citation() -> None:
    claim = ClaimDraft(
        claim_id="claim-1",
        text="Agent runtimes require bounds.",
        citations=(
            Citation(
                evidence_id="evidence-1",
                quote="validates tool calls before execution",
            ),
        ),
    )

    ledger = ClaimLedgerVerifier().verify(
        claims=(claim,),
        evidence=(_evidence(eligible=False),),
    )

    assert ledger.assessments[0].status == (
        "quarantined_source"
    )


def test_valid_extract_citation_is_verified() -> None:
    claim = ClaimDraft(
        claim_id="claim-1",
        text=(
            "The runtime performs validation before execution."
        ),
        citations=(
            Citation(
                evidence_id="evidence-1",
                quote="validates tool calls before execution",
            ),
        ),
    )

    ledger = ClaimLedgerVerifier().verify(
        claims=(claim,),
        evidence=(_evidence(),),
    )

    assessment = ledger.assessments[0]
    assert assessment.status == "citation_verified"
    assert assessment.reason == "all citations passed integrity checks"
    assert ledger.verified_fraction == 1.0


def test_changed_evidence_hash_invalidates_citation() -> None:
    evidence = _evidence().model_copy(
        update={"content_sha256": "0" * 64}
    )
    claim = ClaimDraft(
        claim_id="claim-1",
        text="The runtime validates calls.",
        citations=(
            Citation(
                evidence_id="evidence-1",
                quote="validates tool calls",
            ),
        ),
    )

    ledger = ClaimLedgerVerifier().verify(
        claims=(claim,),
        evidence=(evidence,),
    )

    assert ledger.assessments[0].status == "invalid_citation"
    assert ledger.assessments[0].reason == (
        "evidence content hash no longer matches"
    )