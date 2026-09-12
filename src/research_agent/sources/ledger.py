"""Deterministic claim-to-evidence citation integrity ledger."""

import hashlib
from typing import Literal

from pydantic import BaseModel, Field

from research_agent.orchestration.models import WorkerEvidence

CitationStatus = Literal[
    "citation_verified",
    "unsupported",
    "invalid_citation",
    "quarantined_source",
]


class Citation(BaseModel):
    """One exact quotation attributed to an evidence snapshot."""

    model_config = {"frozen": True}

    evidence_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class ClaimDraft(BaseModel):
    """A claim proposed for a research report."""

    model_config = {"frozen": True}

    claim_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    citations: tuple[Citation, ...] = ()


class ClaimAssessment(BaseModel):
    """Mechanical citation-integrity result for one claim."""

    model_config = {"frozen": True}

    claim_id: str = Field(min_length=1)
    claim_text: str = Field(min_length=1)
    status: CitationStatus
    reason: str = Field(min_length=1)
    evidence_ids: tuple[str, ...]


class ClaimLedger(BaseModel):
    """Ordered assessments for one report."""

    model_config = {"frozen": True}

    assessments: tuple[ClaimAssessment, ...]

    @property
    def verified_fraction(self) -> float:
        if not self.assessments:
            return 0.0

        verified = sum(
            item.status == "citation_verified"
            for item in self.assessments
        )
        return verified / len(self.assessments)


class ClaimLedgerVerifier:
    """Verify provenance mechanics, not semantic truth."""

    def verify(
        self,
        *,
        claims: tuple[ClaimDraft, ...],
        evidence: tuple[WorkerEvidence, ...],
    ) -> ClaimLedger:
        evidence_by_id: dict[str, WorkerEvidence] = {}

        for item in evidence:
            if item.chunk_id in evidence_by_id:
                raise ValueError(
                    "evidence IDs must be unique"
                )
            evidence_by_id[item.chunk_id] = item

        assessments = tuple(
            self._verify_claim(
                claim,
                evidence_by_id,
            )
            for claim in claims
        )
        return ClaimLedger(assessments=assessments)

    def _verify_claim(
        self,
        claim: ClaimDraft,
        evidence_by_id: dict[str, WorkerEvidence],
    ) -> ClaimAssessment:
        cited_ids = tuple(
            citation.evidence_id
            for citation in claim.citations
        )

        if not claim.citations:
            return ClaimAssessment(
                claim_id=claim.claim_id,
                claim_text=claim.text,
                status="unsupported",
                reason="claim has no citations",
                evidence_ids=(),
            )

        saw_quarantined_source = False

        for citation in claim.citations:
            item = evidence_by_id.get(
                citation.evidence_id
            )
            if item is None:
                return ClaimAssessment(
                    claim_id=claim.claim_id,
                    claim_text=claim.text,
                    status="invalid_citation",
                    reason="cited evidence does not exist",
                    evidence_ids=cited_ids,
                )

            actual_hash = hashlib.sha256(
                item.text.encode("utf-8")
            ).hexdigest()
            if (
                item.content_sha256
                and item.content_sha256 != actual_hash
            ):
                return ClaimAssessment(
                    claim_id=claim.claim_id,
                    claim_text=claim.text,
                    status="invalid_citation",
                    reason=(
                        "evidence content hash no longer matches"
                    ),
                    evidence_ids=cited_ids,
                )

            if citation.quote not in item.text:
                return ClaimAssessment(
                    claim_id=claim.claim_id,
                    claim_text=claim.text,
                    status="invalid_citation",
                    reason=(
                        "supporting quote is absent from "
                        "cited evidence"
                    ),
                    evidence_ids=cited_ids,
                )

            if not item.eligible_for_synthesis:
                saw_quarantined_source = True

        if saw_quarantined_source:
            return ClaimAssessment(
                claim_id=claim.claim_id,
                claim_text=claim.text,
                status="quarantined_source",
                reason="citation references quarantined evidence",
                evidence_ids=cited_ids,
            )

        return ClaimAssessment(
            claim_id=claim.claim_id,
            claim_text=claim.text,
            status="citation_verified",
            reason="all citations passed integrity checks",
            evidence_ids=cited_ids,
        )