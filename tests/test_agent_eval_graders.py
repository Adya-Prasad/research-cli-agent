from research_agent.evaluation.graders import (
    BudgetGrader,
    CitationIntegrityGrader,
    ExpectedTermsGrader,
)
from research_agent.evaluation.models import (
    CitedResearchReport,
    EvalTask,
    SystemOutput,
    UsageMetrics,
)
from research_agent.orchestration.models import WorkerEvidence
from research_agent.sources.ledger import (
    Citation,
    ClaimDraft,
    ClaimLedgerVerifier,
)


def _output() -> SystemOutput:
    evidence = WorkerEvidence(
        chunk_id="evidence-1",
        source="runtime.md",
        text=("A bounded runtime validates tool calls before execution."),
        rank=1,
        score=1.0,
    )
    claim = ClaimDraft(
        claim_id="claim-1",
        text="A bounded runtime validates tool calls.",
        citations=(
            Citation(
                evidence_id="evidence-1",
                quote="validates tool calls before execution",
            ),
        ),
    )
    ledger = ClaimLedgerVerifier().verify(
        claims=(claim,),
        evidence=(evidence,),
    )

    return SystemOutput(
        system_name="test-system",
        profile="deterministic",
        status="completed",
        report=CitedResearchReport(
            markdown=("A bounded runtime validates tool calls."),
            claims=(claim,),
            ledger=ledger,
        ),
        evidence=(evidence,),
        trajectory=(),
        usage=UsageMetrics(
            model_calls=0,
            tool_calls=3,
        ),
        duration_ms=10.0,
    )


def test_citation_grader_requires_verified_claims() -> None:
    result = CitationIntegrityGrader().grade(
        EvalTask(
            task_id="runtime",
            question="How do runtimes validate tools?",
        ),
        _output(),
    )

    assert result.passed
    assert result.score == 1.0


def test_budget_grader_detects_tool_call_excess() -> None:
    result = BudgetGrader().grade(
        EvalTask(
            task_id="runtime",
            question="How do runtimes validate tools?",
            max_tool_calls=2,
        ),
        _output(),
    )

    assert not result.passed


def test_expected_terms_is_soft_quality_signal() -> None:
    result = ExpectedTermsGrader().grade(
        EvalTask(
            task_id="runtime",
            question="How do runtimes validate tools?",
            required_terms=(
                "bounded",
                "checkpoint",
            ),
            minimum_term_coverage=1.0,
        ),
        _output(),
    )

    assert result.kind == "soft"
    assert result.score == 0.5
    assert not result.passed
