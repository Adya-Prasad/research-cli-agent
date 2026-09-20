import asyncio
from dataclasses import dataclass

import pytest

from research_agent.evaluation.graders import (
    default_graders,
)
from research_agent.evaluation.harness import (
    EvalHarness,
    limit_eval_suite,
)
from research_agent.evaluation.models import (
    CitedResearchReport,
    EvalProfile,
    EvalSuite,
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


@dataclass
class PassingSystem:
    name: str = "passing-system"
    profile: EvalProfile = "deterministic"

    async def run(
        self,
        task: EvalTask,
        trial_id: str,
    ) -> SystemOutput:
        evidence = WorkerEvidence(
            chunk_id="evidence-1",
            source="runtime.md",
            text=("A bounded runtime validates tool calls."),
            rank=1,
            score=1.0,
        )
        claim = ClaimDraft(
            claim_id=f"{trial_id}-claim-1",
            text="A bounded runtime validates tool calls.",
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

        return SystemOutput(
            system_name=self.name,
            profile=self.profile,
            status="completed",
            report=CitedResearchReport(
                markdown=claim.text,
                claims=(claim,),
                ledger=ledger,
            ),
            evidence=(evidence,),
            trajectory=(),
            usage=UsageMetrics(),
            duration_ms=5.0,
        )


def test_harness_runs_multiple_isolated_trials() -> None:
    suite = EvalSuite(
        schema_version="1",
        dataset_version="test-v1",
        tasks=(
            EvalTask(
                task_id="runtime",
                question="How should runtimes validate tools?",
            ),
        ),
    )

    report = asyncio.run(
        EvalHarness(
            system=PassingSystem(),
            graders=default_graders(),
        ).run(
            suite,
            trials_per_task=2,
        )
    )

    assert len(report.artifacts) == 2
    assert report.artifacts[0].trial_id != (report.artifacts[1].trial_id)
    assert report.summary.trial_pass_rate == 1.0
    assert report.summary.consistent_task_rate == 1.0
    assert report.regression.passed


def test_limit_eval_suite_creates_distinct_subset_version() -> None:
    suite = EvalSuite(
        schema_version="1",
        dataset_version="complete-v1",
        tasks=(
            EvalTask(
                task_id="one",
                question="Question one?",
            ),
            EvalTask(
                task_id="two",
                question="Question two?",
            ),
            EvalTask(
                task_id="three",
                question="Question three?",
            ),
        ),
    )

    limited = limit_eval_suite(suite, task_limit=2)

    assert [task.task_id for task in limited.tasks] == ["one", "two"]
    assert limited.dataset_version == ("complete-v1-subset-first-2")


def test_limit_eval_suite_rejects_excessive_limit() -> None:
    suite = EvalSuite(
        schema_version="1",
        dataset_version="complete-v1",
        tasks=(
            EvalTask(
                task_id="one",
                question="Question one?",
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="task_limit",
    ):
        limit_eval_suite(suite, task_limit=2)
