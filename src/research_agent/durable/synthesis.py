"""Deterministic synthesis of complete and partial worker outcomes."""

from research_agent.durable.models import DurableRunStatus
from research_agent.orchestration.models import (
    ResearchPlan,
    WorkerResult,
)


def classify_run_status(
    results: tuple[WorkerResult, ...],
) -> DurableRunStatus:
    completed = sum(
        result.status == "completed"
        for result in results
    )

    if completed == len(results):
        return "completed"

    if completed == 0:
        return "failed"

    return "partial"


class DeterministicPartialSynthesizer:
    """Produce an evidence-preserving report without an LLM."""

    def synthesize(
        self,
        plan: ResearchPlan,
        results: tuple[WorkerResult, ...],
    ) -> str:
        status = classify_run_status(results)
        successful = sum(
            result.status == "completed"
            for result in results
        )

        lines = [
            f"Research status: {status.upper()}",
            f"Question: {plan.question}",
            (
                "Successful workers: "
                f"{successful}/{len(results)}"
            ),
            "",
            "Worker outcomes:",
        ]

        for result in results:
            detail = (
                result.answer
                or result.error_code
                or "No result detail"
            )
            eligible_evidence = sum(
                evidence.eligible_for_synthesis
                for evidence in result.evidence
            )

            lines.extend(
                [
                    (
                        f"[{result.role}] "
                        f"{result.status.upper()}"
                    ),
                    f"Detail: {detail}",
                    (
                        "Eligible evidence snapshots: "
                        f"{eligible_evidence}"
                    ),
                ]
            )

        if status != "completed":
            lines.extend(
                [
                    "",
                    (
                        "Warning: this is a partial result. "
                        "Missing worker evidence must not be "
                        "silently treated as successful research."
                    ),
                ]
            )

        return "\n".join(lines)