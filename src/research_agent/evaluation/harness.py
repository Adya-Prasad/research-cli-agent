"""Sequential, isolated task/trial evaluation harness."""

import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from research_agent.evaluation.metrics import build_summary
from research_agent.evaluation.models import (
    EvalBaseline,
    EvalReport,
    EvalSuite,
    GradeResult,
    SystemOutput,
    TrialArtifact,
)
from research_agent.evaluation.ports import (
    Grader,
    ResearchSystem,
)
from research_agent.evaluation.regression import (
    evaluate_regression,
)


def load_eval_suite(path: Path) -> EvalSuite:
    return EvalSuite.model_validate_json(path.read_text(encoding="utf-8"))


def _functional_digest(
    output: SystemOutput | None,
    grades: tuple[GradeResult, ...],
    error: str | None,
) -> str:
    payload = {
        "output": (
            None
            if output is None
            else {
                "status": output.status,
                "markdown": output.report.markdown,
                "claims": [claim.model_dump(mode="json") for claim in output.report.claims],
                "rejected_claims": [
                    claim.model_dump(mode="json") for claim in (output.report.rejected_claims)
                ],
                "evidence_ids": [item.chunk_id for item in output.evidence],
            }
        ),
        "grades": [
            {
                "name": grade.name,
                "kind": grade.kind,
                "score": grade.score,
                "passed": grade.passed,
            }
            for grade in grades
        ],
        "error": error,
    }

    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class EvalHarness:
    system: ResearchSystem
    graders: tuple[Grader, ...]
    baseline: EvalBaseline | None = None

    async def run(
        self,
        suite: EvalSuite,
        trials_per_task: int,
    ) -> EvalReport:
        if not 1 <= trials_per_task <= 10:
            raise ValueError("trials_per_task must be between 1 and 10")
        if not self.graders:
            raise ValueError("evaluation requires at least one grader")

        artifacts: list[TrialArtifact] = []

        # Sequential trials avoid API bursts and make local resource
        # contention less likely to distort latency comparisons.
        for task in suite.tasks:
            for trial_index in range(
                1,
                trials_per_task + 1,
            ):
                trial_id = f"{task.task_id}-{self.system.profile}-{trial_index:03d}"

                try:
                    output = await self.system.run(
                        task,
                        trial_id,
                    )
                    grades = tuple(grader.grade(task, output) for grader in self.graders)
                    passed = all(grade.passed for grade in grades if grade.kind == "hard")
                    error = None
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    output = None
                    grades = (
                        GradeResult(
                            name="system_execution",
                            kind="hard",
                            score=0.0,
                            passed=False,
                            explanation=(f"system raised {exc.__class__.__name__}"),
                        ),
                    )
                    passed = False
                    error = f"{exc.__class__.__name__}: {str(exc)[:200]}"

                artifacts.append(
                    TrialArtifact(
                        task_id=task.task_id,
                        trial_id=trial_id,
                        trial_index=trial_index,
                        profile=self.system.profile,
                        output=output,
                        grades=grades,
                        passed=passed,
                        functional_digest=_functional_digest(
                            output,
                            grades,
                            error,
                        ),
                        error=error,
                    )
                )

        artifact_tuple = tuple(artifacts)
        summary = build_summary(artifact_tuple)
        regression = evaluate_regression(
            dataset_version=suite.dataset_version,
            summary=summary,
            thresholds=suite.thresholds,
            baseline=self.baseline,
        )

        return EvalReport(
            schema_version=suite.schema_version,
            dataset_version=suite.dataset_version,
            profile=self.system.profile,
            system_name=self.system.name,
            trials_per_task=trials_per_task,
            artifacts=artifact_tuple,
            summary=summary,
            regression=regression,
        )


def limit_eval_suite(
    suite: EvalSuite,
    task_limit: int,
) -> EvalSuite:
    """Create an explicitly versioned prefix subset for smoke tests."""

    if not 1 <= task_limit <= len(suite.tasks):
        raise ValueError(f"task_limit must be between 1 and {len(suite.tasks)}")

    return suite.model_copy(
        update={
            "dataset_version": (f"{suite.dataset_version}-subset-first-{task_limit}"),
            "tasks": suite.tasks[:task_limit],
        }
    )
