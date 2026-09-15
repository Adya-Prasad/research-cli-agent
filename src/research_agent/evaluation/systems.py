"""Research-system implementations evaluated by the harness."""

import time
from dataclasses import dataclass, field

from research_agent.durable.synthesis import (
    classify_run_status,
)
from research_agent.evaluation.models import (
    CitedResearchReport,
    EvalProfile,
    EvalTask,
    SystemOutput,
    TrajectoryEvent,
    UsageMetrics,
)
from research_agent.evaluation.ports import (
    ResearchSystem,
    StructuredReportGenerator,
)
from research_agent.orchestration.models import (
    ResearchBudget,
    ResearchRun,
    WorkerEvidence,
    WorkerResult,
    WorkerRole,
)
from research_agent.orchestration.planner import (
    DeterministicPlanner,
)
from research_agent.orchestration.ports import ResearchWorker
from research_agent.orchestration.supervisor import (
    ResearchSupervisor,
)
from research_agent.orchestration.worker import (
    DeterministicWorkerDecider,
    LocalRetrievalTool,
    SpecialistWorker,
)
from research_agent.retrieval.lab import RetrievalLab
from research_agent.sources.ledger import (
    Citation,
    ClaimDraft,
    ClaimLedgerVerifier,
)


def _unique_evidence(
    results: tuple[WorkerResult, ...],
) -> tuple[WorkerEvidence, ...]:
    evidence_by_id: dict[str, WorkerEvidence] = {}

    for result in results:
        for evidence in result.evidence:
            evidence_by_id.setdefault(
                evidence.chunk_id,
                evidence,
            )

    return tuple(evidence_by_id.values())


def _exact_quote(text: str, limit: int = 240) -> str:
    """Return a bounded quotation that remains an exact substring."""

    return text[:limit].strip()


def _trajectory(
    run: ResearchRun,
) -> tuple[TrajectoryEvent, ...]:
    events: list[TrajectoryEvent] = []

    for event in run.trace:
        events.append(
            TrajectoryEvent(
                sequence=len(events) + 1,
                event=event.event,
                component="supervisor",
                elapsed_ms=event.elapsed_ms,
                detail=(
                    f"task={event.task_id}; "
                    f"active={event.active_workers}"
                ),
            )
        )

    for result in run.results:
        for event in result.trace:
            events.append(
                TrajectoryEvent(
                    sequence=len(events) + 1,
                    event=event.event,
                    component=result.role,
                    elapsed_ms=event.elapsed_ms,
                    detail=event.detail,
                )
            )

    return tuple(events)


def _deterministic_report(
    task: EvalTask,
    trial_id: str,
    results: tuple[WorkerResult, ...],
    evidence: tuple[WorkerEvidence, ...],
) -> CitedResearchReport:
    claims: list[ClaimDraft] = []

    for result in results:
        eligible = next(
            (
                item
                for item in result.evidence
                if item.eligible_for_synthesis
            ),
            None,
        )

        if (
            result.status != "completed"
            or not result.answer
            or eligible is None
        ):
            continue

        claims.append(
            ClaimDraft(
                claim_id=(
                    f"{trial_id}-claim-{len(claims) + 1}"
                ),
                text=result.answer,
                citations=(
                    Citation(
                        evidence_id=eligible.chunk_id,
                        quote=_exact_quote(
                            eligible.text
                        ),
                    ),
                ),
            )
        )

    claim_tuple = tuple(claims)
    ledger = ClaimLedgerVerifier().verify(
        claims=claim_tuple,
        evidence=evidence,
    )

    lines = [
        "# Deterministic research report",
        "",
        f"Question: {task.question}",
        "",
    ]

    for index, claim in enumerate(
        claim_tuple,
        start=1,
    ):
        evidence_ids = ", ".join(
            citation.evidence_id
            for citation in claim.citations
        )
        lines.extend(
            [
                f"{index}. {claim.text}",
                f"   Evidence: {evidence_ids}",
            ]
        )

    if not claim_tuple:
        lines.append(
            "No citation-eligible claims were produced."
        )

    return CitedResearchReport(
        markdown="\n".join(lines),
        claims=claim_tuple,
        ledger=ledger,
    )


@dataclass(slots=True)
class DeterministicResearchSystem:
    """Current multi-worker system under deterministic synthesis."""

    lab: RetrievalLab
    budget: ResearchBudget = field(
        default_factory=lambda: ResearchBudget(
            max_workers=3,
            max_concurrency=2,
        )
    )
    name: str = "multi-agent-deterministic"
    profile: EvalProfile = field(
        default="deterministic",
        init=False,
    )

    async def run(
        self,
        task: EvalTask,
        trial_id: str,
    ) -> SystemOutput:
        plan = DeterministicPlanner().create_plan(
            task.question
        )
        retrieval_tool = LocalRetrievalTool(self.lab)

        workers: dict[WorkerRole, ResearchWorker] = {
            assignment.role: SpecialistWorker(
                decider=DeterministicWorkerDecider(),
                search_tool=retrieval_tool,
            )
            for assignment in plan.assignments
        }

        run = await ResearchSupervisor(
            workers=workers,
            budget=self.budget,
        ).run(plan)

        evidence = _unique_evidence(run.results)
        report = _deterministic_report(
            task,
            trial_id,
            run.results,
            evidence,
        )

        return SystemOutput(
            system_name=self.name,
            profile=self.profile,
            status=classify_run_status(run.results),
            report=report,
            evidence=evidence,
            trajectory=_trajectory(run),
            usage=UsageMetrics(
                model_calls=0,
                tool_calls=sum(
                    result.usage.tool_calls
                    for result in run.results
                ),
                input_tokens=0,
                output_tokens=0,
                cached_input_tokens=0,
                nominal_cost_microusd=0,
            ),
            duration_ms=run.elapsed_ms,
        )


def _combine_optional(
    left: int | None,
    right: int | None,
) -> int | None:
    if left is None or right is None:
        return None
    return left + right


def _gemini_markdown(
    task: EvalTask,
    summary: str,
    claims: tuple[ClaimDraft, ...],
    rejected_count: int,
) -> str:
    lines = [
        "# Gemini cited research report",
        "",
        f"Question: {task.question}",
        "",
        summary,
        "",
        "## Verified claims",
    ]

    for index, claim in enumerate(claims, start=1):
        evidence_ids = ", ".join(
            citation.evidence_id
            for citation in claim.citations
        )
        lines.extend(
            [
                f"{index}. {claim.text}",
                f"   Evidence: {evidence_ids}",
            ]
        )

    if not claims:
        lines.append(
            "No model-proposed claims passed verification."
        )

    if rejected_count:
        lines.extend(
            [
                "",
                (
                    "Warning: "
                    f"{rejected_count} proposed claim(s) "
                    "were rejected by the runtime."
                ),
            ]
        )

    return "\n".join(lines)


@dataclass(slots=True)
class GeminiSynthesisSystem:
    """Same retrieval system with Gemini synthesis only."""

    base_system: ResearchSystem
    generator: StructuredReportGenerator
    name: str = "multi-agent-gemini-synthesis"
    profile: EvalProfile = field(
        default="gemini",
        init=False,
    )

    async def run(
        self,
        task: EvalTask,
        trial_id: str,
    ) -> SystemOutput:
        started_at = time.perf_counter()
        base = await self.base_system.run(
            task,
            trial_id,
        )

        eligible_evidence = tuple(
            evidence
            for evidence in base.evidence
            if evidence.eligible_for_synthesis
        )

        generated = await self.generator.generate(
            task.question,
            eligible_evidence,
        )

        proposed_claims = tuple(
            ClaimDraft(
                claim_id=(
                    f"{trial_id}-claim-{index}"
                ),
                text=claim.text,
                citations=claim.citations,
            )
            for index, claim in enumerate(
                generated.synthesis.claims,
                start=1,
            )
        )

        ledger = ClaimLedgerVerifier().verify(
            claims=proposed_claims,
            evidence=base.evidence,
        )
        assessments = {
            item.claim_id: item
            for item in ledger.assessments
        }

        accepted = tuple(
            claim
            for claim in proposed_claims
            if assessments[claim.claim_id].status
            == "citation_verified"
        )
        rejected = tuple(
            claim
            for claim in proposed_claims
            if assessments[claim.claim_id].status
            != "citation_verified"
        )

        report = CitedResearchReport(
            markdown=_gemini_markdown(
                task,
                generated.synthesis.summary,
                accepted,
                len(rejected),
            ),
            claims=accepted,
            rejected_claims=rejected,
            ledger=ledger,
        )

        usage = UsageMetrics(
            model_calls=(
                base.usage.model_calls
                + generated.usage.model_calls
            ),
            tool_calls=base.usage.tool_calls,
            input_tokens=_combine_optional(
                base.usage.input_tokens,
                generated.usage.input_tokens,
            ),
            output_tokens=_combine_optional(
                base.usage.output_tokens,
                generated.usage.output_tokens,
            ),
            cached_input_tokens=_combine_optional(
                base.usage.cached_input_tokens,
                generated.usage.cached_input_tokens,
            ),
            nominal_cost_microusd=_combine_optional(
                base.usage.nominal_cost_microusd,
                generated.usage.nominal_cost_microusd,
            ),
        )

        trajectory = (
            *base.trajectory,
            TrajectoryEvent(
                sequence=len(base.trajectory) + 1,
                event="model_synthesis_completed",
                component="gemini",
                elapsed_ms=(
                    time.perf_counter() - started_at
                )
                * 1_000,
                detail=(
                    f"accepted={len(accepted)}; "
                    f"rejected={len(rejected)}"
                ),
            ),
        )

        return SystemOutput(
            system_name=self.name,
            profile=self.profile,
            model_name=generated.model_name,
            status=base.status,
            report=report,
            evidence=base.evidence,
            trajectory=trajectory,
            usage=usage,
            duration_ms=(
                time.perf_counter() - started_at
            )
            * 1_000,
        )