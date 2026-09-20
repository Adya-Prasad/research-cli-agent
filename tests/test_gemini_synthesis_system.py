import asyncio
from dataclasses import dataclass

from research_agent.evaluation.models import (
    CitedResearchReport,
    EvalProfile,
    EvalTask,
    GeneratedClaim,
    GeneratedSynthesis,
    GenerationResult,
    SystemOutput,
    UsageMetrics,
)
from research_agent.evaluation.systems import (
    GeminiSynthesisSystem,
)
from research_agent.orchestration.models import WorkerEvidence
from research_agent.sources.ledger import (
    Citation,
    ClaimLedgerVerifier,
)


@dataclass
class StaticBaseSystem:
    name: str = "base"
    profile: EvalProfile = "deterministic"

    async def run(
        self,
        task: EvalTask,
        trial_id: str,
    ) -> SystemOutput:
        del task, trial_id

        evidence = WorkerEvidence(
            chunk_id="evidence-1",
            source="runtime.md",
            text=("A bounded runtime validates tool calls."),
            rank=1,
            score=1.0,
        )
        ledger = ClaimLedgerVerifier().verify(
            claims=(),
            evidence=(evidence,),
        )

        return SystemOutput(
            system_name=self.name,
            profile=self.profile,
            status="completed",
            report=CitedResearchReport(
                markdown="Base report",
                claims=(),
                ledger=ledger,
            ),
            evidence=(evidence,),
            trajectory=(),
            usage=UsageMetrics(tool_calls=3),
            duration_ms=5.0,
        )


@dataclass
class ScriptedGenerator:
    model_name: str = "scripted-gemini"

    async def generate(
        self,
        question: str,
        evidence: tuple[WorkerEvidence, ...],
    ) -> GenerationResult:
        del question, evidence

        return GenerationResult(
            synthesis=GeneratedSynthesis(
                summary="A bounded runtime is safer.",
                claims=(
                    GeneratedClaim(
                        text="The runtime validates tools.",
                        citations=(
                            Citation(
                                evidence_id="evidence-1",
                                quote="validates tool calls",
                            ),
                        ),
                    ),
                    GeneratedClaim(
                        text="The runtime is perfectly reliable.",
                        citations=(
                            Citation(
                                evidence_id="evidence-1",
                                quote="perfectly reliable",
                            ),
                        ),
                    ),
                ),
            ),
            usage=UsageMetrics(
                model_calls=1,
                input_tokens=100,
                output_tokens=30,
                cached_input_tokens=0,
                nominal_cost_microusd=105,
            ),
            model_name=self.model_name,
            duration_ms=4.0,
        )


def test_gemini_system_rejects_invalid_model_citation() -> None:
    system = GeminiSynthesisSystem(
        base_system=StaticBaseSystem(),
        generator=ScriptedGenerator(),
    )

    output = asyncio.run(
        system.run(
            EvalTask(
                task_id="runtime",
                question="How should runtimes validate tools?",
            ),
            "runtime-gemini-001",
        )
    )

    assert len(output.report.claims) == 1
    assert len(output.report.rejected_claims) == 1
    assert output.usage.model_calls == 1
    assert output.usage.tool_calls == 3
    assert output.model_name == "scripted-gemini"
