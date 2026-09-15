"""Provider-neutral evaluation boundaries."""

from typing import Protocol

from research_agent.evaluation.models import (
    EvalProfile,
    EvalTask,
    GeneratedSynthesis,
    GenerationResult,
    GradeResult,
    SystemOutput,
)
from research_agent.orchestration.models import WorkerEvidence


class ResearchSystem(Protocol):
    """A complete system that can attempt an evaluation task."""

    name: str
    profile: EvalProfile

    async def run(
        self,
        task: EvalTask,
        trial_id: str,
    ) -> SystemOutput: ...


class Grader(Protocol):
    """One independent judgment over a completed output."""

    name: str

    def grade(
        self,
        task: EvalTask,
        output: SystemOutput,
    ) -> GradeResult: ...


class StructuredReportGenerator(Protocol):
    """Synthesis model boundary used by the Gemini profile."""

    @property
    def model_name(self) -> str:
        """Stable provider model identifier."""
        ...

    async def generate(
        self,
        question: str,
        evidence: tuple[WorkerEvidence, ...],
    ) -> GenerationResult: ...

    
def validate_synthesis_contract(
    synthesis: GeneratedSynthesis,
) -> GeneratedSynthesis:
    """Explicit seam for future provider-neutral validation."""

    return synthesis