"""Deterministic Day 4 decomposition of a research question."""

from research_agent.orchestration.models import (
    ResearchPlan,
    WorkerAssignment,
)


class DeterministicPlanner:
    """Create a fixed, bounded division of research responsibilities."""

    def create_plan(self, question: str) -> ResearchPlan:
        normalized = " ".join(question.split())
        if not normalized:
            raise ValueError("question must not be empty")

        return ResearchPlan(
            question=normalized,
            assignments=(
                WorkerAssignment(
                    task_id="papers",
                    role="papers",
                    question=(
                        "Find research evidence, empirical findings, "
                        f"benchmarks, and limitations for: {normalized}"
                    ),
                    expected_output=(
                        "Research evidence with traceable source passages."
                    ),
                ),
                WorkerAssignment(
                    task_id="documentation-code",
                    role="documentation_code",
                    question=(
                        "Find implementation guidance, system architecture, "
                        f"API constraints, and engineering trade-offs for: "
                        f"{normalized}"
                    ),
                    expected_output=(
                        "Implementation evidence with practical constraints."
                    ),
                ),
                WorkerAssignment(
                    task_id="skeptic",
                    role="skeptic",
                    question=(
                        "Find counterexamples, failure modes, unsupported "
                        f"assumptions, and missing evidence for: {normalized}"
                    ),
                    expected_output=(
                        "A skeptical assessment with unresolved questions."
                    ),
                ),
            ),
        )