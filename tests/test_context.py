from research_agent.context import (
    ContextBudget,
    ContextComposer,
    ContextEvidence,
)
from research_agent.domain import Message
from research_agent.memory.models import MemoryHit


def _memory(
    memory_id: str,
    content: str,
    score: float = 0.9,
) -> MemoryHit:
    return MemoryHit(
        memory_id=memory_id,
        user_id="user-1",
        session_id="older-session",
        kind="research_interest",
        content=content,
        score=score,
        rank=1,
    )


def test_composer_labels_evidence_and_memory_separately() -> None:
    composer = ContextComposer()

    result = composer.compose(
        system_instructions="Answer using verifiable evidence.",
        user_query="How should an agent preserve state?",
        working_messages=[
            Message(
                role="assistant",
                content="We were discussing runtime boundaries.",
            )
        ],
        evidence=[
            ContextEvidence(
                evidence_id="chunk-1",
                source="runtime.md",
                text="A durable runtime persists explicit state.",
                rank=1,
            )
        ],
        memories=[
            _memory(
                "memory-1",
                "The user researches durable agent runtimes.",
            )
        ],
        budget=ContextBudget(
            total_tokens=120,
            evidence_tokens=35,
            working_tokens=30,
            memory_tokens=30,
        ),
    )

    assert "<retrieved_evidence" in result.text
    assert "<working_state>" in result.text
    assert "<long_term_memory" in result.text
    assert "chunk-1" in result.included_ids
    assert "memory-1" in result.included_ids
    assert result.total_tokens <= 120


def test_evidence_wins_when_global_budget_is_constrained() -> None:
    composer = ContextComposer()

    result = composer.compose(
        system_instructions="Use evidence.",
        user_query="Explain memory.",
        working_messages=[
            Message(
                role="assistant",
                content=(
                    "This older discussion contains many unnecessary "
                    "words that should not displace evidence."
                ),
            )
        ],
        evidence=[
            ContextEvidence(
                evidence_id="evidence-1",
                source="paper.md",
                text="Persistent state supports reliable recovery.",
                rank=1,
            )
        ],
        memories=[
            _memory(
                "memory-1",
                "This long personal memory should be dropped first.",
            )
        ],
        budget=ContextBudget(
            total_tokens=20,
            evidence_tokens=12,
            working_tokens=20,
            memory_tokens=20,
        ),
    )

    assert "evidence-1" in result.included_ids
    assert result.total_tokens <= 20
    assert result.dropped_ids


def test_duplicate_memory_is_removed_when_evidence_already_contains_it() -> None:
    composer = ContextComposer()
    duplicated_text = "Hybrid retrieval combines lexical and semantic signals."

    result = composer.compose(
        system_instructions="Use evidence.",
        user_query="What is hybrid retrieval?",
        working_messages=[],
        evidence=[
            ContextEvidence(
                evidence_id="evidence-1",
                source="retrieval.md",
                text=duplicated_text,
                rank=1,
            )
        ],
        memories=[
            _memory("memory-1", duplicated_text),
        ],
        budget=ContextBudget(
            total_tokens=100,
            evidence_tokens=40,
            working_tokens=20,
            memory_tokens=30,
        ),
    )

    assert result.text.count(duplicated_text) == 1
    assert "memory-1" in result.dropped_ids