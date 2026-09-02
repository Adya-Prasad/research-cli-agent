from research_agent.memory.in_memory import InMemoryMemoryStore
from research_agent.memory.models import MemoryCandidate
from research_agent.memory.policy import MemoryPolicy
from research_agent.memory.service import MemoryService


def _service() -> MemoryService:
    return MemoryService(
        policy=MemoryPolicy(),
        store=InMemoryMemoryStore(),
    )


def test_memory_is_recalled_in_a_later_session() -> None:
    service = _service()

    service.remember(
        MemoryCandidate(
            user_id="user-1",
            session_id="session-a",
            kind="research_interest",
            content="I research distributed agent runtimes.",
        )
    )

    hits = service.recall(
        user_id="user-1",
        query="distributed agent runtime research",
    )

    assert len(hits) == 1
    assert hits[0].session_id == "session-a"
    assert hits[0].content == "I research distributed agent runtimes."


def test_memory_is_isolated_between_users() -> None:
    service = _service()

    service.remember(
        MemoryCandidate(
            user_id="user-1",
            session_id="session-a",
            kind="user_preference",
            content="I prefer concise explanations.",
        )
    )

    assert service.recall("user-2", "concise explanations") == []


def test_deleted_memory_is_not_recalled() -> None:
    service = _service()

    receipt = service.remember(
        MemoryCandidate(
            user_id="user-1",
            session_id="session-a",
            kind="durable_fact",
            content="My workstation has sixteen gigabytes of RAM.",
        )
    )

    assert service.forget("user-1", receipt.memory_id) is True
    assert service.recall("user-1", "workstation RAM") == []


def test_repeated_write_is_idempotent() -> None:
    service = _service()
    candidate = MemoryCandidate(
        user_id="user-1",
        session_id="session-a",
        kind="research_interest",
        content="I study retrieval evaluation.",
    )

    first = service.remember(candidate)
    second = service.remember(candidate)
    hits = service.recall("user-1", "retrieval evaluation")

    assert first.memory_id == second.memory_id
    assert len(hits) == 1