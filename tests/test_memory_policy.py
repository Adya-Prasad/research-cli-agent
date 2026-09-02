import pytest

from research_agent.memory.errors import MemoryPolicyRejected
from research_agent.memory.models import MemoryCandidate
from research_agent.memory.policy import MemoryPolicy


def test_policy_approves_and_normalizes_allowed_memory() -> None:
    policy = MemoryPolicy()

    approved = policy.approve(
        MemoryCandidate(
            user_id="researcher-1",
            session_id="session-a",
            kind="research_interest",
            content="  I   research   reliable AI agents.  ",
        )
    )

    assert approved.user_id == "researcher-1"
    assert approved.session_id == "session-a"
    assert approved.kind == "research_interest"
    assert approved.content == "I research reliable AI agents."
    assert approved.memory_id.startswith("mem_")


def test_same_semantic_memory_has_same_id_across_sessions() -> None:
    policy = MemoryPolicy()

    first = policy.approve(
        MemoryCandidate(
            user_id="researcher-1",
            session_id="session-a",
            kind="user_preference",
            content="I prefer concise technical answers.",
        )
    )
    second = policy.approve(
        MemoryCandidate(
            user_id="researcher-1",
            session_id="session-b",
            kind="user_preference",
            content="I prefer concise technical answers.",
        )
    )

    assert first.memory_id == second.memory_id


def test_policy_rejects_secret_like_content() -> None:
    policy = MemoryPolicy()

    with pytest.raises(MemoryPolicyRejected) as error:
        policy.approve(
            MemoryCandidate(
                user_id="researcher-1",
                session_id="session-a",
                kind="durable_fact",
                content="SUPERMEMORY_API_KEY=sm_abcdefghijklmnopqrstuvwxyz",
            )
        )

    assert error.value.reason == "sensitive_content"


def test_policy_rejects_unsupported_kind() -> None:
    policy = MemoryPolicy()

    with pytest.raises(MemoryPolicyRejected) as error:
        policy.approve(
            MemoryCandidate(
                user_id="researcher-1",
                session_id="session-a",
                kind="raw_conversation",
                content="Save everything that happened.",
            )
        )

    assert error.value.reason == "unsupported_kind"