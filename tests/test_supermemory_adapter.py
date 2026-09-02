from types import SimpleNamespace
from typing import Any

from research_agent.memory.models import ApprovedMemory
from research_agent.memory.supermemory import SupermemoryStore


class FakeSearchEndpoint:
    def __init__(self, owner: "FakeSupermemoryClient") -> None:
        self.owner = owner

    def memories(self, **arguments: object) -> object:
        self.owner.search_arguments.append(arguments)
        return SimpleNamespace(results=self.owner.search_results)


class FakeDocumentsEndpoint:
    def __init__(self, owner: "FakeSupermemoryClient") -> None:
        self.owner = owner

    def delete(self, *, doc_id: str) -> None:
        self.owner.deleted_ids.append(doc_id)


class FakeSupermemoryClient:
    def __init__(self) -> None:
        self.add_arguments: list[dict[str, object]] = []
        self.search_arguments: list[dict[str, object]] = []
        self.deleted_ids: list[str] = []
        self.search_results: list[Any] = []

        self.search = FakeSearchEndpoint(self)
        self.documents = FakeDocumentsEndpoint(self)

    def add(self, **arguments: object) -> object:
        self.add_arguments.append(arguments)
        return SimpleNamespace(
            id="provider-document-1",
            status="queued",
        )


def _approved_memory() -> ApprovedMemory:
    return ApprovedMemory(
        memory_id="mem_2d6c8a6d5f0f_1234567890abcdef12345678",
        user_id="user-1",
        session_id="session-a",
        kind="research_interest",
        content="I research reliable agent runtimes.",
    )


def test_adapter_maps_approved_memory_to_provider_request() -> None:
    client = FakeSupermemoryClient()
    store = SupermemoryStore(client)

    receipt = store.add(_approved_memory())
    arguments = client.add_arguments[0]

    assert arguments["content"] == "I research reliable agent runtimes."
    assert arguments["custom_id"] == _approved_memory().memory_id
    assert arguments["task_type"] == "memory"
    assert arguments["container_tag"] != "user-1"
    assert arguments["metadata"] == {
        "memory_id": _approved_memory().memory_id,
        "memory_kind": "research_interest",
        "session_id": "session-a",
    }
    assert receipt.provider_id == "provider-document-1"
    assert receipt.status == "queued"


def test_adapter_converts_provider_results_to_typed_hits() -> None:
    client = FakeSupermemoryClient()
    client.search_results = [
        SimpleNamespace(
            id="provider-document-1",
            score=0.91,
            title=None,
            summary=None,
            metadata={
                "memory_id": _approved_memory().memory_id,
                "memory_kind": "research_interest",
                "session_id": "session-a",
            },
            chunks=[
                SimpleNamespace(
                    content="I research reliable agent runtimes.",
                    is_relevant=True,
                    score=0.91,
                )
            ],
        )
    ]
    store = SupermemoryStore(client)

    hits = store.search("user-1", "reliable agent runtime", top_k=3)

    assert len(hits) == 1
    assert hits[0].memory_id == _approved_memory().memory_id
    assert hits[0].kind == "research_interest"
    assert hits[0].rank == 1
    assert client.search_arguments[0]["search_mode"] == "documents"
    assert client.search_arguments[0]["limit"] == 3


def test_adapter_refuses_cross_user_deletion() -> None:
    client = FakeSupermemoryClient()
    store = SupermemoryStore(client)

    deleted = store.delete(
        user_id="different-user",
        memory_id=_approved_memory().memory_id,
    )

    assert deleted is False
    assert client.deleted_ids == []