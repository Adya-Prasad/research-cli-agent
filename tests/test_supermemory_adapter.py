from types import SimpleNamespace
from typing import Any

import pytest
from supermemory.types.search_memories_response import SearchMemoriesResponse

from research_agent.memory.errors import MemoryProviderError
from research_agent.memory.models import ApprovedMemory, MemoryCandidate
from research_agent.memory.policy import MemoryPolicy
from research_agent.memory.supermemory import SupermemoryStore


class FakeSearchEndpoint:
    def __init__(self, owner: "FakeSupermemoryClient") -> None:
        self.owner = owner

    def memories(
        self,
        *,
        q: str,
        container_tag: str,
        search_mode: str,
        threshold: float,
        rerank: bool,
        rewrite_query: bool,
        aggregate: bool,
        include: dict[str, bool],
        limit: int,
    ) -> SearchMemoriesResponse:
        self.owner.search_arguments.append(
            {
                "q": q,
                "container_tag": container_tag,
                "search_mode": search_mode,
                "threshold": threshold,
                "rerank": rerank,
                "rewrite_query": rewrite_query,
                "aggregate": aggregate,
                "include": include,
                "limit": limit,
            }
        )

        # Validate fixtures against the installed SDK's v4 response schema.
        return SearchMemoriesResponse.model_validate(
            {
                "results": self.owner.search_results,
                "timing": 1.0,
                "total": len(self.owner.search_results),
            }
        )


class FakeDocumentsEndpoint:
    def __init__(self, owner: "FakeSupermemoryClient") -> None:
        self.owner = owner

    def delete(self, id: str) -> None:
        # Matches the installed SDK: delete(id), not delete(doc_id=...).
        self.owner.deleted_ids.append(id)


class FakeSupermemoryClient:
    def __init__(self) -> None:
        self.add_arguments: list[dict[str, object]] = []
        self.search_arguments: list[dict[str, object]] = []
        self.deleted_ids: list[str] = []
        self.search_results: list[dict[str, Any]] = []

        self.search = FakeSearchEndpoint(self)
        self.documents = FakeDocumentsEndpoint(self)

    def add(self, **arguments: object) -> object:
        self.add_arguments.append(arguments)
        return SimpleNamespace(
            id="provider-document-1",
            status="queued",
        )


def _approved_memory(
    content: str = "I research reliable agent runtimes.",
) -> ApprovedMemory:
    # Generate a valid application ID rather than inventing an owner hash.
    return MemoryPolicy().approve(
        MemoryCandidate(
            user_id="user-1",
            session_id="session-a",
            kind="research_interest",
            content=content,
        )
    )


def _provider_result(
    memory: ApprovedMemory,
    *,
    field: str,
    text: str,
    score: float = 0.91,
) -> dict[str, Any]:
    return {
        # This is a provider result ID, NOT our application memory ID.
        "id": f"provider-{field}-{memory.memory_id}",
        "updatedAt": "2026-09-02T00:00:00Z",
        "similarity": score,
        field: text,
        "metadata": {
            "memory_id": memory.memory_id,
            "memory_kind": memory.kind,
            "session_id": memory.session_id,
        },
    }


def test_adapter_maps_approved_memory_to_provider_request() -> None:
    client = FakeSupermemoryClient()
    memory = _approved_memory()

    receipt = SupermemoryStore(client).add(memory)

    arguments = client.add_arguments[0]
    assert arguments["content"] == memory.content
    assert arguments["custom_id"] == memory.memory_id
    assert arguments["task_type"] == "memory"
    assert arguments["container_tag"] != memory.user_id
    assert arguments["metadata"] == {
        "memory_id": memory.memory_id,
        "memory_kind": memory.kind,
        "session_id": memory.session_id,
    }
    assert receipt.memory_id == memory.memory_id
    assert receipt.provider_id == "provider-document-1"
    assert receipt.status == "queued"


@pytest.mark.parametrize("field", ["memory", "chunk"])
def test_adapter_reads_v4_text_and_similarity(field: str) -> None:
    client = FakeSupermemoryClient()
    memory = _approved_memory()
    client.search_results = [
        _provider_result(memory, field=field, text=memory.content)
    ]

    hits = SupermemoryStore(client).search(
        "user-1",
        "reliable agent runtime",
        top_k=3,
    )

    assert len(hits) == 1
    assert hits[0].memory_id == memory.memory_id
    assert hits[0].content == memory.content
    assert hits[0].kind == memory.kind
    assert hits[0].session_id == memory.session_id
    assert hits[0].score == pytest.approx(0.91)
    assert hits[0].rank == 1

    arguments = client.search_arguments[0]
    assert arguments["q"] == "reliable agent runtime"
    assert arguments["search_mode"] == "hybrid"
    assert arguments["threshold"] == 0.0
    assert arguments["aggregate"] is False
    assert arguments["rerank"] is False
    assert arguments["rewrite_query"] is False
    assert arguments["include"] == {"documents": True}
    assert arguments["limit"] == 6


@pytest.mark.parametrize("reverse", [False, True])
def test_adapter_deduplicates_and_prefers_source_chunk(reverse: bool) -> None:
    client = FakeSupermemoryClient()
    memory = _approved_memory()

    results = [
        _provider_result(
            memory,
            field="memory",
            text="The user studies dependable agent systems.",
            score=0.97,
        ),
        _provider_result(
            memory,
            field="chunk",
            text=memory.content,
            score=0.88,
        ),
    ]
    client.search_results = list(reversed(results)) if reverse else results

    hits = SupermemoryStore(client).search(
        "user-1",
        "agent systems",
        top_k=1,
    )

    assert len(hits) == 1
    assert hits[0].memory_id == memory.memory_id
    assert hits[0].content == memory.content
    # Keep the score belonging to the selected representation.
    assert hits[0].score == pytest.approx(0.88)
    assert hits[0].rank == 1


def test_adapter_ranks_unique_memories_and_applies_top_k() -> None:
    client = FakeSupermemoryClient()
    first = _approved_memory()
    second = _approved_memory("I study retrieval evaluation.")

    client.search_results = [
        _provider_result(first, field="chunk", text=first.content, score=0.7),
        _provider_result(second, field="chunk", text=second.content, score=0.9),
    ]
    store = SupermemoryStore(client)

    hits = store.search("user-1", "research", top_k=2)

    assert [hit.memory_id for hit in hits] == [
        second.memory_id,
        first.memory_id,
    ]
    assert [hit.rank for hit in hits] == [1, 2]
    assert len(store.search("user-1", "research", top_k=1)) == 1


def test_adapter_returns_empty_list_for_empty_provider_results() -> None:
    client = FakeSupermemoryClient()

    assert SupermemoryStore(client).search("user-1", "research") == []


def test_adapter_does_not_replace_missing_logical_id_with_provider_id() -> None:
    client = FakeSupermemoryClient()
    memory = _approved_memory()
    result = _provider_result(memory, field="chunk", text=memory.content)
    result["metadata"] = {}
    client.search_results = [result]

    with pytest.raises(MemoryProviderError, match="metadata"):
        SupermemoryStore(client).search("user-1", "research")


def test_adapter_rejects_cross_user_search_result() -> None:
    client = FakeSupermemoryClient()
    memory = _approved_memory()
    client.search_results = [
        _provider_result(memory, field="chunk", text=memory.content)
    ]

    with pytest.raises(MemoryProviderError, match="scope"):
        SupermemoryStore(client).search("different-user", "research")


def test_adapter_deletes_using_logical_custom_id() -> None:
    client = FakeSupermemoryClient()
    memory = _approved_memory()

    deleted = SupermemoryStore(client).delete(
        memory.user_id,
        memory.memory_id,
    )

    assert deleted is True
    assert client.deleted_ids == [memory.memory_id]


def test_adapter_refuses_cross_user_deletion() -> None:
    client = FakeSupermemoryClient()
    memory = _approved_memory()

    deleted = SupermemoryStore(client).delete(
        "different-user",
        memory.memory_id,
    )

    assert deleted is False
    assert client.deleted_ids == []