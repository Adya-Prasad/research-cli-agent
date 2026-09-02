"""Supermemory implementation of the provider-neutral MemoryStore port."""
import hashlib
import math
from collections.abc import Mapping
from typing import Any, cast

from research_agent.memory.errors import MemoryProviderError
from research_agent.memory.models import (
    ApprovedMemory,
    MemoryHit,
    MemoryKind,
    MemoryWriteReceipt,
)
from research_agent.memory.policy import memory_id_belongs_to_user

_ALLOWED_KINDS: frozenset[str] = frozenset(
    {
        "user_preference",
        "durable_fact",
        "research_interest",
        "session_summary",
    }
)

def _container_tag(user_id: str) -> str:
    digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:24]
    return f"research_agent_user_{digest}"


def _read_field(
    value: Any,
    *names: str,
    default: Any = None,
) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]

        attribute = getattr(value, name, None)
        if attribute is not None:
            return attribute

    return default

def _extract_content(result: Any) -> tuple[str, bool]:
    """Return (text, is_source_chunk) from a v4 search result."""
    chunk = _read_field(result, "chunk", default=None)
    if isinstance(chunk, str) and chunk.strip():
        return chunk.strip(), True

    memory = _read_field(result, "memory", default=None)
    if isinstance(memory, str) and memory.strip():
        return memory.strip(), False

    raise MemoryProviderError(
        "Supermemory v4 result has no nonempty memory or chunk text"
    )


class SupermemoryStore:
    """Translate application memory contracts to the Supermemory SDK."""

    def __init__(self, client: Any) -> None:
        # Any is intentionally limited to this provider boundary because
        # generated SDK response classes may change independently.
        self._client = client

    @classmethod
    def from_api_key(
        cls,
        api_key: str,
        timeout_seconds: float = 20.0,
    ) -> "SupermemoryStore":
        if not api_key.strip():
            raise ValueError("Supermemory API key must not be empty")

        from supermemory import Supermemory

        client = Supermemory(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=2,
        )
        return cls(client)

    def add(self, memory: ApprovedMemory) -> MemoryWriteReceipt:
        try:
            response = self._client.add(
                content=memory.content,
                container_tag=_container_tag(memory.user_id),
                custom_id=memory.memory_id,
                metadata={
                    "memory_id": memory.memory_id,
                    "memory_kind": memory.kind,
                    "session_id": memory.session_id,
                },
                task_type="memory",
            )
        except Exception as exc:
            raise MemoryProviderError(
                f"Supermemory add failed: {exc}"
            ) from exc

        provider_id = _read_field(
            response,
            "id",
            default=memory.memory_id,
        )
        status = _read_field(response, "status", default="queued")

        return MemoryWriteReceipt(
            memory_id=memory.memory_id,
            provider_id=str(provider_id),
            status=str(status),
        )

    def search(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[MemoryHit]:
        normalized_user = user_id.strip()
        normalized_query = query.strip()

        if not normalized_user:
            raise ValueError("user_id must not be empty")
        if not normalized_query:
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        # Hybrid results may contain multiple representations of one memory.
        # Fetch a bounded candidate pool before deduplicating.
        candidate_limit = min(100, max(5, top_k * 2))

        try:
            response = self._client.search.memories(
                q=normalized_query,
                container_tag=_container_tag(normalized_user),
                search_mode="hybrid",
                threshold=0.0,
                rerank=False,
                rewrite_query=False,
                aggregate=False,
                include={"documents": True},
                limit=candidate_limit,
            )
        except Exception as exc:
            raise MemoryProviderError(
                "Supermemory search request failed"
            ) from exc

        results = _read_field(response, "results", default=None)
        if not isinstance(results, list):
            raise MemoryProviderError(
                "Supermemory search response has no results list"
            )

        # logical memory ID -> (selected hit, is_source_chunk)
        selected: dict[str, tuple[MemoryHit, bool]] = {}

        for result in results:
            metadata = _read_field(result, "metadata", default=None)

            if not isinstance(metadata, Mapping):
                raise MemoryProviderError(
                    "Supermemory result is missing application metadata"
                )

            memory_id = metadata.get("memory_id")
            kind_value = metadata.get("memory_kind")
            session_id = metadata.get("session_id")

            if (
                not isinstance(memory_id, str)
                or not memory_id.strip()
                or not isinstance(kind_value, str)
                or kind_value not in _ALLOWED_KINDS
                or not isinstance(session_id, str)
                or not session_id.strip()
            ):
                raise MemoryProviderError(
                    "Supermemory result has invalid application metadata"
                )

            # Defense against accidental scope/identity mapping mistakes.
            # This is an internal consistency guard, not authentication.
            if not memory_id_belongs_to_user(memory_id, normalized_user):
                raise MemoryProviderError(
                    "Supermemory result does not match the requested user scope"
                )

            content, is_source_chunk = _extract_content(result)

            raw_score = _read_field(result, "similarity", default=None)
            try:
                score = float(raw_score)
            except (TypeError, ValueError) as exc:
                raise MemoryProviderError(
                    "Supermemory result has an invalid similarity score"
                ) from exc

            if not math.isfinite(score) or score < 0:
                raise MemoryProviderError(
                    "Supermemory result has an invalid similarity score"
                )

            hit = MemoryHit(
                memory_id=memory_id,
                user_id=normalized_user,
                session_id=session_id,
                kind=cast(MemoryKind, kind_value),
                content=content,
                score=score,
                rank=1,  # Assigned after deduplication and sorting.
            )

            previous = selected.get(memory_id)

            if previous is None:
                selected[memory_id] = (hit, is_source_chunk)
                continue

            previous_hit, previous_is_source = previous

            # Source chunk beats extracted memory.
            # Within the same representation type, higher score wins.
            # Text provides a deterministic tie-breaker.
            priority = (is_source_chunk, hit.score, hit.content)
            previous_priority = (
                previous_is_source,
                previous_hit.score,
                previous_hit.content,
            )

            if priority > previous_priority:
                selected[memory_id] = (hit, is_source_chunk)

        ordered_hits = sorted(
            (hit for hit, _ in selected.values()),
            key=lambda hit: (-hit.score, hit.memory_id),
        )

        return [
            hit.model_copy(update={"rank": rank})
            for rank, hit in enumerate(ordered_hits[:top_k], start=1)
        ]

    def delete(self, user_id: str, memory_id: str) -> bool:
        normalized_user = user_id.strip()

        if not normalized_user:
            raise ValueError("user_id must not be empty")
        if not memory_id_belongs_to_user(memory_id, normalized_user):
            return False

        try:
            # Supermemory accepts either the provider ID or custom ID.
            self._client.documents.delete(memory_id)
        except Exception as exc:
            if exc.__class__.__name__ == "NotFoundError":
                return False
            raise MemoryProviderError(
                f"Supermemory deletion failed: {exc}"
            ) from exc

        return True