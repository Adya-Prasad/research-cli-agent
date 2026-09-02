"""Supermemory implementation of the provider-neutral MemoryStore port."""

import hashlib
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


def _extract_content(result: Any) -> str:
    chunks = _read_field(result, "chunks", default=[]) or []
    contents: list[str] = []

    for chunk in chunks:
        content = _read_field(chunk, "content", default="")
        if isinstance(content, str) and content.strip():
            contents.append(content.strip())

    if contents:
        return "\n".join(contents)

    summary = _read_field(result, "summary", default="")
    return summary.strip() if isinstance(summary, str) else ""


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

        if not normalized_user:
            raise ValueError("user_id must not be empty")
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        try:
            response = self._client.search.memories(
                q=query,
                container_tag=_container_tag(normalized_user),
                search_mode="documents",
                limit=top_k,
            )
        except Exception as exc:
            raise MemoryProviderError(
                f"Supermemory search failed: {exc}"
            ) from exc

        results = _read_field(response, "results", default=[]) or []
        hits: list[MemoryHit] = []

        for result in results:
            metadata_value = _read_field(
                result,
                "metadata",
                default={},
            )
            metadata: Mapping[str, Any]

            if isinstance(metadata_value, Mapping):
                metadata = metadata_value
            else:
                metadata = {}

            memory_id = str(
                metadata.get("memory_id")
                or _read_field(
                    result,
                    "id",
                    "document_id",
                    "documentId",
                    default="",
                )
            )
            kind_value = str(
                metadata.get("memory_kind", "durable_fact")
            )
            session_id = str(
                metadata.get("session_id", "unknown-session")
            )
            content = _extract_content(result)

            if (
                not memory_id
                or not content
                or kind_value not in _ALLOWED_KINDS
            ):
                continue

            raw_score = _read_field(
                result,
                "score",
                "similarity",
                default=0.0,
            )

            hits.append(
                MemoryHit(
                    memory_id=memory_id,
                    user_id=normalized_user,
                    session_id=session_id,
                    kind=cast(MemoryKind, kind_value),
                    content=content,
                    score=max(0.0, float(raw_score)),
                    rank=len(hits) + 1,
                )
            )

            if len(hits) == top_k:
                break

        return hits

    def delete(self, user_id: str, memory_id: str) -> bool:
        normalized_user = user_id.strip()

        if not normalized_user:
            raise ValueError("user_id must not be empty")
        if not memory_id_belongs_to_user(memory_id, normalized_user):
            return False

        try:
            # Supermemory accepts either the provider ID or custom ID.
            self._client.documents.delete(doc_id=memory_id)
        except Exception as exc:
            if exc.__class__.__name__ == "NotFoundError":
                return False
            raise MemoryProviderError(
                f"Supermemory deletion failed: {exc}"
            ) from exc

        return True