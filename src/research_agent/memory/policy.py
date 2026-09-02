"""Deterministic authority for durable memory writes."""

import hashlib
import re
from dataclasses import dataclass
from typing import cast

from research_agent.memory.errors import MemoryPolicyRejected
from research_agent.memory.models import ApprovedMemory, MemoryCandidate, MemoryKind

_ALLOWED_KINDS: frozenset[str] = frozenset(
    {
        "user_preference",
        "durable_fact",
        "research_interest",
        "session_summary",
    }
)

_SECRET_PATTERN = re.compile(
    r"(?i)(?:password|passcode|api[\s_-]?key|access[\s_-]?token|"
    r"secret|private[\s_-]?key)\s*(?:=|:|\bis\b)\s*\S{4,}"
    r"|(?:sk|sm)_[A-Za-z0-9_-]{12,}"
)


def _normalize_content(content: str) -> str:
    return " ".join(content.split())


def _owner_fragment(user_id: str) -> str:
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:12]


def memory_id_belongs_to_user(memory_id: str, user_id: str) -> bool:
    """Cheap internal ownership guard for logical memory IDs."""

    return memory_id.startswith(f"mem_{_owner_fragment(user_id)}_")


def _stable_memory_id(
    user_id: str,
    kind: MemoryKind,
    content: str,
) -> str:
    owner = _owner_fragment(user_id)
    identity = "\x1f".join((user_id, kind, content.casefold()))
    content_digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return f"mem_{owner}_{content_digest}"


@dataclass(frozen=True, slots=True)
class MemoryPolicy:
    """Allow only explicit, durable, non-secret memory categories."""

    max_content_chars: int = 2_000

    def __post_init__(self) -> None:
        if self.max_content_chars < 1:
            raise ValueError("max_content_chars must be positive")

    def approve(self, candidate: MemoryCandidate) -> ApprovedMemory:
        user_id = candidate.user_id.strip()
        session_id = candidate.session_id.strip()
        content = _normalize_content(candidate.content)

        if not user_id or not session_id:
            raise MemoryPolicyRejected("invalid_identity")
        if candidate.kind not in _ALLOWED_KINDS:
            raise MemoryPolicyRejected("unsupported_kind")
        if not content:
            raise MemoryPolicyRejected("empty_content")
        if len(content) > self.max_content_chars:
            raise MemoryPolicyRejected("content_too_long")
        if _SECRET_PATTERN.search(content):
            raise MemoryPolicyRejected("sensitive_content")

        kind = cast(MemoryKind, candidate.kind)

        return ApprovedMemory(
            memory_id=_stable_memory_id(user_id, kind, content),
            user_id=user_id,
            session_id=session_id,
            kind=kind,
            content=content,
        )