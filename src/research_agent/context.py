"""Deterministic composition of bounded model context."""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel, Field

from research_agent.domain import Message
from research_agent.memory.models import MemoryHit


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class WhitespaceTokenCounter:
    """Deterministic approximation used until a model tokenizer is selected."""

    def count(self, text: str) -> int:
        return len(re.findall(r"\S+", text))


class ContextEvidence(BaseModel):
    model_config = {"frozen": True}

    evidence_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    text: str = Field(min_length=1)
    rank: int = Field(ge=1)


class ContextBudget(BaseModel):
    model_config = {"frozen": True}

    total_tokens: int = Field(gt=0)
    evidence_tokens: int = Field(ge=0)
    working_tokens: int = Field(ge=0)
    memory_tokens: int = Field(ge=0)


class ComposedContext(BaseModel):
    model_config = {"frozen": True}

    text: str
    total_tokens: int = Field(ge=0)
    included_ids: tuple[str, ...]
    dropped_ids: tuple[str, ...]


def _normalize_for_deduplication(text: str) -> str:
    return " ".join(text.casefold().split())


def _clean_attribute(text: str) -> str:
    return " ".join(text.replace('"', "'").split())


@dataclass(slots=True)
class ContextComposer:
    counter: TokenCounter = field(
        default_factory=WhitespaceTokenCounter
    )

    def compose(
        self,
        *,
        system_instructions: str,
        user_query: str,
        working_messages: Sequence[Message],
        evidence: Sequence[ContextEvidence],
        memories: Sequence[MemoryHit],
        budget: ContextBudget,
    ) -> ComposedContext:
        if not system_instructions.strip():
            raise ValueError("system_instructions must not be empty")
        if not user_query.strip():
            raise ValueError("user_query must not be empty")

        parts = [
            (
                "<system_instructions>\n"
                f"{system_instructions.strip()}\n"
                "</system_instructions>"
            ),
            (
                "<current_query>\n"
                f"{user_query.strip()}\n"
                "</current_query>"
            ),
        ]

        base_text = "\n\n".join(parts)
        if self.counter.count(base_text) > budget.total_tokens:
            raise ValueError(
                "system instructions and query exceed total context budget"
            )

        included_ids: list[str] = []
        dropped_ids: list[str] = []
        seen_text: set[str] = set()

        evidence_items = [
            (
                item.evidence_id,
                item.text,
                (
                    "[evidence "
                    f'id="{_clean_attribute(item.evidence_id)}" '
                    f'rank="{item.rank}" '
                    f'source="{_clean_attribute(item.source)}"]\n'
                    f"{item.text.strip()}"
                ),
            )
            for item in sorted(
                evidence,
                key=lambda value: (value.rank, value.evidence_id),
            )
        ]
        self._append_category(
            parts=parts,
            open_tag='<retrieved_evidence trust="untrusted-data">',
            close_tag="</retrieved_evidence>",
            items=evidence_items,
            category_limit=budget.evidence_tokens,
            total_limit=budget.total_tokens,
            seen_text=seen_text,
            included_ids=included_ids,
            dropped_ids=dropped_ids,
        )

        working_items = [
            (
                f"working:{index}",
                message.content,
                (
                    "[working "
                    f'index="{index}" '
                    f'role="{message.role}" '
                    f'name="{_clean_attribute(message.name or "")}"]\n'
                    f"{message.content.strip()}"
                ),
            )
            for index, message in reversed(
                list(enumerate(working_messages))
            )
            if message.content.strip()
        ]
        self._append_category(
            parts=parts,
            open_tag="<working_state>",
            close_tag="</working_state>",
            items=working_items,
            category_limit=budget.working_tokens,
            total_limit=budget.total_tokens,
            seen_text=seen_text,
            included_ids=included_ids,
            dropped_ids=dropped_ids,
        )

        memory_items = [
            (
                memory.memory_id,
                memory.content,
                (
                    "[memory "
                    f'id="{_clean_attribute(memory.memory_id)}" '
                    f'kind="{memory.kind}" '
                    f'score="{memory.score:.4f}"]\n'
                    f"{memory.content.strip()}"
                ),
            )
            for memory in sorted(
                memories,
                key=lambda value: (value.rank, value.memory_id),
            )
        ]
        self._append_category(
            parts=parts,
            open_tag='<long_term_memory trust="personal-context-not-evidence">',
            close_tag="</long_term_memory>",
            items=memory_items,
            category_limit=budget.memory_tokens,
            total_limit=budget.total_tokens,
            seen_text=seen_text,
            included_ids=included_ids,
            dropped_ids=dropped_ids,
        )

        text = "\n\n".join(parts)
        return ComposedContext(
            text=text,
            total_tokens=self.counter.count(text),
            included_ids=tuple(included_ids),
            dropped_ids=tuple(dropped_ids),
        )

    def _append_category(
        self,
        *,
        parts: list[str],
        open_tag: str,
        close_tag: str,
        items: Sequence[tuple[str, str, str]],
        category_limit: int,
        total_limit: int,
        seen_text: set[str],
        included_ids: list[str],
        dropped_ids: list[str],
    ) -> None:
        accepted_blocks: list[str] = []

        for item_id, raw_text, formatted_block in items:
            normalized = _normalize_for_deduplication(raw_text)

            if normalized in seen_text:
                dropped_ids.append(item_id)
                continue

            proposed_blocks = [*accepted_blocks, formatted_block]
            proposed_section = "\n".join(
                [open_tag, *proposed_blocks, close_tag]
            )
            proposed_full_text = "\n\n".join(
                [*parts, proposed_section]
            )

            category_fits = (
                self.counter.count(proposed_section)
                <= category_limit
            )
            total_fits = (
                self.counter.count(proposed_full_text)
                <= total_limit
            )

            if not category_fits or not total_fits:
                dropped_ids.append(item_id)
                continue

            accepted_blocks.append(formatted_block)
            included_ids.append(item_id)
            seen_text.add(normalized)

        if accepted_blocks:
            parts.append(
                "\n".join(
                    [open_tag, *accepted_blocks, close_tag]
                )
            )