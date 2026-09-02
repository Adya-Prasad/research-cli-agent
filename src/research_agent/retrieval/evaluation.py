import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import TypeAdapter

from research_agent.retrieval.models import LabeledQuery
from research_agent.retrieval.ports import Retriever


@dataclass(frozen=True, slots=True)
class RecallCaseResult:
    query_id: str
    query: str
    recall: float
    retrieved_chunk_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RecallReport:
    k: int
    mean_recall: float
    case_count: int
    cases: tuple[RecallCaseResult, ...]


def load_cases(path: Path) -> list[LabeledQuery]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return TypeAdapter(list[LabeledQuery]).validate_python(payload)


def evaluate_recall(
    retriever: Retriever,
    cases: Sequence[LabeledQuery],
    k: int,
) -> RecallReport:
    if not cases:
        raise ValueError("Evaluation requires at least one labeled query")
    if k < 1:
        raise ValueError("k must be at least 1")

    case_results: list[RecallCaseResult] = []

    for case in cases:
        hits = retriever.search(case.query, top_k=k)
        retrieved_chunk_ids = tuple(
            dict.fromkeys(hit.chunk_id for hit in hits)
        )
        relevant_chunk_ids = set(case.relevant_chunk_ids)

        recall = (
            len(set(retrieved_chunk_ids) & relevant_chunk_ids)
            / len(relevant_chunk_ids)
        )

        case_results.append(
            RecallCaseResult(
                query_id=case.query_id,
                query=case.query,
                recall=recall,
                retrieved_chunk_ids=retrieved_chunk_ids,
            )
        )

    return RecallReport(
        k=k,
        mean_recall=(
            sum(result.recall for result in case_results)
            / len(case_results)
        ),
        case_count=len(case_results),
        cases=tuple(case_results),
    )