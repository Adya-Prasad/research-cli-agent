import json
from pathlib import Path

import pytest

from research_agent.retrieval.evaluation import evaluate_recall, load_cases
from research_agent.retrieval.models import LabeledQuery, SearchHit


class ScriptedRetriever:
    def __init__(self, hits_by_query: dict[str, list[str]]) -> None:
        self._hits_by_query = hits_by_query

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        chunk_ids = self._hits_by_query.get(query, [])
        return [
            SearchHit(
                chunk_id=chunk_id,
                rank=rank,
                score=1.0 / rank,
                retriever="dense",
            )
            for rank, chunk_id in enumerate(chunk_ids[:top_k], start=1)
        ]


def test_evaluate_recall_computes_per_case_and_mean_recall() -> None:
    cases = [
        LabeledQuery(
            query_id="q1",
            query="runtime query",
            relevant_chunk_ids=("runtime-a",),
        ),
        LabeledQuery(
            query_id="q2",
            query="memory query",
            relevant_chunk_ids=("memory-a", "memory-b"),
        ),
    ]
    retriever = ScriptedRetriever(
        {
            "runtime query": ["runtime-a", "unrelated"],
            "memory query": ["memory-a", "unrelated"],
        }
    )

    report = evaluate_recall(retriever, cases, k=2)

    assert report.case_count == 2
    assert report.cases[0].recall == 1.0
    assert report.cases[1].recall == 0.5
    assert report.mean_recall == 0.75


def test_load_cases_validates_json_contract(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            [
                {
                    "query_id": "q1",
                    "query": "What is memory?",
                    "relevant_chunk_ids": ["memory-a"],
                }
            ]
        ),
        encoding="utf-8",
    )

    cases = load_cases(path)

    assert cases[0].query_id == "q1"
    assert cases[0].relevant_chunk_ids == ("memory-a",)


def test_evaluate_recall_rejects_invalid_inputs() -> None:
    retriever = ScriptedRetriever({})

    with pytest.raises(ValueError, match="at least one labeled query"):
        evaluate_recall(retriever, [], k=1)

    case = LabeledQuery(
        query_id="q1",
        query="query",
        relevant_chunk_ids=("chunk-a",),
    )
    with pytest.raises(ValueError, match="k must be at least 1"):
        evaluate_recall(retriever, [case], k=0)