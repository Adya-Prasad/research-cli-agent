import json
from pathlib import Path
from typing import Literal

from typer.testing import CliRunner

from research_agent import cli as cli_module
from research_agent.cli import app
from research_agent.retrieval.lab import RetrievalMode
from research_agent.retrieval.models import Chunk, SearchHit
from research_agent.retrieval.ports import Retriever

runner = CliRunner()


class FakeRetriever:
    def __init__(
        self,
        chunk_id: str,
        retriever: Literal["dense", "bm25", "fused"],
    ) -> None:
        self._chunk_id = chunk_id
        self._retriever: Literal["dense", "bm25", "fused"] = retriever

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        del query

        return [
            SearchHit(
                chunk_id=self._chunk_id,
                rank=1,
                score=1.0,
                retriever=self._retriever,
            )
        ][:top_k]


class FakeLab:
    def __init__(self) -> None:
        self.chunk = Chunk(
            chunk_id="chunk-a",
            doc_id="document-a",
            source_path=Path("paper.md"),
            chunk_index=0,
            text="Semantic evidence from the corpus.",
            start_word=0,
            end_word=5,
        )

    def search(
        self,
        query: str,
        mode: RetrievalMode = "hybrid",
        top_k: int = 5,
    ) -> list[SearchHit]:
        del query, mode

        return [
            SearchHit(
                chunk_id=self.chunk.chunk_id,
                rank=1,
                score=1.0,
                retriever="fused",
            )
        ][:top_k]

    def chunk_for(self, chunk_id: str) -> Chunk:
        if chunk_id != self.chunk.chunk_id:
            raise KeyError(chunk_id)
        return self.chunk

    def retriever(self, mode: RetrievalMode) -> Retriever:
        if mode == "dense":
            return FakeRetriever(self.chunk.chunk_id, "dense")
        if mode == "bm25":
            return FakeRetriever(self.chunk.chunk_id, "bm25")
        return FakeRetriever(self.chunk.chunk_id, "fused")


def fake_build_retrieval_lab(
    corpus: Path,
    device: cli_module.DeviceMode,
) -> FakeLab:
    del corpus, device
    return FakeLab()


def test_demo_accepts_unquoted_words_and_prints_answer() -> None:
    result = runner.invoke(
        app,
        ["demo", "reliable", "agents", "preserve", "state"],
    )

    assert result.exit_code == 0
    assert "Your input contains 4 words" in result.output
    assert "model_decision" in result.output


def test_demo_rejects_blank_query() -> None:
    result = runner.invoke(app, ["demo", ""])

    assert result.exit_code != 0
    assert "must not be empty" in result.output


def test_retrieve_renders_resolved_chunk(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        cli_module,
        "build_retrieval_lab",
        fake_build_retrieval_lab,
    )

    result = runner.invoke(
        app,
        [
            "retrieve",
            "--corpus",
            str(tmp_path),
            "--device",
            "cpu",
            "semantic",
            "meaning",
        ],
    )

    assert result.exit_code == 0
    assert "paper.md" in result.output
    assert "Semantic evidence" in result.output


def test_evaluate_retrieval_renders_all_methods(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        cli_module,
        "build_retrieval_lab",
        fake_build_retrieval_lab,
    )

    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            [
                {
                    "query_id": "q1",
                    "query": "semantic evidence",
                    "relevant_chunk_ids": ["chunk-a"],
                }
            ]
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "evaluate-retrieval",
            "--corpus",
            str(tmp_path),
            "--cases",
            str(cases_path),
            "--top-k",
            "1",
            "--device",
            "cpu",
        ],
    )

    assert result.exit_code == 0
    assert "dense" in result.output
    assert "bm25" in result.output
    assert "hybrid" in result.output
    assert "1.000" in result.output