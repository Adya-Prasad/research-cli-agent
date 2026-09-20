import asyncio
import os

import httpx
import pytest

from research_agent.sources.scholarly import (
    CrossrefPaperAdapter,
    SemanticScholarPaperAdapter,
)

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.getenv("RUN_LIVE_SOURCE_TESTS") != "1",
        reason="set RUN_LIVE_SOURCE_TESTS=1",
    ),
]


def test_live_crossref_returns_evidence() -> None:
    async def scenario() -> None:
        async with httpx.AsyncClient(timeout=15.0) as client:
            adapter = CrossrefPaperAdapter(
                client=client,
                mailto=os.getenv(
                    "CROSSREF_MAILTO",
                    "educational-agent@example.com",
                ),
            )
            results = await adapter.search(
                "retrieval augmented generation",
                limit=1,
            )

        assert results
        assert results[0].provider == "crossref"

    asyncio.run(scenario())


def test_live_semantic_scholar_returns_evidence() -> None:
    async def scenario() -> None:
        async with httpx.AsyncClient(timeout=15.0) as client:
            adapter = SemanticScholarPaperAdapter(
                client=client,
                api_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
            )
            results = await adapter.search(
                "language model agent evaluation",
                limit=1,
            )

        assert results
        assert results[0].provider == "semantic-scholar"

    asyncio.run(scenario())
