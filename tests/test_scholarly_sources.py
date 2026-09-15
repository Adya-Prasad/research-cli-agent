import asyncio

import httpx

from research_agent.sources.scholarly import (
    CrossrefPaperAdapter,
    SemanticScholarPaperAdapter,
)


def test_crossref_adapter_parses_metadata() -> None:
    captured_mailto = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_mailto
        captured_mailto = request.url.params["mailto"]

        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "message": {
                    "items": [
                        {
                            "DOI": "10.1000/reliable-agents",
                            "title": ["Reliable Agent Runtimes"],
                            "abstract": (
                                "<jats:p>Bounded runtimes improve "
                                "failure containment.</jats:p>"
                            ),
                            "author": [
                                {
                                    "given": "Ada",
                                    "family": "Researcher",
                                }
                            ],
                            "published": {
                                "date-parts": [[2026, 2, 3]]
                            },
                            "URL": (
                                "https://doi.org/"
                                "10.1000/reliable-agents"
                            ),
                        }
                    ]
                }
            },
        )

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            adapter = CrossrefPaperAdapter(
                client=client,
                mailto="researcher@example.com",
            )
            evidence = await adapter.search(
                "reliable agent runtimes",
                limit=3,
            )

        assert captured_mailto == "researcher@example.com"
        assert len(evidence) == 1

        item = evidence[0]
        assert item.provider == "crossref"
        assert item.scope == "abstract"
        assert item.identifiers.doi == (
            "10.1000/reliable-agents"
        )
        assert item.published_at == "2026-02-03"
        assert "Bounded runtimes" in item.text
        assert "<jats:p>" not in item.text

    asyncio.run(scenario())


def test_semantic_scholar_adapter_parses_abstract() -> None:
    captured_key = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_key
        captured_key = request.headers["x-api-key"]

        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "total": 1,
                "offset": 0,
                "data": [
                    {
                        "paperId": "s2-paper-123",
                        "title": "Reliable Agent Runtimes",
                        "abstract": (
                            "We evaluate retry and checkpoint policies."
                        ),
                        "authors": [
                            {"name": "Ada Researcher"}
                        ],
                        "year": 2026,
                        "publicationDate": "2026-02-03",
                        "url": (
                            "https://www.semanticscholar.org/"
                            "paper/s2-paper-123"
                        ),
                        "externalIds": {
                            "DOI": "10.1000/reliable-agents",
                            "ArXiv": "2602.01234",
                        },
                    }
                ],
            },
        )

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            adapter = SemanticScholarPaperAdapter(
                client=client,
                api_key="test-key",
            )
            evidence = await adapter.search(
                "reliable agent runtimes",
                limit=2,
            )

        assert captured_key == "test-key"
        assert len(evidence) == 1

        item = evidence[0]
        assert item.provider == "semantic-scholar"
        assert item.scope == "abstract"
        assert item.identifiers.semantic_scholar_id == (
            "s2-paper-123"
        )
        assert item.identifiers.doi == (
            "10.1000/reliable-agents"
        )
        assert item.identifiers.arxiv_id == "2602.01234"
        assert "checkpoint policies" in item.text

    asyncio.run(scenario())