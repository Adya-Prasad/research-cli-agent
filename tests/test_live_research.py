import asyncio
import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from research_agent.sources.live import (
    LiveSourceConfig,
    load_live_source_config,
    run_live_research,
)

_ARXIV_RESPONSE = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2601.00001v1</id>
    <updated>2026-01-02T00:00:00Z</updated>
    <published>2026-01-01T00:00:00Z</published>
    <title>Reliable Multi-Agent Research Systems</title>
    <summary>
      Bounded concurrency and explicit state improve recovery.
    </summary>
    <author><name>Ada Researcher</name></author>
  </entry>
</feed>
"""


def _config_payload() -> dict[str, object]:
    return {
        "official_documentation": [
            {
                "title": "Structured concurrency documentation",
                "url": ("https://docs.example.org/structured-concurrency"),
            }
        ],
        "github": [
            {
                "owner": "python",
                "repository": "cpython",
                "path": "Lib/asyncio/taskgroups.py",
                "ref": "v3.13.5",
            }
        ],
        "controlled_web": [
            {
                "title": "Reliable agent engineering",
                "url": ("https://research.example.org/reliable-agents"),
            }
        ],
    }


def _handler(request: httpx.Request) -> httpx.Response:
    host = request.url.host

    if host == "export.arxiv.org":
        return httpx.Response(
            200,
            headers={"content-type": "application/atom+xml"},
            content=_ARXIV_RESPONSE,
        )

    if host == "api.crossref.org":
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "message": {
                    "items": [
                        {
                            "DOI": "10.1000/reliable-agents",
                            "title": ["Reliable Agent Runtime Design"],
                            "abstract": (
                                "<p>Checkpointing and bounded retries improve recovery.</p>"
                            ),
                            "author": [
                                {
                                    "given": "Grace",
                                    "family": "Engineer",
                                }
                            ],
                            "published": {"date-parts": [[2026, 1, 5]]},
                            "URL": ("https://doi.org/10.1000/reliable-agents"),
                        }
                    ]
                }
            },
        )

    if host == "api.semanticscholar.org":
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "total": 1,
                "offset": 0,
                "data": [
                    {
                        "paperId": "s2-reliable-agents",
                        "title": "Reliable Agent Runtime Design",
                        "abstract": ("We evaluate checkpointing and bounded retry policies."),
                        "authors": [{"name": "Grace Engineer"}],
                        "year": 2026,
                        "publicationDate": "2026-01-05",
                        "url": ("https://www.semanticscholar.org/paper/s2-reliable-agents"),
                        "externalIds": {"DOI": "10.1000/reliable-agents"},
                    }
                ],
            },
        )

    if host == "docs.example.org":
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=(
                b"<html><main>"
                b"<h1>Structured concurrency</h1>"
                b"<p>Task groups contain related failures and "
                b"propagate cancellation.</p>"
                b"</main></html>"
            ),
        )

    if host == "api.github.com":
        return httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            content=(b'class TaskGroup:\n    """Manage related asynchronous tasks."""\n'),
        )

    if host == "research.example.org":
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=(
                b"<html><main>"
                b"<h1>Reliable agent engineering</h1>"
                b"<p>Production agents require observable state, "
                b"bounded retries, and explicit failure handling.</p>"
                b"</main></html>"
            ),
        )

    raise AssertionError(f"Unexpected request host: {host}")


def test_live_config_requires_a_controlled_engineering_source() -> None:
    with pytest.raises(
        ValidationError,
        match="at least one",
    ):
        LiveSourceConfig()


def test_live_config_rejects_non_https_targets() -> None:
    with pytest.raises(
        ValidationError,
        match="HTTPS",
    ):
        LiveSourceConfig.model_validate(
            {
                "official_documentation": [
                    {
                        "title": "Unsafe documentation",
                        "url": "http://docs.example.org/agents",
                    }
                ]
            }
        )


def test_load_live_source_config_reads_valid_json(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "sources.json"
    config_path.write_text(
        json.dumps(_config_payload()),
        encoding="utf-8",
    )

    config = load_live_source_config(config_path)

    assert len(config.official_documentation) == 1
    assert len(config.github) == 1
    assert len(config.controlled_web) == 1


def test_live_research_routes_all_external_source_types() -> None:
    config = LiveSourceConfig.model_validate(_config_payload())

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(_handler)) as client:
            return await run_live_research(
                client=client,
                prompt=("How should production agents recover from partial failure?"),
                config=config,
                crossref_mailto="researcher@example.com",
                semantic_scholar_api_key="test-key",
                github_token="test-token",
            )

    result = asyncio.run(scenario())
    results_by_role = {worker_result.role: worker_result for worker_result in result.results}

    assert set(results_by_role) == {
        "papers",
        "documentation_code",
        "skeptic",
    }
    assert all(worker_result.status == "completed" for worker_result in result.results)

    paper_providers = {
        provider
        for evidence in results_by_role["papers"].evidence
        for provider in evidence.provider_chain
    }
    documentation_providers = {
        provider
        for evidence in results_by_role["documentation_code"].evidence
        for provider in evidence.provider_chain
    }

    assert {
        "arxiv",
        "crossref",
        "semantic-scholar",
    }.issubset(paper_providers)

    assert {
        "official-docs",
        "github",
        "controlled-web:research.example.org",
    }.issubset(documentation_providers)

    assert all(
        evidence.canonical_url
        for worker_result in result.results
        for evidence in worker_result.evidence
    )
    assert all(
        evidence.eligible_for_synthesis
        for worker_result in result.results
        for evidence in worker_result.evidence
    )
