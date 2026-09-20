"""Controlled extraction from administrator-approved web pages."""

from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from research_agent.orchestration.models import WorkerEvidence
from research_agent.sources.errors import SourceFormatInvalid
from research_agent.sources.evidence import make_external_evidence
from research_agent.sources.http import bounded_get
from research_agent.sources.text import (
    best_excerpt,
    lexical_score,
)


class ControlledWebTarget(BaseModel):
    """One exact web page approved for evidence retrieval."""

    model_config = {"frozen": True}

    title: str = Field(min_length=1)
    url: str = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class ControlledWebAdapter:
    """Fetch one exact allowlisted page and extract an excerpt."""

    client: httpx.AsyncClient
    target: ControlledWebTarget
    allowed_hosts: frozenset[str]

    async def search(
        self,
        query: str,
        limit: int,
    ) -> tuple[WorkerEvidence, ...]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if limit < 1:
            raise ValueError("limit must be at least 1")

        fetched = await bounded_get(
            self.client,
            self.target.url,
            allowed_hosts=self.allowed_hosts,
            accepted_content_types=frozenset({"text/html"}),
            max_bytes=1_000_000,
        )

        soup = BeautifulSoup(
            fetched.body,
            "html.parser",
        )

        for element in soup(
            [
                "script",
                "style",
                "nav",
                "noscript",
                "form",
                "iframe",
                "footer",
                "aside",
            ]
        ):
            element.decompose()

        plain_text = " ".join(soup.stripped_strings)
        excerpt = best_excerpt(plain_text, query)

        if not excerpt:
            raise SourceFormatInvalid("controlled web page produced no text")

        host = urlsplit(self.target.url).hostname or "unknown"

        return (
            make_external_evidence(
                provider=f"controlled-web:{host.casefold()}",
                source_type="web_page",
                scope="web_excerpt",
                source_identifier=self.target.url,
                title=self.target.title,
                canonical_url=self.target.url,
                text=excerpt,
                rank=1,
                score=lexical_score(query, excerpt),
            ),
        )
