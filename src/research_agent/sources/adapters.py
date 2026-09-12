"""Controlled adapters for papers, official docs, and source code."""

import hashlib
import re
import xml.etree.ElementTree as standard_etree
from dataclasses import dataclass
from urllib.parse import quote

import httpx
from bs4 import BeautifulSoup
from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException
from pydantic import BaseModel, Field

from research_agent.orchestration.models import (
    EvidenceSourceType,
    WorkerEvidence,
)
from research_agent.sources.errors import (
    SourceFormatInvalid,
)
from research_agent.sources.http import bounded_get
from research_agent.sources.text import (
    best_excerpt,
    lexical_score,
    normalize_inline,
)


def _evidence_id(
    provider: str,
    source_identifier: str,
    text: str,
) -> str:
    digest = hashlib.sha256(
        (
            f"{provider}\x1f{source_identifier}\x1f{text}"
        ).encode()
    ).hexdigest()[:24]
    return f"{provider}:{digest}"


def _external_evidence(
    *,
    provider: str,
    source_type: EvidenceSourceType,
    source_identifier: str,
    title: str,
    canonical_url: str,
    text: str,
    rank: int,
    score: float,
    published_at: str | None = None,
) -> WorkerEvidence:
    normalized_title = normalize_inline(title)
    normalized_text = text.strip()

    if not normalized_title:
        raise SourceFormatInvalid("source title is empty")
    if not normalized_text:
        raise SourceFormatInvalid("source text is empty")

    return WorkerEvidence(
        chunk_id=_evidence_id(
            provider,
            source_identifier,
            normalized_text,
        ),
        source=canonical_url,
        source_type=source_type,
        provider=provider,
        title=normalized_title,
        canonical_url=canonical_url,
        text=normalized_text,
        rank=rank,
        score=score,
        published_at=published_at,
    )


@dataclass(frozen=True, slots=True)
class ArxivPaperAdapter:
    """Search arXiv metadata and return abstract-level evidence."""

    client: httpx.AsyncClient
    endpoint: str = "https://export.arxiv.org/api/query"

    async def search(
        self,
        query: str,
        limit: int,
    ) -> tuple[WorkerEvidence, ...]:
        normalized_query = normalize_inline(query)
        if not normalized_query:
            raise ValueError("query must not be empty")
        if not 1 <= limit <= 10:
            raise ValueError("limit must be between 1 and 10")

        terms = re.findall(
            r"[A-Za-z0-9][A-Za-z0-9_-]*",
            normalized_query,
        )[:12]
        if not terms:
            raise ValueError(
                "query has no searchable arXiv terms"
            )

        search_expression = " AND ".join(
            f"all:{term}"
            for term in terms
        )
        fetched = await bounded_get(
            self.client,
            self.endpoint,
            allowed_hosts=frozenset(
                {"export.arxiv.org"}
            ),
            accepted_content_types=frozenset(
                {
                    "application/atom+xml",
                    "application/xml",
                    "text/xml",
                }
            ),
            max_bytes=1_000_000,
            params={
                "search_query": search_expression,
                "start": 0,
                "max_results": limit,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
        )

        try:
            root = ElementTree.fromstring(fetched.body)
        except (
            DefusedXmlException,
            standard_etree.ParseError,
        ) as exc:
            raise SourceFormatInvalid(
                "arXiv returned malformed Atom XML"
            ) from exc

        atom = "{http://www.w3.org/2005/Atom}"
        evidence: list[WorkerEvidence] = []

        for rank, entry in enumerate(
            root.findall(f"{atom}entry"),
            start=1,
        ):
            source_id = normalize_inline(
                entry.findtext(f"{atom}id") or ""
            )
            title = normalize_inline(
                entry.findtext(f"{atom}title") or ""
            )
            summary = normalize_inline(
                entry.findtext(f"{atom}summary") or ""
            )
            published = normalize_inline(
                entry.findtext(f"{atom}published") or ""
            )
            authors = [
                normalize_inline(
                    author.findtext(f"{atom}name") or ""
                )
                for author in entry.findall(
                    f"{atom}author"
                )
            ]
            authors = [
                author
                for author in authors
                if author
            ]

            if not source_id or not title or not summary:
                raise SourceFormatInvalid(
                    "arXiv entry is missing required metadata"
                )

            canonical_url = source_id.replace(
                "http://arxiv.org/",
                "https://arxiv.org/",
                1,
            )
            text = (
                f"{title}\n"
                f"Authors: {', '.join(authors)}\n"
                f"Abstract: {summary}"
            )

            evidence.append(
                _external_evidence(
                    provider="arxiv",
                    source_type="paper_abstract",
                    source_identifier=source_id,
                    title=title,
                    canonical_url=canonical_url,
                    text=text,
                    rank=rank,
                    score=1.0 / rank,
                    published_at=published or None,
                )
            )

        return tuple(evidence)


class OfficialDocumentationTarget(BaseModel):
    """One administrator-approved documentation page."""

    model_config = {"frozen": True}

    title: str = Field(min_length=1)
    url: str = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class OfficialDocumentationAdapter:
    client: httpx.AsyncClient
    target: OfficialDocumentationTarget
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
            accepted_content_types=frozenset(
                {"text/html"}
            ),
            max_bytes=1_000_000,
        )

        soup = BeautifulSoup(
            fetched.body,
            "html.parser",
        )
        for element in soup(
            ["script", "style", "nav", "noscript"]
        ):
            element.decompose()

        plain_text = " ".join(soup.stripped_strings)
        excerpt = best_excerpt(plain_text, query)
        if not excerpt:
            raise SourceFormatInvalid(
                "documentation page produced no text"
            )

        return (
            _external_evidence(
                provider="official-docs",
                source_type="official_documentation",
                source_identifier=self.target.url,
                title=self.target.title,
                canonical_url=self.target.url,
                text=excerpt,
                rank=1,
                score=lexical_score(query, excerpt),
            ),
        )


class GitHubTarget(BaseModel):
    """Exact public repository file and immutable release reference."""

    model_config = {"frozen": True}

    owner: str = Field(
        min_length=1,
        pattern=r"^[A-Za-z0-9_.-]+$",
    )
    repository: str = Field(
        min_length=1,
        pattern=r"^[A-Za-z0-9_.-]+$",
    )
    path: str = Field(min_length=1)
    ref: str = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class GitHubCodeAdapter:
    client: httpx.AsyncClient
    target: GitHubTarget
    token: str | None = None

    async def search(
        self,
        query: str,
        limit: int,
    ) -> tuple[WorkerEvidence, ...]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if limit < 1:
            raise ValueError("limit must be at least 1")

        encoded_path = quote(
            self.target.path.strip("/"),
            safe="/",
        )
        api_url = (
            "https://api.github.com/repos/"
            f"{self.target.owner}/"
            f"{self.target.repository}/contents/"
            f"{encoded_path}"
        )

        headers = {
            "Accept": "application/vnd.github.raw+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token and self.token.strip():
            headers["Authorization"] = (
                f"Bearer {self.token.strip()}"
            )

        fetched = await bounded_get(
            self.client,
            api_url,
            allowed_hosts=frozenset(
                {"api.github.com"}
            ),
            accepted_content_types=frozenset(
                {
                    "text/plain",
                    "application/octet-stream",
                    "application/vnd.github.raw+json",
                }
            ),
            max_bytes=1_000_000,
            params={"ref": self.target.ref},
            headers=headers,
        )

        try:
            code = fetched.body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SourceFormatInvalid(
                "GitHub source file is not UTF-8 text"
            ) from exc

        excerpt = best_excerpt(code, query)
        if not excerpt:
            raise SourceFormatInvalid(
                "GitHub source file produced no text"
            )

        canonical_url = (
            "https://github.com/"
            f"{self.target.owner}/"
            f"{self.target.repository}/blob/"
            f"{self.target.ref}/"
            f"{self.target.path.strip('/')}"
        )

        return (
            _external_evidence(
                provider="github",
                source_type="source_code",
                source_identifier=canonical_url,
                title=(
                    f"{self.target.owner}/"
                    f"{self.target.repository}:"
                    f"{self.target.path}"
                ),
                canonical_url=canonical_url,
                text=excerpt,
                rank=1,
                score=lexical_score(query, excerpt),
            ),
        )