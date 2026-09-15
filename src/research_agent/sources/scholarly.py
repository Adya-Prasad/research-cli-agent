"""Bounded scholarly metadata adapters."""

from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from research_agent.orchestration.models import (
    ScholarlyIdentifiers,
    WorkerEvidence,
)
from research_agent.sources.errors import SourceFormatInvalid
from research_agent.sources.evidence import make_external_evidence
from research_agent.sources.http import bounded_get
from research_agent.sources.text import normalize_inline


class _CrossrefAuthor(BaseModel):
    model_config = ConfigDict(extra="ignore")

    given: str = ""
    family: str = ""


class _CrossrefDate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    date_parts: list[list[int]] = Field(
        default_factory=list,
        alias="date-parts",
    )


class _CrossrefWork(BaseModel):
    model_config = ConfigDict(extra="ignore")

    doi: str = Field(alias="DOI")
    title: list[str] = Field(default_factory=list)
    abstract: str | None = None
    author: list[_CrossrefAuthor] = Field(default_factory=list)
    published: _CrossrefDate | None = None
    published_online: _CrossrefDate | None = Field(
        default=None,
        alias="published-online",
    )
    published_print: _CrossrefDate | None = Field(
        default=None,
        alias="published-print",
    )
    url: str | None = Field(default=None, alias="URL")


class _CrossrefMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: list[_CrossrefWork] = Field(default_factory=list)


class _CrossrefEnvelope(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message: _CrossrefMessage


class _SemanticScholarAuthor(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = ""


class _SemanticScholarExternalIds(BaseModel):
    model_config = ConfigDict(extra="ignore")

    doi: str | None = Field(default=None, alias="DOI")
    arxiv: str | None = Field(default=None, alias="ArXiv")


class _SemanticScholarPaper(BaseModel):
    model_config = ConfigDict(extra="ignore")

    paper_id: str = Field(alias="paperId")
    title: str
    abstract: str | None = None
    authors: list[_SemanticScholarAuthor] = Field(
        default_factory=list
    )
    year: int | None = None
    publication_date: str | None = Field(
        default=None,
        alias="publicationDate",
    )
    url: str | None = None
    external_ids: _SemanticScholarExternalIds | None = Field(
        default=None,
        alias="externalIds",
    )


class _SemanticScholarEnvelope(BaseModel):
    model_config = ConfigDict(extra="ignore")

    data: list[_SemanticScholarPaper] = Field(
        default_factory=list
    )


def _strip_markup(value: str) -> str:
    return normalize_inline(
        BeautifulSoup(value, "html.parser").get_text(
            " ",
            strip=True,
        )
    )


def _crossref_date(work: _CrossrefWork) -> str | None:
    candidates = (
        work.published,
        work.published_online,
        work.published_print,
    )

    for candidate in candidates:
        if candidate is None or not candidate.date_parts:
            continue

        parts = candidate.date_parts[0]
        if not parts:
            continue

        year = parts[0]
        month = parts[1] if len(parts) >= 2 else 1
        day = parts[2] if len(parts) >= 3 else 1
        return f"{year:04d}-{month:02d}-{day:02d}"

    return None


@dataclass(frozen=True, slots=True)
class CrossrefPaperAdapter:
    """Retrieve metadata and available abstracts from Crossref."""

    client: httpx.AsyncClient
    mailto: str
    endpoint: str = "https://api.crossref.org/works"

    async def search(
        self,
        query: str,
        limit: int,
    ) -> tuple[WorkerEvidence, ...]:
        normalized_query = normalize_inline(query)

        if not normalized_query:
            raise ValueError("query must not be empty")
        if not 1 <= limit <= 20:
            raise ValueError("limit must be between 1 and 20")
        if not self.mailto.strip():
            raise ValueError("Crossref mailto must not be empty")

        fetched = await bounded_get(
            self.client,
            self.endpoint,
            allowed_hosts=frozenset({"api.crossref.org"}),
            accepted_content_types=frozenset(
                {
                    "application/json",
                    "application/vnd.api+json",
                }
            ),
            max_bytes=1_000_000,
            params={
                "query.bibliographic": normalized_query,
                "rows": limit,
                "mailto": self.mailto.strip(),
                "select": (
                    "DOI,title,abstract,author,published,"
                    "published-online,published-print,URL"
                ),
            },
            headers={"Accept": "application/json"},
        )

        try:
            envelope = _CrossrefEnvelope.model_validate_json(
                fetched.body
            )
        except ValidationError as exc:
            raise SourceFormatInvalid(
                "Crossref returned malformed JSON metadata"
            ) from exc

        evidence: list[WorkerEvidence] = []

        for rank, work in enumerate(
            envelope.message.items,
            start=1,
        ):
            if not work.title:
                continue

            title = normalize_inline(work.title[0])
            if not title:
                continue

            authors = [
                normalize_inline(
                    f"{author.given} {author.family}"
                )
                for author in work.author
            ]
            authors = [author for author in authors if author]

            abstract = (
                _strip_markup(work.abstract)
                if work.abstract
                else ""
            )
            doi = normalize_inline(work.doi)
            canonical_url = f"https://doi.org/{doi}"

            lines = [
                title,
                f"Authors: {', '.join(authors) or 'Unknown'}",
                f"DOI: {doi}",
            ]
            if abstract:
                lines.append(f"Abstract: {abstract}")

            evidence.append(
                make_external_evidence(
                    provider="crossref",
                    source_type="paper_abstract",
                    scope=(
                        "abstract"
                        if abstract
                        else "metadata"
                    ),
                    source_identifier=doi.casefold(),
                    title=title,
                    canonical_url=canonical_url,
                    text="\n".join(lines),
                    rank=rank,
                    score=1.0 / rank,
                    published_at=_crossref_date(work),
                    identifiers=ScholarlyIdentifiers(
                        doi=doi
                    ),
                )
            )

        return tuple(evidence)


@dataclass(frozen=True, slots=True)
class SemanticScholarPaperAdapter:
    """Retrieve paper metadata from Semantic Scholar Graph API."""

    client: httpx.AsyncClient
    api_key: str | None = None
    endpoint: str = (
        "https://api.semanticscholar.org/"
        "graph/v1/paper/search"
    )

    async def search(
        self,
        query: str,
        limit: int,
    ) -> tuple[WorkerEvidence, ...]:
        normalized_query = normalize_inline(query)

        if not normalized_query:
            raise ValueError("query must not be empty")
        if not 1 <= limit <= 20:
            raise ValueError("limit must be between 1 and 20")

        headers = {"Accept": "application/json"}
        if self.api_key and self.api_key.strip():
            headers["x-api-key"] = self.api_key.strip()

        fetched = await bounded_get(
            self.client,
            self.endpoint,
            allowed_hosts=frozenset(
                {"api.semanticscholar.org"}
            ),
            accepted_content_types=frozenset(
                {"application/json"}
            ),
            max_bytes=1_000_000,
            params={
                "query": normalized_query,
                "limit": limit,
                "fields": (
                    "paperId,title,abstract,authors,year,"
                    "publicationDate,url,externalIds"
                ),
            },
            headers=headers,
        )

        try:
            envelope = (
                _SemanticScholarEnvelope.model_validate_json(
                    fetched.body
                )
            )
        except ValidationError as exc:
            raise SourceFormatInvalid(
                "Semantic Scholar returned malformed JSON metadata"
            ) from exc

        evidence: list[WorkerEvidence] = []

        for rank, paper in enumerate(
            envelope.data,
            start=1,
        ):
            title = normalize_inline(paper.title)
            if not title:
                continue

            abstract = normalize_inline(paper.abstract or "")
            authors = [
                normalize_inline(author.name)
                for author in paper.authors
                if normalize_inline(author.name)
            ]

            external_ids = paper.external_ids
            doi = external_ids.doi if external_ids else None
            arxiv_id = (
                external_ids.arxiv
                if external_ids
                else None
            )
            canonical_url = (
                paper.url
                or (
                    "https://www.semanticscholar.org/"
                    f"paper/{paper.paper_id}"
                )
            )

            lines = [
                title,
                f"Authors: {', '.join(authors) or 'Unknown'}",
            ]
            if paper.year is not None:
                lines.append(f"Year: {paper.year}")
            if abstract:
                lines.append(f"Abstract: {abstract}")

            evidence.append(
                make_external_evidence(
                    provider="semantic-scholar",
                    source_type="paper_abstract",
                    scope=(
                        "abstract"
                        if abstract
                        else "metadata"
                    ),
                    source_identifier=paper.paper_id,
                    title=title,
                    canonical_url=canonical_url,
                    text="\n".join(lines),
                    rank=rank,
                    score=1.0 / rank,
                    published_at=(
                        paper.publication_date
                        or (
                            str(paper.year)
                            if paper.year is not None
                            else None
                        )
                    ),
                    identifiers=ScholarlyIdentifiers(
                        doi=doi,
                        arxiv_id=arxiv_id,
                        semantic_scholar_id=paper.paper_id,
                    ),
                )
            )

        return tuple(evidence)