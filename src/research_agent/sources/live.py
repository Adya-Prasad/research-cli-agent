"""Composition root for controlled live external research."""

from __future__ import annotations

from pathlib import Path
from typing import Self
from urllib.parse import urlsplit

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator,
    model_validator,
)

from research_agent.orchestration.models import (
    EvidenceSource,
    ResearchBudget,
    ResearchRun,
    WorkerRole,
)
from research_agent.orchestration.planner import (
    DeterministicPlanner,
)
from research_agent.orchestration.ports import ResearchWorker
from research_agent.orchestration.supervisor import (
    ResearchSupervisor,
)
from research_agent.orchestration.worker import (
    DeterministicWorkerDecider,
    SpecialistWorker,
)
from research_agent.sources.adapters import (
    ArxivPaperAdapter,
    GitHubCodeAdapter,
    GitHubTarget,
    OfficialDocumentationAdapter,
    OfficialDocumentationTarget,
)
from research_agent.sources.routing import RoleSourceRouter
from research_agent.sources.scholarly import (
    CrossrefPaperAdapter,
    SemanticScholarPaperAdapter,
)
from research_agent.sources.web import (
    ControlledWebAdapter,
    ControlledWebTarget,
)


def _https_host(url: str) -> str:
    parsed = urlsplit(url)
    host = parsed.hostname

    if parsed.scheme.casefold() != "https" or host is None:
        raise ValueError("live source target must use an absolute HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("live source target must not contain credentials")
    if parsed.fragment:
        raise ValueError("live source target must not contain a URL fragment")

    return host.casefold()


class LiveOfficialDocumentationTarget(OfficialDocumentationTarget):
    """Validated administrator-approved documentation target."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        _https_host(value)
        return value


class LiveGitHubTarget(GitHubTarget):
    """Exact repository file at an explicit ref."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )


class LiveControlledWebTarget(ControlledWebTarget):
    """Validated administrator-approved web target."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        _https_host(value)
        return value


class LiveSourceConfig(BaseModel):
    """Static source authority for one live research deployment."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    official_documentation: tuple[
        LiveOfficialDocumentationTarget,
        ...,
    ] = ()
    github: tuple[LiveGitHubTarget, ...] = ()
    controlled_web: tuple[
        LiveControlledWebTarget,
        ...,
    ] = ()

    @model_validator(mode="after")
    def require_engineering_source(self) -> Self:
        if not (self.official_documentation or self.github or self.controlled_web):
            raise ValueError(
                "at least one official-documentation, GitHub, or controlled-web target is required"
            )

        return self


def load_live_source_config(
    path: Path,
) -> LiveSourceConfig:
    """Load and validate an administrator-owned source allowlist."""

    serialized = path.read_text(encoding="utf-8")
    return LiveSourceConfig.model_validate_json(serialized)


def build_live_source_router(
    *,
    client: httpx.AsyncClient,
    config: LiveSourceConfig,
    crossref_mailto: str,
    semantic_scholar_api_key: str | None = None,
    github_token: str | None = None,
) -> RoleSourceRouter:
    """Construct role-specific source collections."""

    normalized_mailto = crossref_mailto.strip()
    if not normalized_mailto:
        raise ValueError("Crossref mailto must not be empty")

    semantic_key = semantic_scholar_api_key.strip() if semantic_scholar_api_key else None
    normalized_github_token = github_token.strip() if github_token else None

    paper_sources: tuple[EvidenceSource, ...] = (
        ArxivPaperAdapter(client=client),
        CrossrefPaperAdapter(
            client=client,
            mailto=normalized_mailto,
        ),
        SemanticScholarPaperAdapter(
            client=client,
            api_key=semantic_key,
        ),
    )

    engineering_sources: list[EvidenceSource] = []

    for target in config.official_documentation:
        engineering_sources.append(
            OfficialDocumentationAdapter(
                client=client,
                target=target,
                allowed_hosts=frozenset({_https_host(target.url)}),
            )
        )

    for target in config.github:
        engineering_sources.append(
            GitHubCodeAdapter(
                client=client,
                target=target,
                token=normalized_github_token,
            )
        )

    for target in config.controlled_web:
        engineering_sources.append(
            ControlledWebAdapter(
                client=client,
                target=target,
                allowed_hosts=frozenset({_https_host(target.url)}),
            )
        )

    engineering_source_tuple = tuple(engineering_sources)

    routes: dict[
        WorkerRole,
        tuple[EvidenceSource, ...],
    ] = {
        "papers": paper_sources,
        "documentation_code": engineering_source_tuple,
        "skeptic": (
            *paper_sources,
            *engineering_source_tuple,
        ),
    }

    return RoleSourceRouter(routes)


async def run_live_research(
    *,
    client: httpx.AsyncClient,
    prompt: str,
    config: LiveSourceConfig,
    crossref_mailto: str,
    semantic_scholar_api_key: str | None = None,
    github_token: str | None = None,
) -> ResearchRun:
    """Execute deterministic multi-agent external research."""

    normalized_prompt = " ".join(prompt.split())
    if not normalized_prompt:
        raise ValueError("prompt must not be empty")

    plan = DeterministicPlanner().create_plan(normalized_prompt)
    router = build_live_source_router(
        client=client,
        config=config,
        crossref_mailto=crossref_mailto,
        semantic_scholar_api_key=(semantic_scholar_api_key),
        github_token=github_token,
    )

    workers: dict[WorkerRole, ResearchWorker] = {
        assignment.role: SpecialistWorker(
            decider=DeterministicWorkerDecider(),
            search_tool=router.tool_for(assignment.role),
        )
        for assignment in plan.assignments
    }

    supervisor = ResearchSupervisor(
        workers=workers,
        budget=ResearchBudget(
            max_workers=3,
            max_concurrency=2,
        ),
    )

    return await supervisor.run(plan)
