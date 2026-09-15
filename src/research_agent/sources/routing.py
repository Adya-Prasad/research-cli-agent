"""Role-specific external source routing."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from research_agent.orchestration.models import EvidenceSource, WorkerRole
from research_agent.sources.federation import FederatedSearchTool
from research_agent.sources.security import EvidenceSecurityGate

_RESEARCH_ROLES: tuple[WorkerRole, ...] = (
    "papers",
    "documentation_code",
    "skeptic",
)


@dataclass(frozen=True, slots=True)
class RoleSourceRouter:
    """Grant each specialist only its approved source collection."""

    sources_by_role: Mapping[
        WorkerRole,
        tuple[EvidenceSource, ...],
    ]
    security_gate: EvidenceSecurityGate = field(
        default_factory=EvidenceSecurityGate
    )

    def __post_init__(self) -> None:
        missing = set(_RESEARCH_ROLES).difference(
            self.sources_by_role
        )

        if missing:
            raise ValueError(
                "missing source routes: "
                + ", ".join(sorted(missing))
            )

        empty = [
            role
            for role in _RESEARCH_ROLES
            if not self.sources_by_role[role]
        ]

        if empty:
            raise ValueError(
                "source routes must not be empty: "
                + ", ".join(empty)
            )

    def tool_for(
        self,
        role: WorkerRole,
    ) -> FederatedSearchTool:
        return FederatedSearchTool(
            sources=self.sources_by_role[role],
            security_gate=self.security_gate,
        )