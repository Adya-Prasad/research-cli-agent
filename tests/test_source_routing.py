import pytest

from research_agent.orchestration.models import WorkerRole
from research_agent.sources.routing import RoleSourceRouter


class EmptySource:
    async def search(self, query: str, limit: int):
        del query, limit
        return ()


def test_router_builds_role_specific_federated_tool() -> None:
    source = EmptySource()
    routes = {
        "papers": (source,),
        "documentation_code": (source,),
        "skeptic": (source,),
    }

    router = RoleSourceRouter(routes)

    assert router.tool_for("papers").sources == (source,)
    assert router.tool_for("skeptic").sources == (source,)


def test_router_requires_every_worker_role() -> None:
    incomplete: dict[WorkerRole, tuple[EmptySource, ...]] = {
        "papers": (EmptySource(),),
        "skeptic": (EmptySource(),),
    }

    with pytest.raises(ValueError, match="missing source routes"):
        RoleSourceRouter(incomplete)