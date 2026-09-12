import asyncio

import httpx
import pytest

from research_agent.sources.errors import (
    SourcePolicyRejected,
    SourceResponseTooLarge,
)
from research_agent.sources.http import bounded_get

_ALLOWED = frozenset({"docs.example.org"})
_TYPES = frozenset({"text/html"})


def test_bounded_get_rejects_non_allowlisted_host() -> None:
    async def scenario() -> None:
        async with httpx.AsyncClient() as client:
            with pytest.raises(
                SourcePolicyRejected,
                match="allowlisted",
            ):
                await bounded_get(
                    client,
                    "https://attacker.example/page",
                    allowed_hosts=_ALLOWED,
                    accepted_content_types=_TYPES,
                )

    asyncio.run(scenario())


def test_bounded_get_does_not_follow_redirects() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302,
            headers={
                "location": "https://attacker.example/secret"
            },
        )

    async def scenario() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport
        ) as client:
            with pytest.raises(
                SourcePolicyRejected,
                match="redirect",
            ):
                await bounded_get(
                    client,
                    "https://docs.example.org/page",
                    allowed_hosts=_ALLOWED,
                    accepted_content_types=_TYPES,
                )

    asyncio.run(scenario())


def test_bounded_get_rejects_wrong_content_type() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/zip"},
            content=b"archive",
        )

    async def scenario() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport
        ) as client:
            with pytest.raises(
                SourcePolicyRejected,
                match="content type",
            ):
                await bounded_get(
                    client,
                    "https://docs.example.org/page",
                    allowed_hosts=_ALLOWED,
                    accepted_content_types=_TYPES,
                )

    asyncio.run(scenario())


def test_bounded_get_enforces_streamed_byte_limit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=b"x" * 101,
        )

    async def scenario() -> None:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            transport=transport
        ) as client:
            with pytest.raises(SourceResponseTooLarge):
                await bounded_get(
                    client,
                    "https://docs.example.org/page",
                    allowed_hosts=_ALLOWED,
                    accepted_content_types=_TYPES,
                    max_bytes=100,
                )

    asyncio.run(scenario())