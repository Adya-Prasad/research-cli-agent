"""Strict bounded HTTP GET used by all source adapters."""

from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from research_agent.sources.errors import (
    SourcePolicyRejected,
    SourceRateLimited,
    SourceResponseTooLarge,
    SourceTimeout,
    SourceUnavailable,
)


@dataclass(frozen=True, slots=True)
class FetchedResponse:
    body: bytes
    content_type: str


async def bounded_get(
    client: httpx.AsyncClient,
    url: str,
    *,
    allowed_hosts: frozenset[str],
    accepted_content_types: frozenset[str],
    max_bytes: int = 1_000_000,
    params: Mapping[str, str | int] | None = None,
    headers: Mapping[str, str] | None = None,
) -> FetchedResponse:
    """Fetch one allowlisted HTTPS resource without following redirects."""
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")

    parsed = urlsplit(url)
    host = (parsed.hostname or "").casefold()

    if parsed.scheme != "https":
        raise SourcePolicyRejected("source URL must use HTTPS")
    if host not in {value.casefold() for value in allowed_hosts}:
        raise SourcePolicyRejected("source host is not allowlisted")

    request_headers = {
        "User-Agent": (
            "ml-research-cli-agent/0.1 "
            "(educational evidence client)"
        )
    }
    if headers:
        request_headers.update(headers)

    try:
        async with client.stream(
            "GET",
            url,
            params=params,
            headers=request_headers,
            follow_redirects=False,
        ) as response:
            if 300 <= response.status_code < 400:
                raise SourcePolicyRejected(
                    "source redirects are not automatically trusted"
                )
            if response.status_code == 429:
                raise SourceRateLimited("source rate limit exceeded")
            if response.status_code >= 400:
                raise SourceUnavailable(
                    f"source returned HTTP {response.status_code}"
                )

            content_type = (
                response.headers.get("content-type", "")
                .split(";", maxsplit=1)[0]
                .strip()
                .casefold()
            )
            accepted = {
                value.casefold()
                for value in accepted_content_types
            }
            if content_type not in accepted:
                raise SourcePolicyRejected(
                    "source returned a disallowed content type"
                )

            declared_length = response.headers.get(
                "content-length"
            )
            if declared_length is not None:
                try:
                    parsed_length = int(declared_length)
                except ValueError as exc:
                    raise SourcePolicyRejected(
                        "source returned invalid content-length"
                    ) from exc

                if parsed_length > max_bytes:
                    raise SourceResponseTooLarge(
                        "source response exceeds byte budget"
                    )

            collected = bytearray()
            async for block in response.aiter_bytes():
                collected.extend(block)
                if len(collected) > max_bytes:
                    raise SourceResponseTooLarge(
                        "source response exceeds byte budget"
                    )

            return FetchedResponse(
                body=bytes(collected),
                content_type=content_type,
            )

    except SourcePolicyRejected:
        raise
    except SourceRateLimited:
        raise
    except SourceResponseTooLarge:
        raise
    except SourceUnavailable:
        raise
    except httpx.TimeoutException as exc:
        raise SourceTimeout("source request timed out") from exc
    except httpx.RequestError as exc:
        raise SourceUnavailable(
            "source transport request failed"
        ) from exc