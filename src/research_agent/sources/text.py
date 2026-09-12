"""Small deterministic text-ranking helpers."""

import re


def normalize_inline(text: str) -> str:
    return " ".join(text.split())


def query_terms(query: str) -> frozenset[str]:
    return frozenset(
        match.casefold()
        for match in re.findall(
            r"[A-Za-z0-9][A-Za-z0-9_-]*",
            query,
        )
        if len(match) > 1
    )


def lexical_score(query: str, text: str) -> float:
    wanted = query_terms(query)
    if not wanted:
        return 0.0

    available = query_terms(text)
    return len(wanted & available) / len(wanted)


def best_excerpt(
    text: str,
    query: str,
    *,
    window_words: int = 180,
) -> str:
    words = text.split()
    if not words:
        return ""

    if len(words) <= window_words:
        return " ".join(words)

    wanted = query_terms(query)
    best_start = 0
    best_overlap = -1

    for start in range(
        0,
        len(words),
        max(1, window_words // 2),
    ):
        window = words[start : start + window_words]
        overlap = len(
            wanted
            & {word.casefold().strip(".,:;()[]{}") for word in window}
        )
        if overlap > best_overlap:
            best_start = start
            best_overlap = overlap

    return " ".join(
        words[best_start : best_start + window_words]
    )