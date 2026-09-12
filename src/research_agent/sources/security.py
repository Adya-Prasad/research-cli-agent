"""Deterministic screening of untrusted external evidence."""

import hashlib
import re
from dataclasses import dataclass

from research_agent.orchestration.models import WorkerEvidence

_RISK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(
            r"\bignore\s+(?:all\s+)?(?:previous|prior|system)"
            r"\s+instructions?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "role_override",
        re.compile(
            r"\byou\s+are\s+now\b|\bsystem\s+override\b",
            re.IGNORECASE,
        ),
    ),
    (
        "prompt_exfiltration",
        re.compile(
            r"\b(?:reveal|print|show|expose)\b.{0,40}"
            r"\b(?:system|developer)\s+prompt\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "secret_exfiltration",
        re.compile(
            r"\b(?:upload|send|exfiltrate|leak)\b.{0,50}"
            r"\b(?:secret|token|api[\s_-]?key|credential)s?\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "tool_instruction",
        re.compile(
            r"\b(?:call|invoke|execute|run)\b.{0,30}"
            r"\b(?:tool|command|shell|terminal)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class EvidenceSecurityGate:
    """Flag obvious attacks without pretending regex is complete security."""

    def scan(self, text: str) -> tuple[str, ...]:
        return tuple(
            flag
            for flag, pattern in _RISK_PATTERNS
            if pattern.search(text)
        )

    def screen(
        self,
        evidence: WorkerEvidence,
    ) -> WorkerEvidence:
        flags = self.scan(evidence.text)
        checksum = hashlib.sha256(
            evidence.text.encode("utf-8")
        ).hexdigest()

        return evidence.model_copy(
            update={
                "content_sha256": checksum,
                "risk_flags": flags,
                "eligible_for_synthesis": (
                    evidence.eligible_for_synthesis
                    and not flags
                ),
            }
        )