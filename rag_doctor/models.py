"""Core data models for rag-doctor."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class RootCause(str, Enum):
    RETRIEVAL_MISS = "retrieval_miss"
    CONTEXT_POSITION_BIAS = "context_position_bias"
    CHUNK_FRAGMENTATION = "chunk_fragmentation"
    HALLUCINATION = "hallucination"
    QUERY_MISMATCH = "query_mismatch"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Document:
    """A retrieved document chunk."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    position: int = 0

    def __repr__(self) -> str:
        preview = self.content[:60].replace("\n", " ")
        return f"Document(pos={self.position}, score={self.score:.3f}, content='{preview}...')"


@dataclass
class ToolResult:
    """Result from a single diagnostic tool."""
    tool_name: str
    passed: bool
    findings: Dict[str, Any] = field(default_factory=dict)
    recommendation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool_name,
            "passed": self.passed,
            "findings": self.findings,
            "recommendation": self.recommendation,
        }


@dataclass
class DiagnosisResult:
    """Full diagnosis output from the rag-doctor agent."""
    query: str
    answer: str
    expected: Optional[str]
    root_cause: RootCause
    severity: Severity
    evidence: Dict[str, Any]
    fix_suggestion: str
    fix_action: str
    config_patch: Dict[str, Any]
    tool_results: List[ToolResult] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "expected": self.expected,
            "root_cause": self.root_cause.value,
            "severity": self.severity.value,
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence,
            "fix": {
                "suggestion": self.fix_suggestion,
                "action": self.fix_action,
                "config_patch": self.config_patch,
            },
            "tool_results": [t.to_dict() for t in self.tool_results],
        }

    def summary(self) -> str:
        lines = [
            "",
            "─" * 50,
            "RAG-DOCTOR DIAGNOSIS REPORT",
            "─" * 50,
            f"Root Cause : {self.root_cause.value}",
            f"Severity   : {self.severity.value.upper()}",
            f"Confidence : {self.confidence:.0%}",
            "",
            "Evidence:",
        ]
        for k, v in self.evidence.items():
            lines.append(f"  {k}: {v}")
        lines += [
            "",
            "Fix:",
            f"  → {self.fix_suggestion}",
            f"  → {self.fix_action}",
        ]
        if self.config_patch:
            lines.append(f"  → Config patch: {self.config_patch}")
        lines.append("─" * 50)
        return "\n".join(lines)
