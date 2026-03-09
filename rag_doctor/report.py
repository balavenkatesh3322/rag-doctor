"""Diagnosis report structure and serialization."""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from .tools.base import ToolResult

ROOT_CAUSES = {
    "retrieval_miss":       "RC-1",
    "context_position_bias":"RC-2",
    "chunk_fragmentation":  "RC-3",
    "hallucination":        "RC-4",
    "query_mismatch":       "RC-5",
    "healthy":              "RC-0",
    "unknown":              "RC-?",
}

SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}


@dataclass
class DiagnosisReport:
    query: str
    answer: str
    expected: Optional[str]
    root_cause: str
    severity: str
    finding: str
    tool_results: List[ToolResult] = field(default_factory=list)
    fix_suggestion: Optional[str] = None
    config_patch: Dict[str, Any] = field(default_factory=dict)

    @property
    def root_cause_id(self) -> str:
        return ROOT_CAUSES.get(self.root_cause, "RC-?")

    @property
    def passed(self) -> bool:
        return self.root_cause == "healthy"

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "expected": self.expected,
            "root_cause": self.root_cause,
            "root_cause_id": self.root_cause_id,
            "severity": self.severity,
            "finding": self.finding,
            "fix_suggestion": self.fix_suggestion,
            "config_patch": self.config_patch,
            "tool_results": [t.to_dict() for t in self.tool_results],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_text(self) -> str:
        lines = [
            "=" * 60,
            "RAG-DOCTOR DIAGNOSIS REPORT",
            "=" * 60,
            f"Query      : {self.query}",
            f"Answer     : {self.answer[:120]}{'...' if len(self.answer)>120 else ''}",
        ]
        if self.expected:
            lines.append(f"Expected   : {self.expected[:120]}")
        lines += [
            "-" * 60,
            f"Root Cause : {self.root_cause} ({self.root_cause_id})",
            f"Severity   : {self.severity.upper()}",
            f"Finding    : {self.finding}",
            "",
            "Tool Results:",
        ]
        for tr in self.tool_results:
            status = "✓" if tr.passed else "✗"
            lines.append(f"  {status} [{tr.severity.upper():8}] {tr.tool_name}: {tr.finding}")

        if self.fix_suggestion:
            lines += ["", f"Fix: {self.fix_suggestion}"]
        if self.config_patch:
            lines += ["", "Config Patch:", json.dumps(self.config_patch, indent=2)]
        lines.append("=" * 60)
        return "\n".join(lines)
