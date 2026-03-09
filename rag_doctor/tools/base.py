"""Base class for all diagnostic tools."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod


@dataclass
class ToolResult:
    tool_name: str
    passed: bool
    severity: str = "low"   # low | medium | high | critical
    finding: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    recommendation: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "tool": self.tool_name,
            "passed": self.passed,
            "severity": self.severity,
            "finding": self.finding,
            "details": self.details,
            "recommendation": self.recommendation,
        }


class BaseTool(ABC):
    """All diagnostic tools inherit from this."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def run(self, **kwargs) -> ToolResult: ...
