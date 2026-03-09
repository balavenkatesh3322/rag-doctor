"""Tool 3: PositionTester — detects 'lost in the middle' pattern."""
from __future__ import annotations
from typing import List
from ..connectors.base import Document
from .base import BaseTool, ToolResult


class PositionTester(BaseTool):
    """
    Detects whether the best document is in a 'danger zone' middle position
    where LLMs tend to ignore context (lost-in-the-middle phenomenon).
    """

    name = "position_tester"

    def run(
        self,
        docs: List[Document],
        best_position: int = -1,
        top_k: int = 5,
    ) -> ToolResult:
        if best_position < 0 or not docs:
            return ToolResult(
                tool_name=self.name, passed=True,
                finding="No position data available.", severity="low"
            )

        n = len(docs)
        # Danger zone: any position that is neither first (0) nor last (n-1)
        in_danger = 0 < best_position < (n - 1)

        # Score the risk: middle positions are worst
        if n <= 1:
            risk = 0.0
        else:
            # Distance from nearest safe position (0 or n-1)
            dist_from_safe = min(best_position, n - 1 - best_position)
            max_dist = (n - 1) / 2
            risk = dist_from_safe / max_dist if max_dist > 0 else 0.0

        if in_danger and risk > 0.6:
            severity = "high"
        elif in_danger:
            severity = "medium"
        else:
            severity = "low"

        passed = not in_danger

        return ToolResult(
            tool_name=self.name,
            passed=passed,
            severity=severity,
            finding=(
                f"Best document at position {best_position}/{n-1} — in danger zone (risk: {risk:.2f})"
                if in_danger
                else f"Best document at position {best_position} — safe position."
            ),
            details={
                "best_position": best_position,
                "total_docs": n,
                "in_danger_zone": in_danger,
                "position_risk_score": round(risk, 3),
            },
            recommendation=(
                "Enable a reranker (e.g. Cohere Rerank) to push best document to position 0. "
                "Alternatively, use Maximal Marginal Relevance (MMR) retrieval."
                if in_danger else None
            ),
        )
