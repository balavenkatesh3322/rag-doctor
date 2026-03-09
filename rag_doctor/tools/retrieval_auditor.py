"""Tool 2: RetrievalAuditor — checks if the correct document was retrieved."""
from __future__ import annotations
import math
from typing import List, Optional
from ..connectors.base import Document
from .base import BaseTool, ToolResult


def _embed(text: str, dim: int = 64) -> List[float]:
    vec = [0.0] * dim
    for ch in text.lower():
        vec[ord(ch) % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


class RetrievalAuditor(BaseTool):
    """
    Checks whether the answer-bearing document was retrieved.
    Reports recall@k and best matching document position.
    """

    name = "retrieval_auditor"

    def __init__(self, recall_threshold: float = 0.75):
        self.recall_threshold = recall_threshold

    def run(
        self,
        docs: List[Document],
        expected: Optional[str] = None,
        query: str = "",
    ) -> ToolResult:
        if not docs:
            return ToolResult(
                tool_name=self.name, passed=False,
                severity="critical", finding="No documents were retrieved."
            )

        if not expected:
            # Without ground truth, check retrieval score distribution
            scores = [d.score for d in docs]
            avg_score = sum(scores) / len(scores) if scores else 0
            top_score = max(scores) if scores else 0
            passed = top_score >= self.recall_threshold
            return ToolResult(
                tool_name=self.name,
                passed=passed,
                severity="medium" if not passed else "low",
                finding=f"Top retrieval score: {top_score:.3f} (threshold: {self.recall_threshold})",
                details={"top_score": round(top_score, 3), "avg_score": round(avg_score, 3)},
                recommendation="Improve embedding model or chunking strategy." if not passed else None,
            )

        # With ground truth: check which doc best matches expected answer
        exp_vec = _embed(expected)
        best_score = 0.0
        best_pos = -1
        scores_by_pos = {}

        for doc in docs:
            doc_vec = _embed(doc.content)
            sim = _cosine(exp_vec, doc_vec)
            scores_by_pos[doc.position] = round(sim, 3)
            if sim > best_score:
                best_score = sim
                best_pos = doc.position

        recall_hit = best_score >= self.recall_threshold

        if not recall_hit:
            severity = "high"
            finding = f"Expected answer not found in top-{len(docs)} results. Best match: {best_score:.3f}"
        else:
            severity = "low"
            finding = f"Expected answer found at position {best_pos} (score: {best_score:.3f})"

        return ToolResult(
            tool_name=self.name,
            passed=recall_hit,
            severity=severity,
            finding=finding,
            details={
                "recall_hit": recall_hit,
                "best_match_position": best_pos,
                "best_match_score": round(best_score, 3),
                "scores_by_position": scores_by_pos,
                "recall_threshold": self.recall_threshold,
            },
            recommendation=(
                "Retrieval failed. Try QueryRewriter (HyDE) or improve corpus coverage."
                if not recall_hit else None
            ),
        )
