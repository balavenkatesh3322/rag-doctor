"""
PositionTester — detects the 'lost in the middle' position bias (RC-2).
"""
from __future__ import annotations
from typing import List, Optional
from ..connectors.base import Document
from ..embeddings import BaseEmbedder
from ._embed_utils import fit_and_embed
from .base import BaseTool, ToolResult


class PositionTester(BaseTool):
    """
    Detects whether the most relevant document is in a middle 'danger zone'
    where LLMs statistically underperform (Liu et al. 2023).
    """

    name = "position_tester"

    def __init__(self, embedder: Optional[BaseEmbedder] = None):
        self._embedder = embedder

    def _find_best_position(
        self, docs: List[Document], query: str = "", expected: Optional[str] = None
    ) -> int:
        if not docs:
            return -1

        # Priority 1: match against expected answer
        if expected:
            all_texts = [expected] + [d.content for d in docs]
            e, vecs = fit_and_embed(all_texts, embedder=self._embedder)
            exp_vec = vecs[0]
            doc_vecs = vecs[1:]
            sims = [e.similarity(exp_vec, dv) for dv in doc_vecs]
            return docs[sims.index(max(sims))].position

        # Priority 2: retrieval scores (already computed by vector search)
        scores = [d.score for d in docs]
        if max(scores) > 0:
            return docs[scores.index(max(scores))].position

        # Priority 3: embedding similarity to query
        if query:
            all_texts = [query] + [d.content for d in docs]
            e, vecs = fit_and_embed(all_texts, embedder=self._embedder)
            q_vec = vecs[0]
            doc_vecs = vecs[1:]
            sims = [e.similarity(q_vec, dv) for dv in doc_vecs]
            return docs[sims.index(max(sims))].position

        return docs[0].position

    def run(
        self,
        docs: List[Document],
        query: str = "",
        expected: Optional[str] = None,
        best_position: int = -1,   # legacy param
        top_k: int = 5,
    ) -> ToolResult:
        if not docs:
            return ToolResult(
                tool_name=self.name, passed=True,
                finding="No documents to test.", severity="low",
            )

        n = len(docs)
        pos = best_position if best_position >= 0 else \
              self._find_best_position(docs, query=query, expected=expected)

        in_danger = (n >= 3) and (0 < pos < n - 1)

        if n <= 2 or not in_danger:
            risk = 0.0
        else:
            dist = min(pos, n - 1 - pos)
            max_dist = (n - 1) / 2
            risk = dist / max_dist if max_dist > 0 else 0.0

        severity = "high" if in_danger and risk >= 0.5 else ("medium" if in_danger else "low")

        return ToolResult(
            tool_name=self.name,
            passed=not in_danger,
            severity=severity,
            finding=(
                f"Best document at position {pos}/{n-1} — in danger zone (risk: {risk:.2f})"
                if in_danger else
                f"Best document at position {pos} — safe position."
            ),
            details={
                "best_position": pos, "total_docs": n,
                "in_danger_zone": in_danger, "position_risk_score": round(risk, 3),
            },
            recommendation=(
                "Enable a reranker to push the most relevant document to position 0."
                if in_danger else None
            ),
        )
