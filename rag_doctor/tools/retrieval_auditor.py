"""
RetrievalAuditor — checks if the correct document was retrieved (RC-1).
"""
from __future__ import annotations
import re
from typing import List, Optional
from ..connectors.base import Document
from ..embeddings import BaseEmbedder
from ._embed_utils import fit_and_embed
from .base import BaseTool, ToolResult


class RetrievalAuditor(BaseTool):
    """
    Checks whether the expected answer is semantically present in retrieved docs.
    Uses fit_and_embed() so vocabulary covers both expected text and doc content.
    """

    name = "retrieval_auditor"

    def __init__(self, recall_threshold: float = 0.50, embedder: Optional[BaseEmbedder] = None):
        self.recall_threshold = recall_threshold
        self._embedder = embedder

    def run(self, docs: List[Document], expected: Optional[str] = None, query: str = "") -> ToolResult:
        if not docs:
            return ToolResult(
                tool_name=self.name, passed=False, severity="critical",
                finding="No documents were retrieved.",
                details={"recall_hit": False},
            )

        # Without ground truth: use retrieval scores from vector search
        if not expected:
            scores = [d.score for d in docs]
            top = max(scores) if scores else 0.0
            avg = sum(scores) / len(scores) if scores else 0.0
            passed = top >= self.recall_threshold
            return ToolResult(
                tool_name=self.name, passed=passed,
                severity="medium" if not passed else "low",
                finding=(
                    f"Top retrieval score: {top:.3f} "
                    f"({'above' if passed else 'below'} threshold {self.recall_threshold})"
                ),
                details={
                    "top_score": round(top, 3), "avg_score": round(avg, 3),
                    "recall_threshold": self.recall_threshold,
                },
                recommendation="Low retrieval scores. Check embedding quality or query." if not passed else None,
            )

        # With ground truth: embed expected + all doc sentences together
        def _sentences(text):
            parts = re.split(r'(?<=[.!?])\s+', text.strip())
            return [p.strip() for p in parts if len(p.strip()) > 5] or [text]

        doc_segments: List[str] = []
        seg_positions: List[int] = []
        for doc in docs:
            for sent in _sentences(doc.content):
                doc_segments.append(sent)
                seg_positions.append(doc.position)

        all_texts = [expected] + doc_segments
        e, all_vecs = fit_and_embed(all_texts, embedder=self._embedder)

        exp_vec  = all_vecs[0]
        seg_vecs = all_vecs[1:]

        best_score, best_pos = 0.0, -1
        scores_by_pos = {}
        for seg, pos, s_vec in zip(doc_segments, seg_positions, seg_vecs):
            sim = e.similarity(exp_vec, s_vec)
            if sim > best_score:
                best_score = sim
                best_pos = pos
            scores_by_pos[pos] = max(scores_by_pos.get(pos, 0), round(sim, 3))

        recall_hit = best_score >= self.recall_threshold

        return ToolResult(
            tool_name=self.name,
            passed=recall_hit,
            severity="high" if not recall_hit else "low",
            finding=(
                f"Expected answer NOT found in top-{len(docs)} results. "
                f"Best similarity: {best_score:.3f} (threshold: {self.recall_threshold})"
                if not recall_hit else
                f"Expected answer found at position {best_pos} (similarity: {best_score:.3f})"
            ),
            details={
                "recall_hit": recall_hit,
                "best_match_position": best_pos,
                "best_match_score": round(best_score, 3),
                "scores_by_position": scores_by_pos,
                "recall_threshold": self.recall_threshold,
                "embedder": e.name,
            },
            recommendation=(
                "Retrieval failed. Run QueryRewriter or check corpus coverage."
                if not recall_hit else None
            ),
        )
