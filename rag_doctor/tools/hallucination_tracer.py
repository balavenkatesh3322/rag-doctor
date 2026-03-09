"""Tool 4: HallucinationTracer — traces claims to source documents."""
from __future__ import annotations
import math
import re
from typing import List, Tuple
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


def _split_claims(text: str) -> List[str]:
    """Split answer into atomic claims at sentence boundaries."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 10]


class HallucinationTracer(BaseTool):
    """
    Traces each claim in the generated answer to a source document.
    Claims without a source match above threshold are flagged as hallucinations.
    """

    name = "hallucination_tracer"

    def __init__(self, faithfulness_threshold: float = 0.60):
        self.faithfulness_threshold = faithfulness_threshold

    def run(self, answer: str, docs: List[Document]) -> ToolResult:
        if not answer or not answer.strip():
            return ToolResult(
                tool_name=self.name, passed=True,
                finding="No answer to trace.", severity="low"
            )

        if not docs:
            return ToolResult(
                tool_name=self.name, passed=False,
                severity="critical",
                finding="Answer generated with no retrieved documents — full hallucination risk."
            )

        claims = _split_claims(answer)
        if not claims:
            claims = [answer]

        grounded: List[Tuple[str, float, int]] = []   # (claim, score, best_doc_pos)
        hallucinated: List[Tuple[str, float]] = []

        for claim in claims:
            c_vec = _embed(claim)
            best_sim = 0.0
            best_pos = -1
            for doc in docs:
                # Check against sentences in doc for finer granularity
                doc_sents = [s.strip() for s in re.split(r'[.!?]+', doc.content) if len(s.strip()) > 5]
                for sent in doc_sents or [doc.content]:
                    sim = _cosine(c_vec, _embed(sent))
                    if sim > best_sim:
                        best_sim = sim
                        best_pos = doc.position

            if best_sim >= self.faithfulness_threshold:
                grounded.append((claim, round(best_sim, 3), best_pos))
            else:
                hallucinated.append((claim, round(best_sim, 3)))

        n_claims = len(claims)
        n_grounded = len(grounded)
        faithfulness = n_grounded / n_claims if n_claims > 0 else 1.0

        passed = faithfulness >= 0.8

        if faithfulness < 0.4:
            severity = "critical"
        elif faithfulness < 0.6:
            severity = "high"
        elif faithfulness < 0.8:
            severity = "medium"
        else:
            severity = "low"

        return ToolResult(
            tool_name=self.name,
            passed=passed,
            severity=severity,
            finding=f"Faithfulness: {faithfulness:.2f} ({n_grounded}/{n_claims} claims grounded)",
            details={
                "faithfulness_score": round(faithfulness, 3),
                "total_claims": n_claims,
                "grounded_claims": n_grounded,
                "hallucinated_claims": [
                    {"claim": c, "best_similarity": s} for c, s in hallucinated
                ],
                "grounded_claim_details": [
                    {"claim": c, "similarity": s, "source_position": p}
                    for c, s, p in grounded
                ],
            },
            recommendation=(
                "Add explicit 'cite your sources' prompt instruction. "
                "Consider lowering LLM temperature or adding a faithfulness filter."
                if not passed else None
            ),
        )
