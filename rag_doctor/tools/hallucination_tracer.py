"""
HallucinationTracer — traces answer claims to source documents (RC-4).

Each claim is checked against every sentence in retrieved docs.
Faithfulness = fraction of claims grounded in at least one source sentence.
"""
from __future__ import annotations
import re
from typing import List, Optional
from ..connectors.base import Document
from ..embeddings import BaseEmbedder
from ._embed_utils import fit_and_embed
from .base import BaseTool, ToolResult


def _split_claims(text: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 10]


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 5]


class HallucinationTracer(BaseTool):
    """
    Checks whether each claim in the generated answer can be traced
    to at least one sentence in the retrieved documents.

    Uses fit_and_embed() so TF-IDF vocabulary covers both the answer
    claims AND the document sentences — avoiding false high-similarity
    from char-frequency fallback.
    """

    name = "hallucination_tracer"

    def __init__(
        self,
        faithfulness_threshold: float = 0.60,
        embedder: Optional[BaseEmbedder] = None,
    ):
        self.faithfulness_threshold = faithfulness_threshold
        self._embedder = embedder

    def run(self, answer: str, docs: List[Document]) -> ToolResult:
        if not answer or not answer.strip():
            return ToolResult(
                tool_name=self.name, passed=True,
                finding="No answer to trace.", severity="low",
            )

        if not docs:
            return ToolResult(
                tool_name=self.name, passed=False, severity="critical",
                finding="Answer generated with no retrieved documents — full hallucination risk.",
                details={"faithfulness_score": 0.0, "total_claims": 0},
            )

        claims = _split_claims(answer) or [answer]

        # Collect all doc sentences
        doc_sentences: List[str] = []
        doc_positions: List[int] = []
        for doc in docs:
            for sent in (_split_sentences(doc.content) or [doc.content]):
                doc_sentences.append(sent)
                doc_positions.append(doc.position)

        # Embed everything together so TF-IDF vocabulary covers all texts
        all_texts = claims + doc_sentences
        e, all_vecs = fit_and_embed(all_texts, embedder=self._embedder)

        claim_vecs = all_vecs[:len(claims)]
        sent_vecs  = all_vecs[len(claims):]

        grounded, hallucinated = [], []

        for claim, c_vec in zip(claims, claim_vecs):
            best_sim, best_pos = 0.0, -1
            for (sent, s_pos, s_vec) in zip(doc_sentences, doc_positions, sent_vecs):
                sim = e.similarity(c_vec, s_vec)
                if sim > best_sim:
                    best_sim = sim
                    best_pos = s_pos

            if best_sim >= self.faithfulness_threshold:
                grounded.append({"claim": claim, "similarity": round(best_sim, 3),
                                  "source_position": best_pos})
            else:
                hallucinated.append({"claim": claim, "best_similarity": round(best_sim, 3)})

        n = len(claims)
        faithfulness = len(grounded) / n if n > 0 else 1.0
        passed = faithfulness >= 0.80

        if faithfulness < 0.35:   severity = "critical"
        elif faithfulness < 0.60: severity = "high"
        elif faithfulness < 0.80: severity = "medium"
        else:                     severity = "low"

        return ToolResult(
            tool_name=self.name,
            passed=passed,
            severity=severity,
            finding=f"Faithfulness: {faithfulness:.2f} ({len(grounded)}/{n} claims grounded in sources)",
            details={
                "faithfulness_score": round(faithfulness, 3),
                "total_claims": n,
                "grounded_claims": len(grounded),
                "hallucinated_claims": hallucinated,
                "grounded_claim_details": grounded,
                "faithfulness_threshold": self.faithfulness_threshold,
                "embedder": e.name,
            },
            recommendation=(
                "Add 'Answer only from provided sources' to your system prompt. "
                "Lower LLM temperature. Add a faithfulness filter."
                if not passed else None
            ),
        )
