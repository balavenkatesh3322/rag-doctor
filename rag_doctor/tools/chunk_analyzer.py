"""Tool 1: ChunkAnalyzer — detects fragmentation and incoherence."""
from __future__ import annotations
import math
import re
from typing import List
from ..connectors.base import Document
from .base import BaseTool, ToolResult


def _embed_simple(text: str, dim: int = 32) -> List[float]:
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


def _token_count(text: str) -> int:
    """Approximate token count (words * 1.3)."""
    return int(len(text.split()) * 1.3)


def _sentence_coherence(text: str) -> float:
    """Measure avg cosine similarity between adjacent sentences."""
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]
    if len(sentences) < 2:
        return 1.0
    scores = []
    for i in range(len(sentences) - 1):
        a = _embed_simple(sentences[i])
        b = _embed_simple(sentences[i + 1])
        scores.append(_cosine(a, b))
    return sum(scores) / len(scores) if scores else 1.0


def _detect_mid_sentence_cut(text: str) -> bool:
    """Check if text starts or ends mid-sentence."""
    stripped = text.strip()
    starts_lowercase = bool(stripped) and stripped[0].islower()
    ends_no_punct = bool(stripped) and stripped[-1] not in ".!?\"'"
    return starts_lowercase or ends_no_punct


class ChunkAnalyzer(BaseTool):
    """
    Analyzes document chunks for:
    - Mid-sentence truncation
    - Low intra-chunk semantic coherence
    - Abnormal chunk sizes
    """

    name = "chunk_analyzer"

    def __init__(self, coherence_threshold: float = 0.65, ideal_token_range=(100, 800)):
        self.coherence_threshold = coherence_threshold
        self.ideal_token_range = ideal_token_range

    def run(self, docs: List[Document], query: str = "") -> ToolResult:
        if not docs:
            return ToolResult(
                tool_name=self.name, passed=True,
                finding="No documents to analyze.", severity="low"
            )

        issues = []
        coherence_scores = []
        token_counts = []
        truncated_count = 0

        for doc in docs:
            tokens = _token_count(doc.content)
            token_counts.append(tokens)
            coh = _sentence_coherence(doc.content)
            coherence_scores.append(coh)

            if _detect_mid_sentence_cut(doc.content):
                truncated_count += 1

            if tokens < self.ideal_token_range[0]:
                issues.append(f"Doc at pos {doc.position}: too small ({tokens} tokens)")
            elif tokens > self.ideal_token_range[1]:
                issues.append(f"Doc at pos {doc.position}: too large ({tokens} tokens)")

            if coh < self.coherence_threshold:
                issues.append(
                    f"Doc at pos {doc.position}: low coherence ({coh:.2f} < {self.coherence_threshold})"
                )

        avg_coherence = sum(coherence_scores) / len(coherence_scores)
        avg_tokens = sum(token_counts) / len(token_counts)
        passed = len(issues) == 0

        if truncated_count > len(docs) * 0.5:
            severity = "high"
            finding = f"{truncated_count}/{len(docs)} chunks appear truncated mid-sentence."
        elif avg_coherence < self.coherence_threshold:
            severity = "medium"
            finding = f"Low average chunk coherence: {avg_coherence:.2f}"
        elif issues:
            severity = "low"
            finding = f"{len(issues)} minor chunk quality issue(s) found."
        else:
            severity = "low"
            finding = "Chunks appear well-formed."

        return ToolResult(
            tool_name=self.name,
            passed=passed,
            severity=severity,
            finding=finding,
            details={
                "avg_coherence": round(avg_coherence, 3),
                "avg_tokens": round(avg_tokens, 1),
                "truncated_chunks": truncated_count,
                "total_chunks": len(docs),
                "issues": issues,
            },
            recommendation=(
                "Reduce chunk_size or switch to recursive/semantic chunking strategy."
                if not passed else None
            ),
        )
