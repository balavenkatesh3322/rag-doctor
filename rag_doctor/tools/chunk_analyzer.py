"""
ChunkAnalyzer — detects chunk fragmentation and incoherence (RC-3).
"""
from __future__ import annotations
import re
from typing import List, Optional
from ..connectors.base import Document
from ..embeddings import BaseEmbedder
from ._embed_utils import fit_and_embed
from .base import BaseTool, ToolResult


def _token_count(text: str) -> int:
    return int(len(text.split()) * 1.3)


def _detect_mid_sentence_cut(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return stripped[0].islower() or stripped[-1] not in ".!?\"'\u201d"


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 10]


class ChunkAnalyzer(BaseTool):
    """
    Analyzes retrieved chunks for:
    - Mid-sentence truncation
    - Low semantic coherence between adjacent sentences
    - Abnormal chunk sizes

    Uses fit_and_embed() so TF-IDF vocab covers all chunk sentences.
    """

    name = "chunk_analyzer"

    def __init__(
        self,
        coherence_threshold: float = 0.30,
        ideal_token_range: tuple = (80, 800),
        embedder: Optional[BaseEmbedder] = None,
    ):
        self.coherence_threshold = coherence_threshold
        self.ideal_token_range = ideal_token_range
        self._embedder = embedder

    def _chunk_coherence(self, sentences: List[str], e, vecs) -> float:
        if len(sentences) < 2:
            return 1.0
        scores = [e.similarity(vecs[i], vecs[i+1]) for i in range(len(vecs)-1)]
        return sum(scores) / len(scores)

    def run(self, docs: List[Document], query: str = "") -> ToolResult:
        if not docs:
            return ToolResult(
                tool_name=self.name, passed=True,
                finding="No documents to analyze.", severity="low",
            )

        # Collect all sentences across all docs for joint embedding
        doc_sentences: List[List[str]] = []
        all_sentences: List[str] = []
        for doc in docs:
            sents = _split_sentences(doc.content) or [doc.content]
            doc_sentences.append(sents)
            all_sentences.extend(sents)

        # Embed all at once so TF-IDF vocab covers every chunk
        e, all_vecs = fit_and_embed(all_sentences, embedder=self._embedder)

        issues = []
        coherence_scores = []
        token_counts = []
        truncated_count = 0

        idx = 0
        for doc, sents in zip(docs, doc_sentences):
            n = len(sents)
            vecs = all_vecs[idx:idx+n]
            idx += n

            tokens = _token_count(doc.content)
            token_counts.append(tokens)
            coh = self._chunk_coherence(sents, e, vecs)
            coherence_scores.append(coh)

            if _detect_mid_sentence_cut(doc.content):
                truncated_count += 1

            lo, hi = self.ideal_token_range
            if tokens < lo:
                issues.append(f"pos={doc.position}: too small ({tokens} tokens)")
            elif tokens > hi:
                issues.append(f"pos={doc.position}: too large ({tokens} tokens)")

            if coh < self.coherence_threshold:
                issues.append(f"pos={doc.position}: low coherence ({coh:.3f})")

        avg_coh = sum(coherence_scores) / len(coherence_scores)
        avg_tok = sum(token_counts) / len(token_counts)

        if truncated_count > len(docs) * 0.4:
            severity = "high"
            finding = f"{truncated_count}/{len(docs)} chunks appear truncated mid-sentence."
            passed = False
        elif avg_coh < self.coherence_threshold:
            severity = "medium"
            finding = f"Low average chunk coherence: {avg_coh:.3f} (threshold: {self.coherence_threshold})"
            passed = False
        elif issues:
            severity = "low"
            finding = f"{len(issues)} minor chunk quality issue(s) found."
            passed = False
        else:
            severity = "low"
            finding = "Chunks appear well-formed."
            passed = True

        return ToolResult(
            tool_name=self.name,
            passed=passed,
            severity=severity,
            finding=finding,
            details={
                "avg_coherence": round(avg_coh, 3),
                "avg_tokens": round(avg_tok, 1),
                "truncated_chunks": truncated_count,
                "total_chunks": len(docs),
                "coherence_threshold": self.coherence_threshold,
                "issues": issues,
                "embedder": e.name,
            },
            recommendation=(
                "Reduce chunk_size or switch to recursive/semantic chunking."
                if not passed else None
            ),
        )
