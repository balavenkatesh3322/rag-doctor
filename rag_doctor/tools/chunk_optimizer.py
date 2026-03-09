"""
ChunkOptimizer — finds the best chunking strategy via grid search (RC-3).
"""
from __future__ import annotations
import re
from typing import List, Dict, Optional
from ..connectors.base import Document
from ..embeddings import BaseEmbedder, TFIDFEmbedder
from ._embed_utils import fit_and_embed
from .base import BaseTool, ToolResult


STRATEGIES = [
    {"strategy": "fixed",     "chunk_size": 128, "chunk_overlap": 16},
    {"strategy": "fixed",     "chunk_size": 256, "chunk_overlap": 32},
    {"strategy": "fixed",     "chunk_size": 512, "chunk_overlap": 64},
    {"strategy": "fixed",     "chunk_size": 1024,"chunk_overlap": 128},
    {"strategy": "recursive", "chunk_size": 256, "chunk_overlap": 32},
]


def _fixed_chunk(text: str, size: int, overlap: int) -> List[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i+size])
        if chunk.strip():
            chunks.append(chunk)
        i += max(1, size - overlap)
    return chunks


def _recursive_chunk(text: str, size: int, overlap: int) -> List[str]:
    for sep in ["\n\n", "\n", ". ", " "]:
        parts = [p.strip() for p in text.split(sep) if p.strip()]
        if all(len(p.split()) <= size for p in parts):
            return [p for p in parts if len(p.split()) >= 5]
    return _fixed_chunk(text, size, overlap)


def _build_chunks(corpus: List[str], strategy: str, size: int, overlap: int) -> List[str]:
    chunks = []
    for text in corpus:
        fn = _recursive_chunk if strategy == "recursive" else _fixed_chunk
        chunks.extend(fn(text, size, overlap))
    return [c for c in chunks if len(c.split()) >= 5]


def _recall_at_k(
    chunks: List[str],
    test_pairs: List[Dict],
    embedder: Optional[BaseEmbedder],
    top_k: int = 5,
    threshold: float = 0.60,
) -> float:
    if not chunks or not test_pairs:
        return 0.0

    # Build all texts: queries + expected answers + all chunks
    queries   = [p["query"]    for p in test_pairs]
    expecteds = [p["expected"] for p in test_pairs]
    all_texts = queries + expecteds + chunks

    e, all_vecs = fit_and_embed(all_texts, embedder=embedder)

    q_vecs   = all_vecs[:len(queries)]
    exp_vecs = all_vecs[len(queries):len(queries)+len(expecteds)]
    c_vecs   = all_vecs[len(queries)+len(expecteds):]

    hits = 0
    for q_vec, exp_vec in zip(q_vecs, exp_vecs):
        # Score chunks by query similarity
        chunk_scores = [(e.similarity(q_vec, cv), cv) for cv in c_vecs]
        chunk_scores.sort(key=lambda x: -x[0])
        top_cvecs = [cv for _, cv in chunk_scores[:top_k]]

        # Check if expected answer matches any top-k chunk
        best_match = max((e.similarity(exp_vec, cv) for cv in top_cvecs), default=0.0)
        if best_match >= threshold:
            hits += 1

    return hits / len(test_pairs)


class ChunkOptimizer(BaseTool):
    """
    Grid-searches chunking strategies using real recall@k measurement.
    """

    name = "chunk_optimizer"

    def __init__(self, embedder: Optional[BaseEmbedder] = None):
        self._embedder = embedder

    def run(
        self,
        corpus_texts: List[str],
        test_pairs: List[Dict],
        current_strategy: Optional[Dict] = None,
    ) -> ToolResult:
        if not corpus_texts or not test_pairs:
            return ToolResult(
                tool_name=self.name, passed=True,
                finding="Insufficient data for optimization.", severity="low",
            )

        results = []
        for cfg in STRATEGIES:
            chunks = _build_chunks(corpus_texts, cfg["strategy"], cfg["chunk_size"], cfg["chunk_overlap"])
            if not chunks:
                continue
            recall = _recall_at_k(chunks, test_pairs, self._embedder)
            results.append({**cfg, "recall_at_5": round(recall, 3), "num_chunks": len(chunks)})

        results.sort(key=lambda x: -x["recall_at_5"])
        best = results[0] if results else None

        current_recall = None
        if current_strategy and results:
            cur_chunks = _build_chunks(
                corpus_texts,
                current_strategy.get("strategy", "fixed"),
                current_strategy.get("chunk_size", 512),
                current_strategy.get("chunk_overlap", 64),
            )
            current_recall = round(_recall_at_k(cur_chunks, test_pairs, self._embedder), 3)

        improvement = round(best["recall_at_5"] - current_recall, 3) \
                      if best and current_recall is not None else None
        passed = improvement is None or improvement <= 0.05

        return ToolResult(
            tool_name=self.name,
            passed=passed,
            severity="medium" if not passed and improvement and improvement > 0.10 else "low",
            finding=(
                f"Best: {best['strategy']} size={best['chunk_size']} "
                f"(recall@5: {best['recall_at_5']:.3f})"
                if best else "No results."
            ),
            details={
                "ranked_strategies": results,
                "current_recall": current_recall,
                "best_improvement": improvement,
            },
            recommendation=(
                f"Switch to strategy={best['strategy']}, chunk_size={best['chunk_size']}, "
                f"chunk_overlap={best['chunk_overlap']} (+{improvement:.1%} recall)"
                if not passed and best and improvement else None
            ),
        )
