"""Tool 5: ChunkOptimizer — finds best chunking strategy via grid search."""
from __future__ import annotations
import math
import re
from typing import List, Dict, Any
from ..connectors.base import Document
from .base import BaseTool, ToolResult

STRATEGIES = [
    {"strategy": "fixed",     "chunk_size": 128, "chunk_overlap": 16},
    {"strategy": "fixed",     "chunk_size": 256, "chunk_overlap": 32},
    {"strategy": "fixed",     "chunk_size": 512, "chunk_overlap": 64},
    {"strategy": "fixed",     "chunk_size": 1024,"chunk_overlap": 128},
    {"strategy": "recursive", "chunk_size": 256, "chunk_overlap": 32},
]


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


def _fixed_chunk(text: str, size: int, overlap: int) -> List[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i + size])
        chunks.append(chunk)
        i += size - overlap
    return chunks


def _recursive_chunk(text: str, size: int, overlap: int) -> List[str]:
    """Split on paragraphs first, then sentences, then words."""
    separators = ["\n\n", "\n", ". ", " "]
    for sep in separators:
        parts = text.split(sep)
        if all(len(p.split()) <= size for p in parts):
            return [p.strip() for p in parts if p.strip()]
    return _fixed_chunk(text, size, overlap)


def _chunk_corpus(corpus_texts: List[str], strategy: str, size: int, overlap: int) -> List[str]:
    chunks = []
    for text in corpus_texts:
        if strategy == "recursive":
            chunks.extend(_recursive_chunk(text, size, overlap))
        else:
            chunks.extend(_fixed_chunk(text, size, overlap))
    return [c for c in chunks if len(c.split()) >= 5]


def _recall_at_k(chunks: List[str], test_pairs: List[Dict], top_k: int = 5) -> float:
    """Compute recall@k: fraction of test queries where ground truth in top-k."""
    hits = 0
    for pair in test_pairs:
        q_vec = _embed(pair["query"])
        exp_vec = _embed(pair["expected"])
        scored = sorted(
            [(_cosine(q_vec, _embed(c)), c) for c in chunks], reverse=True
        )[:top_k]
        top_chunks = [c for _, c in scored]
        best_match = max(_cosine(exp_vec, _embed(c)) for c in top_chunks) if top_chunks else 0
        if best_match >= 0.70:
            hits += 1
    return hits / len(test_pairs) if test_pairs else 0.0


class ChunkOptimizer(BaseTool):
    """
    Tests multiple chunking strategies on a corpus + test set.
    Returns ranked strategies with recall@5 scores.
    """

    name = "chunk_optimizer"

    def run(
        self,
        corpus_texts: List[str],
        test_pairs: List[Dict],  # [{"query": ..., "expected": ...}]
        current_strategy: Dict = None,
    ) -> ToolResult:
        if not corpus_texts or not test_pairs:
            return ToolResult(
                tool_name=self.name, passed=True,
                finding="Insufficient data for optimization.", severity="low"
            )

        results = []
        for cfg in STRATEGIES:
            chunks = _chunk_corpus(corpus_texts, cfg["strategy"], cfg["chunk_size"], cfg["chunk_overlap"])
            if not chunks:
                continue
            recall = _recall_at_k(chunks, test_pairs)
            results.append({**cfg, "recall_at_5": round(recall, 3), "num_chunks": len(chunks)})

        results.sort(key=lambda x: -x["recall_at_5"])
        best = results[0] if results else None

        # Compare against current config if provided
        current_recall = None
        if current_strategy and results:
            cur_chunks = _chunk_corpus(
                corpus_texts,
                current_strategy.get("strategy", "fixed"),
                current_strategy.get("chunk_size", 512),
                current_strategy.get("chunk_overlap", 64),
            )
            current_recall = round(_recall_at_k(cur_chunks, test_pairs), 3)

        improvement = None
        if best and current_recall is not None:
            improvement = round(best["recall_at_5"] - current_recall, 3)

        passed = improvement is None or improvement <= 0.05

        return ToolResult(
            tool_name=self.name,
            passed=passed,
            severity="medium" if not passed and improvement and improvement > 0.1 else "low",
            finding=(
                f"Best strategy: {best['strategy']} size={best['chunk_size']} "
                f"(recall@5: {best['recall_at_5']:.3f})"
                if best else "No optimization results."
            ),
            details={
                "ranked_strategies": results,
                "current_recall": current_recall,
                "best_improvement": improvement,
            },
            recommendation=(
                f"Switch to: strategy={best['strategy']}, chunk_size={best['chunk_size']}, "
                f"chunk_overlap={best['chunk_overlap']} for +{improvement:.1%} recall improvement."
                if not passed and best and improvement else None
            ),
        )
