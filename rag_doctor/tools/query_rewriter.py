"""
QueryRewriter — rewrites queries to improve retrieval recall (RC-5).
"""
from __future__ import annotations
import re
from typing import List, Optional
from ..connectors.base import Document, PipelineConnector
from ..embeddings import BaseEmbedder
from ._embed_utils import fit_and_embed
from .base import BaseTool, ToolResult


def _hyde_rewrite(query: str) -> str:
    q = query.strip().rstrip("?")
    for phrase in ["what is", "what are", "how does", "how do", "when does",
                   "where is", "who is", "why does", "explain", "describe",
                   "how many", "which", "tell me about"]:
        if q.lower().startswith(phrase):
            q = q[len(phrase):].strip()
            break
    return (
        f"The following information describes {q}: "
        f"{q} is defined by the relevant policy, procedure, or technical "
        f"specification. The key details about {q} include the specific "
        f"values, dates, requirements, and conditions that apply."
    )


def _step_back_rewrite(query: str) -> str:
    broad = re.sub(r'"[^"]*"', 'the specified item', query)
    broad = re.sub(r'\b\d+\b', 'the specified value', broad)
    return broad if broad != query else f"general information about {query.lower()}"


def _synonym_expand(query: str) -> str:
    synonyms = {
        r'\bmaternity\b': 'maternity parental',
        r'\bpaternity\b': 'paternity parental',
        r'\bfired\b':     'fired terminated',
        r'\blayoff\b':    'layoff termination',
        r'\bpto\b':       'pto paid time off leave',
        r'\bvacation\b':  'vacation annual leave',
        r'\bsalary\b':    'salary compensation',
        r'\bchunked\b':   'chunked streaming',
    }
    result = query
    for pattern, replacement in synonyms.items():
        result = re.sub(pattern, replacement, result, flags=re.I)
    return result


class QueryRewriter(BaseTool):
    name = "query_rewriter"

    def __init__(self, embedder: Optional[BaseEmbedder] = None):
        self._embedder = embedder

    def _score_rewrite(
        self, rewritten: str, original_docs: List[Document], expected: Optional[str]
    ) -> float:
        """Score a rewrite by embedding similarity to docs/expected."""
        if not original_docs:
            return 0.0
        candidates = [expected] if expected else []
        candidates += [d.content for d in original_docs[:3]]
        all_texts = [rewritten] + candidates
        e, vecs = fit_and_embed(all_texts, embedder=self._embedder)
        rw_vec = vecs[0]
        return max(e.similarity(rw_vec, v) for v in vecs[1:]) if len(vecs) > 1 else 0.0

    def run(
        self,
        query: str,
        original_docs: List[Document],
        connector: Optional[PipelineConnector] = None,
        expected: Optional[str] = None,
        top_k: int = 5,
    ) -> ToolResult:
        original_score = max((d.score for d in original_docs), default=0.0)
        rewrites = {
            "hyde":      _hyde_rewrite(query),
            "step_back": _step_back_rewrite(query),
            "synonym":   _synonym_expand(query),
        }

        results = {"original": {"query": query, "top_score": round(original_score, 3)}}
        best_strategy = "original"
        best_score = original_score

        for name, rewritten in rewrites.items():
            if rewritten == query:
                continue
            if connector:
                try:
                    new_docs = connector.retrieve(rewritten, top_k=top_k)
                    new_score = max((d.score for d in new_docs), default=0.0)
                    if expected and new_docs:
                        # Also check semantic match to expected
                        all_texts = [expected] + [d.content for d in new_docs]
                        e, vecs = fit_and_embed(all_texts, embedder=self._embedder)
                        match = max(e.similarity(vecs[0], v) for v in vecs[1:])
                        new_score = max(new_score, match)
                    results[name] = {"query": rewritten, "top_score": round(new_score, 3),
                                     "docs_retrieved": len(new_docs)}
                    if new_score > best_score:
                        best_score = new_score
                        best_strategy = name
                except Exception as ex:
                    results[name] = {"query": rewritten, "error": str(ex)}
            else:
                new_score = self._score_rewrite(rewritten, original_docs, expected)
                results[name] = {"query": rewritten, "top_score": round(new_score, 3)}
                if new_score > best_score:
                    best_score = new_score
                    best_strategy = name

        improvement = best_score - original_score
        improved = improvement > 0.03

        return ToolResult(
            tool_name=self.name,
            passed=not improved,
            severity="medium" if improved else "low",
            finding=(
                f"'{best_strategy}' rewrite improved retrieval by +{improvement:.3f} "
                f"({original_score:.3f} → {best_score:.3f})"
                if improved else
                f"No rewrite improved retrieval (best: {original_score:.3f})"
            ),
            details={
                "original_query": query,
                "strategy_results": results,
                "best_strategy": best_strategy,
                "best_score": round(best_score, 3),
                "improvement": round(improvement, 3),
            },
            recommendation=(
                f"Use '{best_strategy}' rewrite: '{rewrites.get(best_strategy, query)}'"
                if improved else None
            ),
        )
