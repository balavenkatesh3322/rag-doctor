"""Tool 6: QueryRewriter — rewrites queries via HyDE for better retrieval."""
from __future__ import annotations
import math
from typing import List, Optional
from ..connectors.base import Document, PipelineConnector
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


def _hyde_rewrite(query: str) -> str:
    """
    HyDE: generate a hypothetical document that would answer this query.
    In production this would call an LLM. Here we use keyword expansion heuristics.
    """
    # Simple heuristic: expand question into declarative statement
    q = query.strip().rstrip("?")
    # Remove question words
    for word in ["what is", "what are", "how does", "how do", "when does",
                 "where is", "who is", "why does", "explain", "describe"]:
        if q.lower().startswith(word):
            q = q[len(word):].strip()
            break
    # Construct a hypothetical answer document
    hyde = f"The answer regarding {q} is as follows: {q} refers to the specific policy, "
    hyde += f"process, or technical detail associated with {q}."
    return hyde


def _step_back_rewrite(query: str) -> str:
    """Step-back prompting: broaden the query to a higher abstraction level."""
    # Heuristic: remove specific names/numbers and broaden
    import re
    # Remove quoted strings (specific names)
    broad = re.sub(r'"[^"]*"', 'the specified item', query)
    # Replace specific numbers with "the specified value"
    broad = re.sub(r'\b\d+\b', 'the specified value', broad)
    broad = broad.replace("specifically", "generally")
    return broad if broad != query else f"What is the general policy about {query.lower()}?"


class QueryRewriter(BaseTool):
    """
    Rewrites queries using HyDE and step-back techniques
    to improve retrieval recall for low-recall queries.
    """

    name = "query_rewriter"

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
            "hyde": _hyde_rewrite(query),
            "step_back": _step_back_rewrite(query),
        }

        results = {"original": {"query": query, "top_score": round(original_score, 3)}}
        best_strategy = "original"
        best_score = original_score

        if connector:
            for strategy_name, rewritten_query in rewrites.items():
                try:
                    new_docs = connector.retrieve(rewritten_query, top_k=top_k)
                    new_score = max((d.score for d in new_docs), default=0.0)
                    results[strategy_name] = {
                        "query": rewritten_query,
                        "top_score": round(new_score, 3),
                        "docs_retrieved": len(new_docs),
                    }
                    if new_score > best_score:
                        best_score = new_score
                        best_strategy = strategy_name
                except Exception as e:
                    results[strategy_name] = {"error": str(e)}
        else:
            # Without connector, just report the rewrite suggestions
            for strategy_name, rw in rewrites.items():
                results[strategy_name] = {"query": rw, "top_score": None}
            best_strategy = "hyde"

        improvement = best_score - original_score
        improved = improvement > 0.05

        return ToolResult(
            tool_name=self.name,
            passed=not improved,  # "passed" means no rewrite needed
            severity="medium" if improved else "low",
            finding=(
                f"Query rewriting improved retrieval by +{improvement:.3f} using '{best_strategy}'"
                if improved
                else f"Query rewriting did not improve retrieval (original score: {original_score:.3f})"
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
