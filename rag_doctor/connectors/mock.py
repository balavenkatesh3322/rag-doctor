"""Mock connector for testing without external APIs."""
from __future__ import annotations
import math
import re
from typing import List, Dict, Optional
from .base import Document, PipelineConnector


def _simple_embed(text: str, dim: int = 64) -> List[float]:
    """Deterministic pseudo-embedding based on character frequencies."""
    vec = [0.0] * dim
    for i, ch in enumerate(text.lower()):
        vec[ord(ch) % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


class MockConnector(PipelineConnector):
    """
    In-memory connector for unit/integration tests.
    Supports injecting a corpus of documents and a custom answer function.
    """

    def __init__(
        self,
        corpus: Optional[List[Dict]] = None,
        answer_fn=None,
        inject_position_bias: bool = False,
        inject_hallucination: bool = False,
    ):
        self.corpus: List[Document] = []
        self.inject_position_bias = inject_position_bias
        self.inject_hallucination = inject_hallucination
        self._answer_fn = answer_fn

        if corpus:
            for i, item in enumerate(corpus):
                self.corpus.append(Document(
                    content=item["content"],
                    metadata=item.get("metadata", {}),
                    doc_id=item.get("id", f"doc_{i}"),
                ))

    def add_document(self, content: str, metadata: dict = None, doc_id: str = None):
        idx = len(self.corpus)
        self.corpus.append(Document(
            content=content,
            metadata=metadata or {},
            doc_id=doc_id or f"doc_{idx}",
        ))

    def embed(self, text: str) -> List[float]:
        return _simple_embed(text)

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        if not self.corpus:
            return []

        q_vec = _simple_embed(query)
        scored = []
        for doc in self.corpus:
            d_vec = _simple_embed(doc.content)
            score = _cosine(q_vec, d_vec)
            scored.append((score, doc))

        scored.sort(key=lambda x: -x[0])
        results = []
        for pos, (score, doc) in enumerate(scored[:top_k]):
            results.append(Document(
                content=doc.content,
                metadata=doc.metadata,
                score=score,
                position=pos,
                doc_id=doc.doc_id,
            ))

        # Inject position bias: shuffle best doc to middle
        if self.inject_position_bias and len(results) >= 3:
            best = results.pop(0)
            mid = len(results) // 2
            best.position = mid
            results.insert(mid, best)
            for i, r in enumerate(results):
                r.position = i

        return results

    def generate(self, query: str, docs: List[Document]) -> str:
        if self._answer_fn:
            return self._answer_fn(query, docs)

        if self.inject_hallucination:
            return "This feature supports advanced quantum entanglement processing with the --quantum flag."

        # Simple extractive: return first sentence of best doc
        if docs:
            first_sent = docs[0].content.split(".")[0].strip()
            return first_sent + "."
        return "I don't have enough information to answer."
