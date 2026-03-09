"""
MockConnector — offline connector for testing and development.

Uses VectorStore + real embeddings so retrieval quality matches production.
The embed() method exposes a fixed-dim interface via CharFreqEmbedder for
tests that check embedding dimensions directly.
"""

from __future__ import annotations

from typing import List, Dict, Optional, Callable

from .base import Document, PipelineConnector
from ..vector_store import VectorStore
from ..embeddings import BaseEmbedder, CharFreqEmbedder, reset_embedder_cache


class MockConnector(PipelineConnector):
    """
    In-memory connector for testing. Uses real semantic retrieval.

    Args:
        corpus:               List of {"content", "id", "metadata"} dicts.
        answer_fn:            Custom fn(query, docs) -> str.
        embedder:             Override embedding backend.
        inject_position_bias: Move best doc to middle position.
        inject_hallucination: Force an off-topic hallucinated answer.
        quiet:                Suppress embedder messages.
    """

    def __init__(
        self,
        corpus: Optional[List[Dict]] = None,
        answer_fn: Optional[Callable] = None,
        embedder: Optional[BaseEmbedder] = None,
        inject_position_bias: bool = False,
        inject_hallucination: bool = False,
        quiet: bool = True,
    ):
        self._store = VectorStore(embedder=embedder, quiet=quiet)
        self._answer_fn = answer_fn
        self.inject_position_bias = inject_position_bias
        self.inject_hallucination = inject_hallucination

        # Fixed-dim embedder for the public embed() method so tests
        # can assert on a stable dimension regardless of backend.
        self._char_embedder = CharFreqEmbedder(dim=CharFreqEmbedder.FIXED_DIM)

        if corpus:
            self._store.add_batch(corpus)

    # ── PipelineConnector interface ──────────────────────────────────────────

    def embed(self, text: str) -> List[float]:
        """
        Returns a fixed-dim (256) char-bigram embedding.
        For retrieval the VectorStore uses its own (better) embedder internally.
        """
        return self._char_embedder.embed(text)

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        results = self._store.search(query, top_k=top_k)

        if self.inject_position_bias and len(results) >= 3:
            best = results.pop(0)
            mid = len(results) // 2
            results.insert(mid, best)
            for i, doc in enumerate(results):
                doc.position = i

        return results

    def generate(self, query: str, docs: List[Document]) -> str:
        if self._answer_fn:
            return self._answer_fn(query, docs)

        if self.inject_hallucination:
            return (
                "This feature supports advanced quantum entanglement processing "
                "with the --quantum flag."
            )

        if not docs:
            return "I don't have enough information to answer."

        # Extractive: first complete sentence of top doc
        best = docs[0].content
        sentences = [s.strip() for s in best.split(".") if s.strip()]
        return (sentences[0] + ".") if sentences else best[:100]

    # ── Convenience ─────────────────────────────────────────────────────────

    def add_document(self, content: str, metadata: Optional[Dict] = None,
                     doc_id: Optional[str] = None) -> None:
        self._store.add(content, metadata=metadata, doc_id=doc_id)

    def load_corpus(self, corpus: List[Dict]) -> None:
        self._store.clear()
        self._store.add_batch(corpus)

    def __len__(self) -> int:
        return len(self._store)
