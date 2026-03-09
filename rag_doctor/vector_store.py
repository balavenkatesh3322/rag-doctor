"""
In-memory vector store with real semantic retrieval.

Key design decisions:
- Each VectorStore owns its own TFIDFEmbedder instance (never shared)
  so corpus-specific vocabulary is always correct.
- sentence-transformers and OllamaEmbedder ARE shared via cache (no corpus state).
- add_batch() fits TF-IDF on the FULL corpus at once before any embed() call.
- All similarity uses the embedder's cosine similarity method.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

from .connectors.base import Document
from .embeddings import (
    BaseEmbedder, TFIDFEmbedder,
    SentenceTransformerEmbedder, OllamaEmbedder,
    get_embedder,
)


@dataclass
class _Entry:
    doc: Document
    embedding: List[float]


class VectorStore:
    """
    In-memory vector store.

    TF-IDF note: each VectorStore creates its own TFIDFEmbedder instance
    so different connectors/tests never share vocabulary state.
    """

    def __init__(
        self,
        embedder: Optional[BaseEmbedder] = None,
        ollama_url: str = "http://localhost:11434",
        quiet: bool = True,
    ):
        self._entries: List[_Entry] = []
        self._ollama_url = ollama_url
        self._quiet = quiet

        # If caller passed an embedder, use it directly.
        # If it's TF-IDF, make a fresh copy so we own the vocabulary.
        if embedder is not None:
            if isinstance(embedder, TFIDFEmbedder):
                self._embedder: Optional[BaseEmbedder] = TFIDFEmbedder(dim=embedder._dim)
            else:
                self._embedder = embedder
        else:
            self._embedder = None

    def _get_or_create_embedder(self, corpus_texts: Optional[List[str]] = None) -> BaseEmbedder:
        """
        Return the embedder, initialising it if needed.
        For TF-IDF: always fits on corpus_texts if provided and not yet fitted.
        For sentence-transformers/Ollama: uses the global cache (no corpus state).
        """
        if self._embedder is None:
            # Check global cache first — but only reuse stateless embedders
            cached = get_embedder(ollama_url=self._ollama_url, quiet=self._quiet)
            if isinstance(cached, TFIDFEmbedder):
                # Don't reuse global TF-IDF — create a fresh instance for this store
                self._embedder = TFIDFEmbedder(dim=cached._dim)
            else:
                # sentence-transformers and Ollama are stateless — safe to share
                self._embedder = cached

        # Fit TF-IDF if needed
        if isinstance(self._embedder, TFIDFEmbedder) and not self._embedder._fitted:
            if corpus_texts:
                self._embedder.fit(corpus_texts)

        return self._embedder

    @property
    def embedder(self) -> BaseEmbedder:
        return self._get_or_create_embedder()

    def add_batch(self, items: List[Dict]) -> None:
        """
        Add documents. TF-IDF is fitted on all texts at once before embedding.
        items: [{"content": str, "id": str (opt), "metadata": dict (opt)}]
        """
        if not items:
            return

        texts = [item["content"] for item in items]
        embedder = self._get_or_create_embedder(corpus_texts=texts)

        try:
            embeddings = embedder.embed_batch(texts)
        except Exception:
            embeddings = [embedder.embed(t) for t in texts]

        for i, (item, emb) in enumerate(zip(items, embeddings)):
            idx = len(self._entries)
            doc = Document(
                content=item["content"],
                metadata=item.get("metadata", {}),
                doc_id=item.get("id", f"doc_{idx}"),
            )
            self._entries.append(_Entry(doc=doc, embedding=emb))

    def add(self, content: str, metadata: Optional[Dict] = None,
            doc_id: Optional[str] = None) -> None:
        """Add a single document."""
        idx = len(self._entries)

        # For TF-IDF: re-fit on all texts including new one
        if isinstance(self._embedder, TFIDFEmbedder) or self._embedder is None:
            all_texts = [e.doc.content for e in self._entries] + [content]
            embedder = self._get_or_create_embedder(corpus_texts=all_texts)
            for entry in self._entries:
                entry.embedding = embedder.embed(entry.doc.content)
        else:
            embedder = self._get_or_create_embedder()

        doc = Document(
            content=content,
            metadata=metadata or {},
            doc_id=doc_id or f"doc_{idx}",
        )
        self._entries.append(_Entry(doc=doc, embedding=embedder.embed(content)))

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        if not self._entries:
            return []

        embedder = self._get_or_create_embedder()
        q_vec = embedder.embed(query)

        scored: List[Tuple[float, _Entry]] = [
            (embedder.similarity(q_vec, e.embedding), e)
            for e in self._entries
        ]
        scored.sort(key=lambda x: -x[0])

        return [
            Document(
                content=e.doc.content,
                metadata=e.doc.metadata,
                score=score,
                position=pos,
                doc_id=e.doc.doc_id,
            )
            for pos, (score, e) in enumerate(scored[:top_k])
        ]

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        if not docs:
            return docs
        embedder = self._get_or_create_embedder()
        q_vec = embedder.embed(query)
        scored = sorted(
            [(embedder.similarity(q_vec, embedder.embed(d.content)), d) for d in docs],
            key=lambda x: -x[0],
        )
        return [
            Document(content=d.content, metadata=d.metadata,
                     score=s, position=i, doc_id=d.doc_id)
            for i, (s, d) in enumerate(scored)
        ]

    def clear(self) -> None:
        self._entries.clear()
        if isinstance(self._embedder, TFIDFEmbedder):
            self._embedder._fitted = False
            self._embedder._vocab = {}
            self._embedder._idf = []

    def __len__(self) -> int:
        return len(self._entries)
