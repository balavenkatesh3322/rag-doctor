"""Abstract connector interface."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Document:
    """A retrieved document chunk."""
    content: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0
    position: int = 0
    doc_id: Optional[str] = None

    def __repr__(self):
        preview = self.content[:60].replace("\n", " ")
        return f"Document(pos={self.position}, score={self.score:.3f}, content='{preview}...')"


class PipelineConnector(ABC):
    """Abstract base for all RAG pipeline connectors."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve top-k documents for query."""
        ...

    @abstractmethod
    def generate(self, query: str, docs: List[Document]) -> str:
        """Generate answer given query and retrieved docs."""
        ...

    def embed(self, text: str) -> List[float]:
        """Embed text. Override for custom embeddings."""
        raise NotImplementedError
