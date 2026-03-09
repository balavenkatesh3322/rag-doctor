"""
OllamaConnector — connects rag-doctor to a local Ollama instance.

Uses the real embedding engine (sentence-transformers → Ollama embed → TF-IDF)
for retrieval, and Ollama's generate API for answer generation.

Install Ollama on Mac:
    brew install ollama
    ollama serve
    ollama pull llama3.2
    ollama pull nomic-embed-text   # optional but recommended for better retrieval
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import List, Optional, Dict

from .base import Document, PipelineConnector
from ..vector_store import VectorStore
from ..embeddings import get_embedder, OllamaEmbedder, BaseEmbedder

OLLAMA_BASE_URL = "http://localhost:11434"

# Models ranked by RAG suitability
RECOMMENDED_MODELS = {
    "tier3": ["llama3.1:8b", "llama3.2:3b", "mistral:7b", "gemma2:9b"],
    "tier2": ["llama3.2:1b", "phi3:medium", "qwen2.5:7b"],
    "tier1": ["phi3:mini", "gemma2:2b", "qwen2.5:1.5b", "tinyllama"],
}


def _post(url: str, payload: dict, timeout: int = 120) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Cannot reach Ollama at {url}.\n"
            f"Make sure Ollama is running: ollama serve\n"
            f"Error: {e}"
        ) from e


def _get(url: str, timeout: int = 5) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        raise ConnectionError(f"Cannot reach {url}: {e}") from e


def list_installed_models(base_url: str = OLLAMA_BASE_URL) -> List[str]:
    try:
        data = _get(f"{base_url}/api/tags")
        return [m["name"] for m in data.get("models", [])]
    except ConnectionError:
        return []


def select_best_generation_model(installed: List[str]) -> Optional[str]:
    installed_lower = {m.lower(): m for m in installed}
    for tier in ["tier3", "tier2", "tier1"]:
        for preferred in RECOMMENDED_MODELS[tier]:
            if preferred in installed:
                return preferred
            stem = preferred.split(":")[0]
            for name_lower, name_orig in installed_lower.items():
                if name_lower.startswith(stem):
                    return name_orig
    return installed[0] if installed else None


class OllamaConnector(PipelineConnector):
    """
    Production-quality connector using Ollama for generation
    and the best available embedder for retrieval.

    Args:
        model:        Ollama model for generation. Auto-selected if None.
        embedder:     Embedding backend. Auto-selected if None.
                      Priority: sentence-transformers > nomic-embed-text > TF-IDF
        corpus:       Initial corpus to load.
        top_k:        Default retrieval depth.
        temperature:  LLM temperature (0 = deterministic).
        base_url:     Ollama server URL.
        quiet:        Suppress startup messages.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        embedder: Optional[BaseEmbedder] = None,
        corpus: Optional[List[Dict]] = None,
        top_k: int = 5,
        temperature: float = 0.1,
        base_url: str = OLLAMA_BASE_URL,
        quiet: bool = False,
    ):
        self.base_url = base_url
        self.temperature = temperature
        self.top_k = top_k

        # Select generation model
        installed = list_installed_models(base_url)
        if not installed:
            raise RuntimeError(
                "No Ollama models found.\n"
                "Install one: ollama pull llama3.2"
            )

        self.model = model or select_best_generation_model(installed)

        # Select embedder
        # Prefer: sentence-transformers → Ollama nomic-embed-text → TF-IDF
        if embedder:
            _embedder = embedder
        else:
            # Try Ollama embed model first if no sentence-transformers
            ollama_embed = OllamaEmbedder.best_available(base_url=base_url)
            _embedder = get_embedder(
                ollama_url=base_url,
                quiet=quiet,
            )

        self._store = VectorStore(embedder=_embedder, quiet=quiet)

        if not quiet:
            print(f"  OllamaConnector ready")
            print(f"    Generation : {self.model}")
            print(f"    Embeddings : {self._store.embedder.name}")

        if corpus:
            self.load_corpus(corpus)

    # ── PipelineConnector interface ──────────────────────────────────────────

    def embed(self, text: str) -> List[float]:
        return self._store.embedder.embed(text)

    def retrieve(self, query: str, top_k: int = None) -> List[Document]:
        return self._store.search(query, top_k=top_k or self.top_k)

    def generate(self, query: str, docs: List[Document]) -> str:
        if not docs:
            return "I don't have enough context to answer this question."

        context = "\n\n".join(
            f"[Source {i+1}]\n{doc.content}"
            for i, doc in enumerate(docs)
        )

        prompt = (
            "You are a precise assistant. Answer the question using ONLY the provided sources. "
            "If the sources don't contain the answer, say exactly: "
            "'I cannot find this information in the provided context.'\n\n"
            f"Sources:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        result = _post(
            f"{self.base_url}/api/generate",
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": 300,
                    "stop": ["\n\n", "Question:", "Sources:"],
                },
            },
            timeout=120,
        )
        return result.get("response", "").strip()

    # ── Corpus management ────────────────────────────────────────────────────

    def load_corpus(self, corpus: List[Dict], show_progress: bool = True) -> None:
        """Load and embed a corpus of documents."""
        if show_progress:
            print(f"  Embedding {len(corpus)} documents...", end="", flush=True)
        self._store.clear()
        self._store.add_batch(corpus)
        if show_progress:
            print(f" ✓")

    def add_document(
        self,
        content: str,
        metadata: Optional[Dict] = None,
        doc_id: Optional[str] = None,
    ) -> None:
        self._store.add(content, metadata=metadata, doc_id=doc_id)

    def __len__(self) -> int:
        return len(self._store)
