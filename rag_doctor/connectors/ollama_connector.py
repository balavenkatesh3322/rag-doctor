"""
Ollama connector for rag-doctor.

Connects rag-doctor to a local Ollama instance.
Supports model selection, embedding generation, and RAG generation.

Usage:
    from rag_doctor.connectors.ollama_connector import OllamaConnector
    connector = OllamaConnector(model="llama3.2")
    doctor = Doctor.default(connector=connector)
"""

from __future__ import annotations

import json
import math
import urllib.request
import urllib.error
from typing import List, Optional, Dict, Any

from .base import Document, PipelineConnector


# ─── Ollama API helpers ────────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"

# Models ranked by capability tier for RAG tasks.
# Tiers: 3=best, 2=good, 1=fast/small
RECOMMENDED_MODELS = {
    "tier3": ["llama3.1:8b", "llama3.2:3b", "mistral:7b", "mixtral:8x7b"],
    "tier2": ["llama3.2:1b", "phi3:medium", "gemma2:9b", "qwen2.5:7b"],
    "tier1": ["phi3:mini", "gemma2:2b", "qwen2.5:1.5b", "tinyllama"],
}

EMBED_MODELS = ["nomic-embed-text", "mxbai-embed-large", "all-minilm"]


def _http_post(url: str, payload: dict, timeout: int = 60) -> dict:
    """Make a POST request to Ollama API."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}. "
            f"Is Ollama running? Start it with: ollama serve\n"
            f"Error: {e}"
        ) from e


def _http_get(url: str, timeout: int = 10) -> dict:
    """Make a GET request to Ollama API."""
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}. "
            f"Is Ollama running? Start it with: ollama serve\n"
            f"Error: {e}"
        ) from e


def list_installed_models() -> List[str]:
    """Return list of model names installed in Ollama."""
    try:
        data = _http_get(f"{OLLAMA_BASE_URL}/api/tags")
        return [m["name"] for m in data.get("models", [])]
    except ConnectionError:
        return []


def select_best_model(installed: List[str]) -> Optional[str]:
    """
    Pick the best available model for RAG tasks from installed list.
    Priority: tier3 > tier2 > tier1 > any installed model.
    """
    installed_lower = {m.lower(): m for m in installed}

    for tier in ["tier3", "tier2", "tier1"]:
        for preferred in RECOMMENDED_MODELS[tier]:
            # Exact match
            if preferred in installed:
                return preferred
            # Prefix match (e.g. "llama3.2" matches "llama3.2:latest")
            for name_lower, name_orig in installed_lower.items():
                if name_lower.startswith(preferred.split(":")[0]):
                    return name_orig

    # Fall back to first installed model
    return installed[0] if installed else None


def select_best_embed_model(installed: List[str]) -> Optional[str]:
    """Pick best embedding model, fall back to generation model for embeddings."""
    for preferred in EMBED_MODELS:
        for name in installed:
            if preferred in name.lower():
                return name
    return None


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


def _simple_embed(text: str, dim: int = 64) -> List[float]:
    """Fallback char-frequency embedding when no embed model available."""
    vec = [0.0] * dim
    for ch in text.lower():
        vec[ord(ch) % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


# ─── OllamaConnector ──────────────────────────────────────────────────────────

class OllamaConnector(PipelineConnector):
    """
    Connects rag-doctor to a local Ollama instance.

    Args:
        model:       Ollama model name for generation (e.g. "llama3.2").
                     If None, auto-selects the best installed model.
        embed_model: Ollama model for embeddings (e.g. "nomic-embed-text").
                     If None, uses generation model or char-frequency fallback.
        corpus:      List of dicts with "content" and optional "metadata", "id".
        top_k:       Default number of documents to retrieve.
        temperature: LLM generation temperature (0.0 = deterministic).
        base_url:    Ollama server URL (default: http://localhost:11434).
    """

    def __init__(
        self,
        model: Optional[str] = None,
        embed_model: Optional[str] = None,
        corpus: Optional[List[Dict]] = None,
        top_k: int = 5,
        temperature: float = 0.1,
        base_url: str = OLLAMA_BASE_URL,
    ):
        global OLLAMA_BASE_URL
        OLLAMA_BASE_URL = base_url

        self.temperature = temperature
        self.top_k = top_k
        self._corpus: List[Document] = []
        self._embeddings: List[List[float]] = []  # cached embeddings

        # Auto-select model if not specified
        installed = list_installed_models()
        if not installed:
            raise RuntimeError(
                "No models found in Ollama. Install one:\n"
                "  ollama pull llama3.2\n"
                "  ollama pull mistral\n"
            )

        self.model = model or select_best_model(installed)
        self.embed_model = embed_model or select_best_embed_model(installed)

        print(f"  rag-doctor OllamaConnector")
        print(f"  Generation model : {self.model}")
        print(f"  Embedding model  : {self.embed_model or 'char-frequency fallback'}")

        if corpus:
            self.load_corpus(corpus)

    def load_corpus(self, corpus: List[Dict]) -> None:
        """Load documents into in-memory store and pre-embed them."""
        print(f"  Embedding {len(corpus)} documents...", end="", flush=True)
        self._corpus = []
        self._embeddings = []
        for i, item in enumerate(corpus):
            doc = Document(
                content=item["content"],
                metadata=item.get("metadata", {}),
                doc_id=item.get("id", f"doc_{i}"),
            )
            self._corpus.append(doc)
            self._embeddings.append(self.embed(item["content"]))
            if (i + 1) % 5 == 0:
                print(".", end="", flush=True)
        print(f" done ({len(self._corpus)} docs)")

    def add_document(self, content: str, metadata: dict = None, doc_id: str = None) -> None:
        """Add a single document to the corpus."""
        idx = len(self._corpus)
        doc = Document(
            content=content,
            metadata=metadata or {},
            doc_id=doc_id or f"doc_{idx}",
        )
        self._corpus.append(doc)
        self._embeddings.append(self.embed(content))

    def embed(self, text: str) -> List[float]:
        """Generate embedding using Ollama embed API, or fall back to char-frequency."""
        if self.embed_model:
            try:
                result = _http_post(
                    f"{OLLAMA_BASE_URL}/api/embed",
                    {"model": self.embed_model, "input": text},
                    timeout=30,
                )
                embeddings = result.get("embeddings", [[]])
                if embeddings and embeddings[0]:
                    return embeddings[0]
            except Exception:
                pass  # fall through to fallback

        # Legacy /api/embeddings endpoint
        if self.embed_model:
            try:
                result = _http_post(
                    f"{OLLAMA_BASE_URL}/api/embeddings",
                    {"model": self.embed_model, "prompt": text},
                    timeout=30,
                )
                if result.get("embedding"):
                    return result["embedding"]
            except Exception:
                pass

        # Final fallback: char-frequency embedding
        return _simple_embed(text)

    def retrieve(self, query: str, top_k: int = None) -> List[Document]:
        """Retrieve top-k documents by embedding similarity."""
        k = top_k or self.top_k
        if not self._corpus:
            return []

        q_vec = self.embed(query)
        scored = []
        for doc, doc_vec in zip(self._corpus, self._embeddings):
            score = _cosine(q_vec, doc_vec)
            scored.append((score, doc))

        scored.sort(key=lambda x: -x[0])
        results = []
        for pos, (score, doc) in enumerate(scored[:k]):
            results.append(Document(
                content=doc.content,
                metadata=doc.metadata,
                score=score,
                position=pos,
                doc_id=doc.doc_id,
            ))
        return results

    def generate(self, query: str, docs: List[Document]) -> str:
        """Generate an answer using Ollama given query and retrieved docs."""
        if not docs:
            return "I don't have enough information to answer this question."

        context = "\n\n".join(
            f"[Document {i+1}]\n{doc.content}"
            for i, doc in enumerate(docs)
        )

        prompt = (
            f"You are a helpful assistant. Answer the question using ONLY the provided context. "
            f"If the context doesn't contain the answer, say 'I don't know based on the provided context.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        result = _http_post(
            f"{OLLAMA_BASE_URL}/api/generate",
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": self.temperature, "num_predict": 256},
            },
            timeout=120,
        )
        return result.get("response", "").strip()
