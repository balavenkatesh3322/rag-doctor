"""
Embedding engine for rag-doctor.

Priority chain (auto-selected at runtime):
  1. sentence-transformers  — best quality, fully local, no API key (~90MB download)
  2. Ollama embed API       — good quality, requires: ollama serve + ollama pull nomic-embed-text
  3. TF-IDF sparse vectors  — decent keyword retrieval, stdlib + numpy, needs corpus to fit
  4. Char n-gram fallback   — always available, stdlib only, poor semantic quality

Install the best backend:
    pip install sentence-transformers      # recommended
    ollama pull nomic-embed-text           # alternative
"""

from __future__ import annotations

import math
import re
from typing import List, Optional, Dict
from abc import ABC, abstractmethod


# ─────────────────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────────────────

class BaseEmbedder(ABC):
    name: str = "base"

    @abstractmethod
    def embed(self, text: str) -> List[float]: ...

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]

    def similarity(self, a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na  = math.sqrt(sum(x * x for x in a)) or 1e-9
        nb  = math.sqrt(sum(x * x for x in b)) or 1e-9
        return max(0.0, min(1.0, dot / (na * nb)))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Sentence-Transformers
# ─────────────────────────────────────────────────────────────────────────────

class SentenceTransformerEmbedder(BaseEmbedder):
    """
    High-quality semantic embeddings. Fully local, no API key.
    First use downloads the model (~90MB).
    Install: pip install sentence-transformers
    """
    name = "sentence-transformers"
    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [v.tolist() for v in
                self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Ollama Embeddings
# ─────────────────────────────────────────────────────────────────────────────

class OllamaEmbedder(BaseEmbedder):
    """
    Embeddings via local Ollama server.
    Requires: ollama serve + ollama pull nomic-embed-text
    """
    name = "ollama"
    PREFERRED_MODELS = ["nomic-embed-text", "mxbai-embed-large", "all-minilm"]

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def embed(self, text: str) -> List[float]:
        import json, urllib.request
        for endpoint, key, payload in [
            ("/api/embed",      "embeddings", {"model": self.model, "input": text}),
            ("/api/embeddings", "embedding",  {"model": self.model, "prompt": text}),
        ]:
            try:
                data = json.dumps(payload).encode()
                req = urllib.request.Request(
                    f"{self.base_url}{endpoint}", data=data,
                    headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=30) as r:
                    result = json.loads(r.read())
                raw = result.get(key)
                if raw:
                    return raw[0] if isinstance(raw[0], list) else raw
            except Exception:
                continue
        raise RuntimeError(f"Ollama embedding failed for model '{self.model}'")

    @classmethod
    def best_available(cls, base_url: str = "http://localhost:11434") -> Optional["OllamaEmbedder"]:
        import json, urllib.request
        try:
            with urllib.request.urlopen(f"{base_url}/api/tags", timeout=3) as r:
                models = [m["name"] for m in json.loads(r.read()).get("models", [])]
        except Exception:
            return None
        for preferred in cls.PREFERRED_MODELS:
            for installed in models:
                if preferred in installed.lower():
                    e = cls(model=installed, base_url=base_url)
                    try:
                        e.embed("test")
                        return e
                    except Exception:
                        continue
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. TF-IDF (stdlib + numpy, must be corpus-fitted before use)
# ─────────────────────────────────────────────────────────────────────────────

class TFIDFEmbedder(BaseEmbedder):
    """
    TF-IDF sparse embeddings using numpy only.

    IMPORTANT: Must be fit() on the full corpus BEFORE calling embed().
    VectorStore handles this automatically when you use add_batch().
    Calling embed() before fit() falls back to char-freq for that text.
    """
    name = "tfidf"

    def __init__(self, dim: int = 4096):
        self._dim = dim
        self._vocab: Dict[str, int] = {}
        self._idf: List[float] = []
        self._fitted = False

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b[a-z]{2,}\b', text.lower())

    def fit(self, corpus: List[str]) -> "TFIDFEmbedder":
        """Build vocabulary and IDF weights. Call this on the ENTIRE corpus at once."""
        token_doc_freq: Dict[str, int] = {}
        token_freq: Dict[str, int] = {}
        for doc in corpus:
            seen = set()
            for t in self._tokenize(doc):
                token_freq[t] = token_freq.get(t, 0) + 1
                if t not in seen:
                    token_doc_freq[t] = token_doc_freq.get(t, 0) + 1
                    seen.add(t)

        # Keep top-dim most frequent tokens
        top_tokens = sorted(token_freq, key=lambda t: -token_freq[t])[:self._dim]
        self._vocab = {t: i for i, t in enumerate(top_tokens)}

        n = len(corpus)
        self._idf = [
            math.log((n + 1) / (token_doc_freq.get(t, 0) + 1)) + 1
            for t in top_tokens
        ]
        self._fitted = True
        return self

    def embed(self, text: str) -> List[float]:
        if not self._fitted:
            # Fallback: char n-gram so we don't crash (result is low quality)
            return CharFreqEmbedder(dim=self._dim).embed(text)

        tokens = self._tokenize(text)
        tf: Dict[str, float] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        n = len(tokens) or 1
        for t in tf:
            tf[t] /= n

        vec = [0.0] * len(self._vocab)
        for t, freq in tf.items():
            if t in self._vocab:
                idx = self._vocab[t]
                vec[idx] = freq * self._idf[idx]

        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Char n-gram fallback (always available)
# ─────────────────────────────────────────────────────────────────────────────

class CharFreqEmbedder(BaseEmbedder):
    """
    Character bigram frequency embedding.
    stdlib only, zero dependencies, always available.
    Poor semantic quality — use only as last resort.
    Fixed output dimension regardless of input.
    """
    name = "char-frequency"
    FIXED_DIM = 256  # fixed so tests can assert on dim

    def __init__(self, dim: int = FIXED_DIM, ngram: int = 2):
        self._dim = dim
        self._ngram = ngram

    def embed(self, text: str) -> List[float]:
        text = text.lower()
        vec = [0.0] * self._dim
        for ch in text:
            vec[ord(ch) % self._dim] += 1.0
        if self._ngram >= 2:
            for i in range(len(text) - 1):
                h = (ord(text[i]) * 31 + ord(text[i+1])) % self._dim
                vec[h] += 0.5
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


# ─────────────────────────────────────────────────────────────────────────────
# Auto-selector
# ─────────────────────────────────────────────────────────────────────────────

_EMBEDDER_CACHE: Optional[BaseEmbedder] = None


def get_embedder(
    prefer: Optional[str] = None,
    ollama_url: str = "http://localhost:11434",
    quiet: bool = False,
) -> BaseEmbedder:
    """
    Return the best available embedder.

    Priority: sentence-transformers → Ollama → TF-IDF → char-frequency

    NOTE: TF-IDF returned here is UNFITTED. VectorStore.add_batch() fits it
    on the corpus automatically. If you use TF-IDF outside VectorStore,
    call embedder.fit(corpus_texts) before embed().

    Args:
        prefer: Force a backend: "sentence-transformers"|"ollama"|"tfidf"|"char"
        ollama_url: Ollama server URL.
        quiet: Suppress selection message.
    """
    global _EMBEDDER_CACHE
    if _EMBEDDER_CACHE and not prefer:
        return _EMBEDDER_CACHE

    def _log(msg):
        if not quiet:
            print(f"  [embedder] {msg}")

    # 1. sentence-transformers
    if prefer in (None, "sentence-transformers"):
        try:
            e = SentenceTransformerEmbedder()
            _log(f"✓ sentence-transformers ({e.model_name})")
            if not prefer:
                _EMBEDDER_CACHE = e
            return e
        except ImportError:
            if prefer == "sentence-transformers":
                raise RuntimeError("pip install sentence-transformers")
            _log("sentence-transformers not installed → trying Ollama")

    # 2. Ollama embed
    if prefer in (None, "ollama"):
        e = OllamaEmbedder.best_available(base_url=ollama_url)
        if e:
            _log(f"✓ Ollama ({e.model})")
            if not prefer:
                _EMBEDDER_CACHE = e
            return e
        elif prefer == "ollama":
            raise RuntimeError("ollama pull nomic-embed-text")
        else:
            _log("No Ollama embed model → trying TF-IDF")

    # 3. TF-IDF (returned unfitted; VectorStore.add_batch fits it)
    if prefer in (None, "tfidf"):
        try:
            import numpy  # noqa
            e = TFIDFEmbedder()
            _log("✓ TF-IDF (will fit on corpus via VectorStore)")
            if not prefer:
                _EMBEDDER_CACHE = e
            return e
        except ImportError:
            pass

    # TF-IDF without numpy: pure-python TF-IDF is fine since we don't use numpy
    if prefer in (None, "tfidf"):
        e = TFIDFEmbedder()
        _log("✓ TF-IDF (pure-python)")
        if not prefer:
            _EMBEDDER_CACHE = e
        return e

    # 4. Char fallback
    _log("⚠  char-frequency fallback (install sentence-transformers for semantic search)")
    e = CharFreqEmbedder()
    if not prefer:
        _EMBEDDER_CACHE = e
    return e


def reset_embedder_cache() -> None:
    global _EMBEDDER_CACHE
    _EMBEDDER_CACHE = None
