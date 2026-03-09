"""
Shared embedding utility for tools.

Tools must never call get_embedder() and then embed() sentence-by-sentence
because TF-IDF needs to be fitted on ALL texts it will compare.
Use fit_and_embed() which handles this correctly for every backend.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
from ..embeddings import (
    BaseEmbedder, TFIDFEmbedder, get_embedder
)


def fit_and_embed(
    texts: List[str],
    embedder: Optional[BaseEmbedder] = None,
) -> Tuple[BaseEmbedder, List[List[float]]]:
    """
    Embed a list of texts correctly for any backend.

    For TF-IDF: fits on all texts together so vocabulary covers everything.
    For sentence-transformers/Ollama: uses the shared instance, batch-embeds.

    Returns (embedder, list_of_vectors).
    """
    if not texts:
        e = embedder or get_embedder(quiet=True)
        return e, []

    if embedder is None:
        cached = get_embedder(quiet=True)
        if isinstance(cached, TFIDFEmbedder):
            # Create a fresh per-call TFIDFEmbedder so we don't pollute global state
            e: BaseEmbedder = TFIDFEmbedder(dim=cached._dim)
        else:
            e = cached
    else:
        e = embedder

    # For TF-IDF: fit on all provided texts at once
    if isinstance(e, TFIDFEmbedder) and not e._fitted:
        e.fit(texts)

    try:
        vecs = e.embed_batch(texts)
    except Exception:
        vecs = [e.embed(t) for t in texts]

    return e, vecs


def similarity(e: BaseEmbedder, a: List[float], b: List[float]) -> float:
    return e.similarity(a, b)
