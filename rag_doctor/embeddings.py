"""
Lightweight embedding utilities using TF-IDF cosine similarity.
No external API needed for testing. Swap embed() for OpenAI/Cohere in production.
"""

from __future__ import annotations
import math
import re
from collections import Counter
from typing import List


def tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    text = text.lower()
    tokens = re.findall(r'\b[a-z][a-z0-9]*\b', text)
    return tokens


def tfidf_vector(tokens: List[str], vocab: dict) -> List[float]:
    """Build a simple TF vector aligned to vocab."""
    counts = Counter(tokens)
    total = max(len(tokens), 1)
    return [counts.get(w, 0) / total for w in vocab]


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return min(dot / (mag_a * mag_b), 1.0)


def text_similarity(text_a: str, text_b: str) -> float:
    """Compute semantic similarity between two texts using TF-IDF cosine."""
    tok_a = tokenize(text_a)
    tok_b = tokenize(text_b)
    if not tok_a or not tok_b:
        return 0.0
    vocab = {w: i for i, w in enumerate(sorted(set(tok_a + tok_b)))}
    vec_a = tfidf_vector(tok_a, vocab)
    vec_b = tfidf_vector(tok_b, vocab)
    return cosine_similarity(vec_a, vec_b)


def intra_chunk_coherence(text: str) -> float:
    """
    Measure coherence within a chunk by averaging pairwise similarity
    between consecutive sentences.
    """
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]
    if len(sentences) < 2:
        return 1.0  # single sentence = perfectly coherent
    scores = []
    for i in range(len(sentences) - 1):
        scores.append(text_similarity(sentences[i], sentences[i + 1]))
    return sum(scores) / len(scores)


def split_into_claims(text: str) -> List[str]:
    """Split answer text into individual atomic claims (sentences)."""
    claims = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 15]
    return claims
