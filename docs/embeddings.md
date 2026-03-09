# Embedding Backends

rag-doctor auto-selects the best available embedding backend. No configuration required.

---

## The Four-Level Chain

```
sentence-transformers  →  Ollama  →  TF-IDF  →  char n-gram fallback
     (best)                                          (always available)
```

Each level is tried in order. The first available backend is used.

---

## Level 1: sentence-transformers

**Best quality. Recommended for production use.**

```bash
pip install sentence-transformers
```

- Downloads `all-MiniLM-L6-v2` (~90MB) on first use
- All subsequent uses are fully local and offline
- 384-dimensional embeddings
- True semantic similarity: related sentences score 0.80–0.99
- Recommended thresholds: `recall=0.65`, `faithfulness=0.65`, `coherence=0.55`

---

## Level 2: Ollama

**Excellent quality. Best for teams already using Ollama locally.**

```bash
# Install Ollama: https://ollama.ai
ollama pull nomic-embed-text
```

- Requires Ollama running on `localhost:11434`
- 768-dimensional embeddings
- Recommended thresholds: same as sentence-transformers

---

## Level 3: TF-IDF

**Good quality for keyword-heavy domains. Zero dependencies beyond numpy.**

- Automatically used when sentence-transformers and Ollama are unavailable
- Fitted per `VectorStore` instance (no shared vocabulary across test cases)
- Works well for: legal documents, medical codes, product SKUs, policy documents
- Similarity ceiling for paraphrases: ~0.60
- Recommended thresholds: `recall=0.35`, `faithfulness=0.40`, `coherence=0.25`

---

## Level 4: Char n-gram Fallback

**Always available. Structural checks still valid.**

- Used when nothing else is available (no numpy, no sentence-transformers, no Ollama)
- 256-dimensional fixed-size embeddings
- Similarity scores are not semantically meaningful
- Truncation detection, position analysis, and chunk size checks remain fully valid

---

## Threshold Guide

Choose thresholds based on your embedding backend:

| Threshold | TF-IDF | sentence-transformers / Ollama |
|---|---|---|
| `recall_threshold` | 0.35 | 0.65 |
| `faithfulness_threshold` | 0.40 | 0.65 |
| `coherence_threshold` | 0.25 | 0.55 |

```yaml
# rag-doctor.yaml — for sentence-transformers
diagnosis:
  recall_threshold: 0.65
  faithfulness_threshold: 0.65
  coherence_threshold: 0.55
```

---

## Forcing a Specific Backend

```bash
# Environment variable
export RAGDOCTOR_EMBEDDER=sentence-transformers  # or: ollama, tfidf, char
```

```python
# In Python
import os
os.environ["RAGDOCTOR_EMBEDDER"] = "tfidf"  # force TF-IDF for CI
from rag_doctor import Doctor
```

---

## Upgrade Path

```
Week 1: Install rag-doctor only
→ TF-IDF runs automatically
→ Catches structural issues: truncation, position bias, vocabulary mismatch
→ Thresholds: recall=0.35, faithfulness=0.40, coherence=0.25

Week 2: pip install sentence-transformers
→ Semantic similarity becomes meaningful
→ Faithfulness detection improves significantly
→ Raise thresholds: recall=0.65, faithfulness=0.65, coherence=0.55

Week 3+: ollama pull nomic-embed-text llama3.2
→ Full local pipeline testing (retrieval + generation)
→ Test realistic hallucination scenarios
→ No cloud API costs
```

---

## VectorStore Isolation

Each `VectorStore` instance owns its own TF-IDF vocabulary. This ensures:

- Two test cases with different corpora never share embedding state
- Tests are fully isolated even when run in parallel
- A CI run with a medical corpus doesn't affect another with a legal corpus

```python
# These are independent — different vocabularies, no interference
store1 = VectorStore()
store1.add_batch(medical_docs)

store2 = VectorStore()
store2.add_batch(legal_docs)
```
