# Configuration Reference

rag-doctor works with zero configuration. Create `rag-doctor.yaml` in your project root to customize thresholds and pipeline settings.

---

## Full Configuration File

```yaml
# rag-doctor.yaml

# Pipeline metadata (informational only)
pipeline_type: langchain          # langchain | llamaindex | custom
vector_db: chroma                 # chroma | pinecone | weaviate | pgvector
embedding_model: all-MiniLM-L6-v2
llm: gpt-4o

# Retrieval settings
retrieval:
  top_k: 5
  reranker: false
  hybrid_search: false
  bm25_weight: 0.3               # weight for BM25 in hybrid search

# Chunking settings  
chunking:
  strategy: recursive             # fixed | recursive | semantic | hierarchical
  chunk_size: 256
  chunk_overlap: 32

# Diagnosis thresholds
diagnosis:
  # For TF-IDF (default): recall=0.35, faithfulness=0.40, coherence=0.25
  # For sentence-transformers/Ollama: recall=0.65, faithfulness=0.65, coherence=0.55
  recall_threshold: 0.35
  faithfulness_threshold: 0.40
  coherence_threshold: 0.25
  severity_threshold: medium      # minimum severity to report as issue
```

---

## Diagnosis Thresholds

### `recall_threshold` (default: `0.35`)

Minimum similarity score between the expected answer and retrieved documents.

- Below this: `retrieval_miss` (RC-1)
- Raise to `0.65` when using sentence-transformers
- Lower to `0.20` for very short documents or sparse domains

### `faithfulness_threshold` (default: `0.40`)

Minimum similarity between an answer claim and its best matching source sentence.

- Below this: the claim is considered hallucinated
- Raise to `0.65` when using sentence-transformers
- If too many false positives: raise threshold slightly

### `coherence_threshold` (default: `0.25`)

Minimum average cosine similarity between adjacent sentences in a chunk.

- Below this: `chunk_fragmentation` (RC-3)
- Raise to `0.55` when using sentence-transformers
- Lower to `0.15` for naturally diverse technical documentation

---

## Loading Config in Python

```python
from rag_doctor import Doctor, RagDoctorConfig

# From file
cfg = RagDoctorConfig.from_yaml("rag-doctor.yaml")
doctor = Doctor(cfg)

# Default config
doctor = Doctor.default()

# Modify defaults programmatically
cfg = RagDoctorConfig.default()
cfg.diagnosis.recall_threshold = 0.65       # after installing sentence-transformers
cfg.diagnosis.faithfulness_threshold = 0.65
cfg.diagnosis.coherence_threshold = 0.55
doctor = Doctor(cfg)
```

---

## Config Patches in Reports

When rag-doctor finds an issue, the report includes a `config_patch` with the recommended config changes:

```python
report = doctor.diagnose(...)
print(report.config_patch)
# → {"chunking.strategy": "recursive", "chunking.chunk_size": 256}
```

Apply the patch to your `rag-doctor.yaml`:

```yaml
chunking:
  strategy: recursive   # ← was: fixed
  chunk_size: 256       # ← was: 512
```

---

## Environment Variables

| Variable | Values | Description |
|---|---|---|
| `RAGDOCTOR_EMBEDDER` | `sentence-transformers`, `ollama`, `tfidf`, `char` | Force a specific embedding backend |

```bash
# Force TF-IDF for reproducible CI runs
export RAGDOCTOR_EMBEDDER=tfidf
rag-doctor batch --input test_set.jsonl
```
