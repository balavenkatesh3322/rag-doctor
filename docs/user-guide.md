# rag-doctor User Guide

> **No database. No API keys. No cloud calls.**
> Pass your documents as Python objects, get a root cause and a concrete fix.

---

## The Two Core Questions

### Do I need a database?

**No.** rag-doctor builds an ephemeral `VectorStore` in memory from the documents you pass in. Your production database — Chroma, Pinecone, pgvector, Weaviate — is never touched.

The `PipelineConnector` interface lets you *call* your own database from your own code, but rag-doctor itself has zero dependency on any DB.

| Approach | Problem |
|---|---|
| Connect production vector DB | Fragile — auth rotation, schema changes, network timeouts all break CI |
| Snapshot DB into test fixture | Stale — diverges from reality the moment someone adds a document |
| **Pass documents as Python lists** | **What rag-doctor does — zero coupling, always accurate** |

### Do I need an LLM?

**No — but one improves faithfulness accuracy.**

| Diagnostic Job | LLM Required? | How |
|---|---|---|
| Retrieval quality (RC-1) | No | Embedding cosine similarity |
| Position bias (RC-2) | No | Structural position check |
| Chunk coherence (RC-3) | No | Intra-chunk embedding similarity |
| Query rewriting (RC-5) | No | Heuristics + synonym expansion |
| Hallucination (RC-4) | No | Embedding similarity works well; LLM catches subtle cases |
| Chunk optimization | No | Grid search over recall@k |

**Practical upgrade path:**

| Week | Action | Benefit |
|---|---|---|
| 1 | Use TF-IDF (zero install) | Catches structural problems immediately |
| 2 | `pip install sentence-transformers` | Meaningful semantic similarity scores |
| 3+ | `ollama pull llama3.2` + `nomic-embed-text` | Full local LLM pipeline |

---

## Five User Journeys

### Journey 1 — Developer: Debug a Ticket in 5 Minutes

**Situation:** A user reported that the chatbot gave them the wrong acetaminophen dose. You have the query, the answer, and the retrieved docs in your application logs.

```python
# samples/02_from_logs.py — reconstruct from your production logs
import sys, os
sys.path.insert(0, "..")   # or: pip install rag-doctor

from rag_doctor import Doctor
from rag_doctor.connectors.base import Document

# Reconstruct from logs — no DB connection needed
docs = [
    Document(content=row["text"], score=row["score"], position=i)
    for i, row in enumerate(my_log_rows)
]

report = Doctor.default().diagnose(
    query    = "Max acetaminophen dose for liver disease?",
    answer   = "The maximum daily dose is 4000mg.",
    docs     = docs,
    expected = "For liver disease patients max dose is 2000mg per day.",
)
print(report.root_cause)      # → context_position_bias
print(report.fix_suggestion)  # → Enable a reranker to push best doc to position 0.
```

**Result:** Root cause in under 2 seconds. No DB connection. Works from logs alone.

---

### Journey 2 — Data Scientist: Nightly Evaluation Tests

**Situation:** You have a golden dataset of 50 query-answer pairs. You want CI to catch any RAG regression automatically.

```python
# samples/03_batch_ci_gate.py
from rag_doctor import Doctor
from rag_doctor.connectors.mock import MockConnector

conn   = MockConnector(corpus=YOUR_CORPUS, quiet=True)
doctor = Doctor.default(conn)

for query, expected in GOLDEN_PAIRS:
    docs   = conn.retrieve(query, top_k=5)
    answer = conn.generate(query, docs)
    report = doctor.diagnose(query=query, answer=answer, docs=docs, expected=expected)
    assert report.severity in ("low", "medium"), report.to_text()
```

In CI (GitHub Actions, Jenkins, etc.) this exits with code 1 when any case fails — your pipeline blocks automatically.

---

### Journey 3 — MLOps Engineer: CI Quality Gate (Zero External Deps)

**Situation:** You need this to run in CI in under 30 seconds with no network access, no pip installs beyond rag-doctor itself.

```yaml
# .github/workflows/rag-quality.yml
- name: RAG quality gate
  run: |
    pip install rag-doctor
    python examples/ci_quality_gate.py
```

With no sentence-transformers, TF-IDF handles all structural checks (truncation, position bias, faithfulness ratio). The quality gate runs in under 5 seconds on any CI runner.

---

### Journey 4 — Researcher: Compare Chunking Strategies

**Situation:** You want to know whether `chunk_size=256` or `chunk_size=512` gives better recall for your domain corpus.

```python
from rag_doctor.tools.chunk_optimizer import ChunkOptimizer

result = ChunkOptimizer().run(
    corpus_texts = corpus_texts,        # your raw documents
    test_pairs   = [{"query": q, "expected": a} for q, a in golden_pairs],
)

# Ranked grid-search results
for s in result.details["ranked_strategies"]:
    print(f"{s['strategy']:10} size={s['chunk_size']:4}  recall@5={s['recall_at_5']:.2f}")
# recursive  size= 256  recall@5=0.82
# fixed      size= 512  recall@5=0.71
```

All in memory — no DB writes, no persistent index.

---

### Journey 5 — Team Lead: Post-Incident Retrospective

**Situation:** Five wrong answers in the last week. Diagnose all at once before the post-mortem.

```python
# samples/06_json_report.py
from rag_doctor import Doctor

cases = [
    {"query": q, "answer": a, "expected": e}
    for q, a, e in incident_log
]
reports = Doctor.default().batch_diagnose(cases)

for r in reports:
    print(f"{r.root_cause:30}  [{r.severity}]  {r.query[:50]}")
```

Or from the CLI:
```bash
rag-doctor batch --input incidents.jsonl --output report.json
```

---

## End-to-End Flow

```
Your Documents (Python list of dicts or Document objects)
         │
         ▼
  VectorStore (ephemeral, in-memory, never persisted)
  ├── Embedder: sentence-transformers → Ollama → TF-IDF → char-freq
  └── embed_batch(all_corpus_texts)   ← entire corpus fitted at once
         │
         ▼
  Doctor.diagnose(query, answer, docs, expected, corpus_texts)
         │
         ├── RetrievalAuditor   → pass? → continue
         │        └── fail?    → QueryRewriter (RC-5 or RC-1)
         │
         ├── PositionTester     → pass? → continue   [only if len(docs)>=3]
         │        └── fail?    → RC-2
         │
         ├── ChunkAnalyzer      → pass? → continue
         │        └── fail?    → ChunkOptimizer → RC-3
         │
         └── HallucinationTracer → pass? → RC-0 healthy
                  └── fail?      → RC-4
         │
         ▼
  DiagnosisReport
  ├── root_cause        (string key)
  ├── root_cause_id     (RC-0 … RC-5)
  ├── severity          (low / medium / high / critical)
  ├── finding           (human-readable explanation)
  ├── fix_suggestion    (concrete action to take)
  ├── config_patch      (dict of YAML settings to change)
  ├── faithfulness_score (0.0–1.0)
  ├── retrieval_score    (0.0–1.0)
  └── tool_results      (one ToolResult per tool that ran)
```

---

## The Six Root Causes

| Code | Name | Cause | Primary Fix |
|---|---|---|---|
| RC-0 | `healthy` | No issues | Nothing |
| RC-1 | `retrieval_miss` | Correct doc not in top-k | Increase `top_k`, add hybrid search |
| RC-2 | `context_position_bias` | Correct doc in danger-zone middle position | Enable reranker, use MMR |
| RC-3 | `chunk_fragmentation` | Chunks truncated mid-sentence | Recursive chunking, smaller `chunk_size` |
| RC-4 | `hallucination` | Answer not grounded in retrieved docs | Stronger system prompt, lower temperature |
| RC-5 | `query_mismatch` | Query vocabulary ≠ document vocabulary | Query expansion, HyDE, domain embeddings |

---

## Choosing Your Setup

| Your Situation | Recommended Setup |
|---|---|
| Debug production incident right now | Mode A: pass `docs` + `answer` directly (see `samples/02_from_logs.py`) |
| Add RAG quality tests to CI | MockConnector + TF-IDF, no extra install (see `samples/03_batch_ci_gate.py`) |
| Improve semantic accuracy | `pip install sentence-transformers` — raise thresholds to 0.65 |
| Test full production stack locally | Implement `PipelineConnector` (see `samples/04_custom_connector.py`) |
| Compare chunking strategies | Pass `corpus_texts` to `diagnose()`, use `ChunkOptimizer` |
| Build a monitoring dashboard | Use `batch_diagnose()` + `to_json()` (see `samples/06_json_report.py`) |
| Air-gapped / offline | rag-doctor + numpy only — zero network calls at any point |

---

## Running the Samples

All samples run standalone from the repo root with no pip install needed (they add `..` to `sys.path` automatically):

```bash
git clone https://github.com/your-org/rag-doctor
cd rag-doctor

python samples/01_basic_diagnosis.py    # simplest possible diagnosis
python samples/02_from_logs.py          # reproduce an incident from logs
python samples/03_batch_ci_gate.py      # CI quality gate demo
python samples/04_custom_connector.py   # build your own connector
python samples/05_all_root_causes.py    # trigger all 6 root causes
python samples/06_json_report.py        # export JSON for monitoring
```

Or run them all at once:
```bash
./scripts/test_local_mac.sh
```
