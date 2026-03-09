<div align="center">

# 🩺 rag-doctor

**Diagnose why your RAG pipeline returned the wrong answer — in under 2 seconds.**

[![PyPI version](https://img.shields.io/pypi/v/rag-doctor?color=blue)](https://pypi.org/project/rag-doctor/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-62%20passing-brightgreen)](run_tests.py)
[![CI](https://github.com/your-org/rag-doctor/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/rag-doctor/actions)

No database. No API keys. No cloud calls. Just pass your documents and get a root cause.

[Quick Start](#quick-start) · [How It Works](#how-it-works) · [Examples](#examples) · [CLI](#cli-reference) · [Docs](docs/)

</div>

---

## The Problem

RAG pipelines fail silently. You get a wrong answer and have no idea if it's a chunking problem, a retrieval miss, a position bias, or a hallucination. Existing evaluation tools give you a score — they don't tell you **why**.

**rag-doctor tells you why.**

```
══════════════════════════════════════════════════════════════
  RAG-DOCTOR ✗ ISSUES FOUND
══════════════════════════════════════════════════════════════
  Root Cause : context_position_bias (RC-2)
  Severity   : HIGH
  Finding    : Best document at position 1/2 — in danger zone (risk: 1.00)
──────────────────────────────────────────────────────────────
  Fix: Enable a reranker to push the most relevant document to position 0.
  Config Patch: {"retrieval.reranker": true}
══════════════════════════════════════════════════════════════
```

---

## Quick Start

```bash
pip install rag-doctor
```

```python
from rag_doctor import Doctor
from rag_doctor.connectors.mock import MockConnector

connector = MockConnector(corpus=[
    {"id": "doc1", "content": "For liver disease patients maximum acetaminophen dose is 2000mg per day."},
    {"id": "doc2", "content": "Standard adult dose: up to 4000mg per day."},
])

docs   = connector.retrieve("acetaminophen dose liver disease", top_k=3)
answer = "The maximum daily dose is 4000mg."

report = Doctor.default().diagnose(
    query    = "What is the max acetaminophen dose for liver disease?",
    answer   = answer,
    docs     = docs,
    expected = "For liver disease patients max dose is 2000mg per day.",
)
print(report.to_text())
```

---

## How It Works

rag-doctor runs a deterministic six-tool agent loop. Each tool targets a specific failure mode:

| Tool | Root Cause | What It Catches |
|---|---|---|
| `RetrievalAuditor` | RC-1 `retrieval_miss` | Correct document not in top-k results |
| `PositionTester` | RC-2 `context_position_bias` | Correct doc retrieved but ignored in middle position |
| `ChunkAnalyzer` | RC-3 `chunk_fragmentation` | Mid-sentence truncation, incoherent chunks |
| `HallucinationTracer` | RC-4 `hallucination` | Answer claims not grounded in retrieved documents |
| `QueryRewriter` | RC-5 `query_mismatch` | Query vocabulary doesn't match document vocabulary |
| `ChunkOptimizer` | RC-3 sub-tool | Grid-searches best chunk_size and strategy |

---

## No Database. No LLM. No API Keys.

rag-doctor builds an ephemeral `VectorStore` in memory from whatever documents you pass. Your production database is never touched.

**Embedding backends** (auto-selected, no config needed):

| Priority | Backend | Install |
|---|---|---|
| 1 | `sentence-transformers` | `pip install sentence-transformers` |
| 2 | Ollama `nomic-embed-text` | `ollama pull nomic-embed-text` |
| 3 | TF-IDF (stdlib + numpy) | nothing — built in |
| 4 | Char n-gram fallback | nothing — always available |

---

## Three Ways to Use It

### Mode A — Debug from Logs (no re-query needed)

```python
from rag_doctor import Doctor
from rag_doctor.connectors.base import Document

docs = [
    Document(content=row["text"], score=row["score"], position=i)
    for i, row in enumerate(db_rows)
]
report = Doctor.default().diagnose(
    query="What is the refund policy?",
    answer="30 days for all customers.",
    docs=docs,
    expected="Enterprise customers get 90-day refunds.",
)
print(report.root_cause)      # retrieval_miss
print(report.fix_suggestion)  # Increase top_k or check corpus coverage.
```

### Mode B — Corpus-Level Evaluation (CI / pytest)

```python
connector = MockConnector(corpus=YOUR_CORPUS)
docs      = connector.retrieve(query, top_k=5)
report    = Doctor.default(connector).diagnose(query=query, answer=answer, docs=docs, expected=expected)
assert report.severity in ("low", "medium"), report.to_text()
```

### Mode C — Connect Your Production Stack

```python
class ChromaConnector(PipelineConnector):
    def retrieve(self, query, top_k=5):
        results = self.collection.query(query_texts=[query], n_results=top_k)
        return [Document(content=d, score=s, position=i) for i,(d,s) in enumerate(...)]

report = Doctor.default(ChromaConnector(my_collection)).diagnose(...)
```

---

## CLI Reference

```bash
# Single query
rag-doctor diagnose \
  --query    "What is the termination notice?" \
  --answer   "30 days." \
  --expected "Enterprise requires 90 days written notice."

# Batch from JSONL
rag-doctor batch --input examples/batch_example.jsonl --fail-on-severity high

# JSON output for CI
rag-doctor diagnose --query "..." --answer "..." --output json | jq .root_cause
```

---

## Local Setup (Mac)

```bash
git clone https://github.com/your-org/rag-doctor
cd rag-doctor
chmod +x scripts/test_local_mac.sh
./scripts/test_local_mac.sh
```

---

## Documentation

| Doc | Description |
|---|---|
| [docs/user-guide.md](docs/user-guide.md) | Complete user guide — all 5 journeys, all 6 root causes |
| [docs/architecture.md](docs/architecture.md) | Internal design: embedding chain, VectorStore, agent loop |
| [docs/tools-reference.md](docs/tools-reference.md) | API reference for all 6 tools |
| [docs/connectors.md](docs/connectors.md) | Building custom connectors |
| [docs/configuration.md](docs/configuration.md) | Thresholds and config options |
| [docs/publishing.md](docs/publishing.md) | How to release to PyPI |

---

## License

[MIT](LICENSE) — free for personal and commercial use.
