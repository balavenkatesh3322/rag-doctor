<div align="center">

# 🩺 rag-doctor

**Stop guessing why your RAG pipeline is wrong. Get the root cause in 2 seconds.**

[![PyPI](https://img.shields.io/pypi/v/rag-doctor?color=0066CC&label=pypi&style=flat-square)](https://pypi.org/project/rag-doctor/)
[![Python](https://img.shields.io/pypi/pyversions/rag-doctor?style=flat-square)](https://pypi.org/project/rag-doctor/)
[![Tests](https://img.shields.io/badge/tests-62%20passing-22c55e?style=flat-square)](#)
[![License](https://img.shields.io/pypi/l/rag-doctor?style=flat-square&color=22c55e)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/your-org/rag-doctor/ci.yml?style=flat-square&label=CI)](https://github.com/your-org/rag-doctor/actions)

</div>

---

```bash
pip install rag-doctor
```

```python
from rag_doctor import Doctor

report = Doctor.default().diagnose(
    query    = "Max acetaminophen dose for liver disease?",
    answer   = "The maximum daily dose is 4000mg.",
    docs     = your_retrieved_docs,
    expected = "For liver disease patients max dose is 2000mg/day.",
)
print(report.root_cause)      # context_position_bias
print(report.fix_suggestion)  # Enable a reranker to push the most relevant document to position 0.
```

```
Root Cause : context_position_bias (RC-2)
Severity   : HIGH
Finding    : Best document at position 1/2 — in danger zone (risk: 1.00)
Fix        : Enable a reranker to push the most relevant document to position 0.
Config     : {"retrieval.reranker": true}
```

---

## Why rag-doctor

RAG pipelines fail silently. You get a wrong answer and have no idea if it's a retrieval miss, a chunking problem, position bias, or a hallucination. Existing eval tools give you a **score**. rag-doctor gives you the **cause** — and tells you exactly what to change.

| Instead of... | You get... |
|---|---|
| "Accuracy dropped from 0.82 to 0.74" | "RC-3 chunk_fragmentation — switch to recursive chunking, chunk_size=256" |
| Grepping logs for an hour | Root cause in 2 seconds from the same log data |
| Guessing which knob to turn | A concrete `config_patch` dict to apply |

---

## Six root causes. One tool.

```
RC-0  healthy               All tools passed — pipeline is working correctly
RC-1  retrieval_miss        Correct document never in top-k results
RC-2  context_position_bias Correct doc retrieved but ignored in middle of context
RC-3  chunk_fragmentation   Key context split across chunk boundaries
RC-4  hallucination         Answer not grounded in retrieved documents
RC-5  query_mismatch        Query vocabulary doesn't match document vocabulary
```

Every diagnosis includes: root cause · severity · human-readable finding · fix suggestion · machine-readable `config_patch`.

---

## Zero dependencies. Zero infrastructure.

**No database.** rag-doctor builds an ephemeral vector store in memory from the documents you pass — as Python objects from logs, fixtures, or a list literal. Your production database is never touched.

**No LLM API.** All structural checks work offline. Embedding backend is auto-selected:

```
sentence-transformers  ←  pip install sentence-transformers  (recommended)
Ollama nomic-embed-text ←  ollama pull nomic-embed-text       (local LLM stack)
TF-IDF                 ←  stdlib + numpy                      (zero install, always works)
```

**No configuration.** Works out of the box. Tune thresholds when you need to.

---

## Three ways to use it

### 1 — Debug a production incident from logs (no DB re-query)

```python
from rag_doctor import Doctor
from rag_doctor.connectors.base import Document

# Reconstruct from your log store — no database connection needed
docs = [
    Document(content=row["text"], score=row["score"], position=i)
    for i, row in enumerate(logged_rows)
]

report = Doctor.default().diagnose(
    query    = "What is the Acme Corp termination notice period?",
    answer   = "30 days notice required.",
    docs     = docs,
    expected = "Acme Corp requires 90 days written notice.",
)

print(report.root_cause)      # → hallucination
print(report.severity)        # → critical
print(report.fix_suggestion)  # → Add 'Answer only from provided sources' to your system prompt.
```

### 2 — Quality gate in CI (no external services)

```python
# pytest — blocks deploys when RAG quality regresses
def test_rag_quality(golden_pairs):
    connector = MockConnector(corpus=CORPUS)
    for query, expected in golden_pairs:
        docs   = connector.retrieve(query, top_k=5)
        report = Doctor.default(connector).diagnose(
            query=query, answer=connector.generate(query, docs),
            docs=docs, expected=expected,
        )
        assert report.severity in ("low", "medium"), report.to_text()
```

```yaml
# GitHub Actions — zero setup, runs in < 5s
- name: RAG quality gate
  run: python samples/03_batch_ci_gate.py
```

### 3 — Connect your production stack

```python
from rag_doctor.connectors.base import PipelineConnector, Document

class ChromaConnector(PipelineConnector):
    def retrieve(self, query, top_k=5):
        results = self.collection.query(query_texts=[query], n_results=top_k)
        return [
            Document(content=d, score=1-s, position=i)
            for i,(d,s) in enumerate(zip(
                results["documents"][0], results["distances"][0]
            ))
        ]
    def generate(self, query, docs):
        ...  # your LLM call

report = Doctor.default(ChromaConnector(my_collection)).diagnose(...)
```

Drop-in templates for **Chroma, Pinecone, pgvector, LangChain** in [`docs/connectors.md`](docs/connectors.md).

---

## CLI

```bash
# Single diagnosis
rag-doctor diagnose \
  --query    "What is the termination notice period?" \
  --answer   "30 days." \
  --expected "Enterprise requires 90 days written notice."

# Batch from JSONL — exits 1 when severity >= high (perfect for CI)
rag-doctor batch \
  --input  incidents.jsonl \
  --output report.json \
  --fail-on-severity high

# JSON output for Datadog / PagerDuty / Slack
rag-doctor diagnose --query "..." --answer "..." --output json \
  | jq '{cause: .root_cause, severity: .severity, fix: .fix_suggestion}'
```

---

## Batch diagnosis

```python
cases = [
    {"query": q, "answer": a, "expected": e}
    for q, a, e in incident_log
]
reports = Doctor.default().batch_diagnose(cases)

for r in reports:
    print(f"{r.root_cause:30} [{r.severity:8}]  {r.query[:50]}")
```

```
hallucination                  [critical]  What is the enterprise SLA uptime?
context_position_bias          [high    ]  Max acetaminophen dose for liver disease?
chunk_fragmentation            [high    ]  Acme Corp termination notice period?
healthy                        [low     ]  What is the return policy?
```

---

## Install

```bash
# Minimum — TF-IDF embeddings, zero extra dependencies
pip install rag-doctor

# Recommended — semantic embeddings (much better accuracy)
pip install rag-doctor sentence-transformers

# Full local stack with Ollama
pip install rag-doctor
ollama pull nomic-embed-text   # embeddings
ollama pull llama3.2           # generation
```

**Python 3.9+ · macOS · Linux · Windows · CI-friendly**

---

## Samples — run immediately

```bash
git clone https://github.com/your-org/rag-doctor
cd rag-doctor
python samples/01_basic_diagnosis.py    # simplest possible usage
python samples/02_from_logs.py          # reproduce incident from logs
python samples/03_batch_ci_gate.py      # CI quality gate with exit codes
python samples/04_custom_connector.py   # connect your own retrieval stack
python samples/05_all_root_causes.py    # see all 6 root causes triggered live
python samples/06_json_report.py        # structured JSON for alerting/logging
```

No install needed beyond `pip install rag-doctor`. No API keys. No Ollama. Runs fully offline.

---

## Documentation

| | |
|---|---|
| [Quick Start](docs/quickstart.md) | Install and first diagnosis in 5 minutes |
| [User Guide](docs/user-guide.md) | 5 user journeys end-to-end |
| [Root Causes](docs/root-causes.md) | All 6 root causes with symptoms, metrics, and fixes |
| [Tools Reference](docs/tools-reference.md) | API for RetrievalAuditor, PositionTester, ChunkAnalyzer, HallucinationTracer, QueryRewriter, ChunkOptimizer |
| [Connectors](docs/connectors.md) | Chroma, Pinecone, pgvector, LangChain templates |
| [Architecture](docs/architecture.md) | Embedding chain, VectorStore, agent loop internals |
| [Configuration](docs/configuration.md) | Thresholds, YAML config, severity levels |
| [CI/CD Integration](docs/ci-cd.md) | GitHub Actions, pytest plugin, quality gates |
| [Publishing](docs/publishing.md) | Build and release to PyPI |

---

## Local development

```bash
git clone https://github.com/your-org/rag-doctor
cd rag-doctor
chmod +x scripts/test_local_mac.sh
./scripts/test_local_mac.sh     # sets up venv, runs 62 tests, runs all samples
```

```bash
make test          # 62 offline unit tests
make test-samples  # run all 6 sample scripts
make lint          # ruff
make build         # wheel + sdist
make publish       # ship to PyPI
```

---

## Contributing

Issues and PRs are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

- Adding a new tool: subclass `BaseTool` in `rag_doctor/tools/`, add it to the agent loop in `doctor.py`
- Adding a connector: subclass `PipelineConnector` in `rag_doctor/connectors/`
- All PRs require 62/62 tests passing

---

<div align="center">

**MIT License** · Built for developers who are tired of wrong RAG answers with no explanation.

[PyPI](https://pypi.org/project/rag-doctor/) · [Issues](https://github.com/your-org/rag-doctor/issues) · [Docs](docs/)

</div>