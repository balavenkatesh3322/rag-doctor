# rag-doctor Documentation

> Diagnose why your RAG pipeline failed — without a database, without an LLM, in under 5 seconds.

---

## Contents

| Document | Description |
|---|---|
| [Quick Start](quickstart.md) | Install, first diagnosis in 5 minutes |
| [User Guide](user-guide.md) | All 5 user journeys, end-to-end flow, root causes |
| [Architecture](architecture.md) | Internal design: embedding chain, VectorStore, agent loop |
| [Tools Reference](tools-reference.md) | API for all 6 tools: parameters, outputs, examples |
| [Root Causes](root-causes.md) | All 6 root causes with symptoms, metrics, and fixes |
| [Connectors](connectors.md) | Connect Chroma, Pinecone, pgvector, LangChain |
| [Configuration](configuration.md) | All thresholds and `rag-doctor.yaml` reference |
| [API Reference](api-reference.md) | `Doctor`, `DiagnosisReport`, `Document`, `PipelineConnector` |
| [CI/CD Integration](ci-cd.md) | GitHub Actions, quality gates, pytest plugin |
| [Embedding Backends](embeddings.md) | TF-IDF → sentence-transformers → Ollama upgrade path |
| [Publishing](publishing.md) | How to build and release to PyPI |
| [Contributing](../CONTRIBUTING.md) | How to add tools, connectors, bug fixes |

---

## Samples Index

All samples in `samples/` run directly from the repo root — no install needed.

| File | Demonstrates |
|---|---|
| [`samples/01_basic_diagnosis.py`](../samples/01_basic_diagnosis.py) | Simplest possible diagnosis |
| [`samples/02_from_logs.py`](../samples/02_from_logs.py) | Reproduce production incident from log data |
| [`samples/03_batch_ci_gate.py`](../samples/03_batch_ci_gate.py) | CI quality gate with exit-code 1 on failures |
| [`samples/04_custom_connector.py`](../samples/04_custom_connector.py) | Build your own retrieval connector |
| [`samples/05_all_root_causes.py`](../samples/05_all_root_causes.py) | Trigger and inspect all 6 root causes |
| [`samples/06_json_report.py`](../samples/06_json_report.py) | Export JSON reports for logging/alerting |

---

## Philosophy

rag-doctor is built on one principle: **diagnosis should be frictionless**.

No database connection. No API keys. No cloud calls. You pass your documents as Python objects — from logs, a test fixture, a list literal, anywhere — and get a root cause, a severity, and a concrete fix.

The library auto-selects the best available embedding backend: TF-IDF if nothing is installed, `sentence-transformers` if available, Ollama if running. The same code works in CI with no internet access, on a developer laptop, and against a production Ollama instance.

---

## Root Causes at a Glance

| Code | Root Cause | Symptom |
|---|---|---|
| RC-0 | `healthy` | All tools passed |
| RC-1 | `retrieval_miss` | Correct document never in top-k results |
| RC-2 | `context_position_bias` | Correct doc retrieved but in middle of context |
| RC-3 | `chunk_fragmentation` | Key context split across chunk boundaries |
| RC-4 | `hallucination` | Answer not grounded in retrieved docs |
| RC-5 | `query_mismatch` | Query vocabulary doesn't match document vocabulary |
