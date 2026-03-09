# Changelog

All notable changes to rag-doctor are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- LangChain connector template
- LlamaIndex connector template
- Cohere Rerank integration
- pytest plugin (`pytest-rag-doctor`)
- HTML batch report output
- MCP server implementation

---

## [1.0.0] — 2025-03-01

### Added

**Core library**
- `Doctor` orchestrator — deterministic 6-tool agent loop with root cause priority ordering
- `DiagnosisReport` dataclass with `.to_text()`, `.to_json()`, `.to_dict()` serialization
- `RagDoctorConfig` dataclass with YAML load/save and calibrated TF-IDF defaults
- `Doctor.default()` factory — zero-config entry point
- `Doctor.batch_diagnose()` — batch diagnosis with shared connector

**Six diagnostic tools**
- `ChunkAnalyzer` — RC-3: mid-sentence truncation + intra-chunk coherence
- `RetrievalAuditor` — RC-1: recall@k with ground truth similarity
- `PositionTester` — RC-2: lost-in-the-middle detection
- `HallucinationTracer` — RC-4: claim grounding via embedding similarity
- `QueryRewriter` — RC-5: HyDE, step-back, synonym expansion
- `ChunkOptimizer` — RC-3 sub-tool: grid search over 5 chunking strategies

**Embedding chain** (auto-selected, no config)
- `SentenceTransformerEmbedder` — semantic embeddings via sentence-transformers
- `OllamaEmbedder` — local embeddings via Ollama nomic-embed-text
- `TFIDFEmbedder` — stdlib + numpy, zero extra install
- `CharFreqEmbedder` — stdlib-only fallback, always available

**Connectors**
- `MockConnector` — full offline testing with real embeddings + inject flags
- `OllamaConnector` — local LLM pipeline with auto model selection
- `PipelineConnector` ABC — base class for custom connectors

**CLI** (`rag-doctor` command)
- `diagnose` sub-command — single query with text/JSON output
- `batch` sub-command — JSONL batch with `--fail-on-severity` gate

**Samples** (`samples/`)
- `01_basic_diagnosis.py` — simplest possible diagnosis
- `02_from_logs.py` — reproduce production incident from logs
- `03_batch_ci_gate.py` — CI quality gate with exit-code 1
- `04_custom_connector.py` — build your own PipelineConnector
- `05_all_root_causes.py` — trigger all 6 root causes
- `06_json_report.py` — export JSON for logging/alerting

**Scripts**
- `scripts/test_local_mac.sh` — full local setup + all tests + samples
- `scripts/push_github.sh` — clean, commit, push to GitHub
- `scripts/publish_pypi.sh` — build, check, publish to TestPyPI or PyPI

**Documentation** (`docs/`)
- `index.md` — full doc index with samples reference
- `quickstart.md` — first diagnosis in 5 minutes
- `user-guide.md` — all 5 user journeys, end-to-end flow
- `architecture.md` — embedding chain, VectorStore, agent loop
- `tools-reference.md` — all 6 tools: parameters, outputs, examples
- `root-causes.md` — RC-0 through RC-5 with symptoms, metrics, and fixes
- `connectors.md` — Chroma, Pinecone, pgvector, LangChain templates
- `configuration.md` — all thresholds and YAML reference
- `api-reference.md` — Python API reference
- `ci-cd.md` — GitHub Actions integration
- `embeddings.md` — embedding backend upgrade path
- `publishing.md` — PyPI release process

**CI/CD**
- GitHub Actions CI: Python 3.9, 3.10, 3.11, 3.12 matrix
- Automated PyPI publish on GitHub Release
- RAG quality gate workflow

**Test suite**
- 62 tests: unit, integration, and end-to-end
- `run_tests.py` standalone runner — no pytest required
- 4 test modules: connectors, tools, doctor, end-to-end

### Fixed

**Embedding correctness (4 bugs)**
- `VectorStore` now fits TF-IDF on full corpus before embedding (was: fitted per-query)
- `TFIDFEmbedder` is no longer shared across `VectorStore` instances (was: stale vocabulary)
- All tools use `fit_and_embed(all_texts)` to embed comparison texts together (was: sentence-by-sentence)
- `PositionTester` determines best position internally (was: relied on caller)

[Unreleased]: https://github.com/your-org/rag-doctor/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/your-org/rag-doctor/releases/tag/v1.0.0
