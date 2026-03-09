# Changelog

All notable changes to rag-doctor are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
rag-doctor uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- LangChain connector
- LlamaIndex connector
- Cohere Rerank integration
- MCP server implementation
- pytest plugin (`pytest-rag-doctor`)
- HTML batch report output

---

## [1.0.0] — 2025-03-01

### Added
- **Six diagnostic tools**: ChunkAnalyzer, RetrievalAuditor, PositionTester, HallucinationTracer, ChunkOptimizer, QueryRewriter
- **Doctor orchestrator**: deterministic rule-based agent loop with 5 root cause IDs (RC-1 through RC-5)
- **DiagnosisReport**: structured report with JSON and text serialization
- **CLI**: `diagnose` and `batch` sub-commands via argparse
- **Python SDK**: `Doctor`, `RagDoctorConfig`, `DiagnosisReport` public API
- **MockConnector**: full offline testing without external APIs
- **Config schema**: YAML-based configuration with dataclasses
- **62 tests**: unit, integration, and end-to-end coverage — 100% pass rate
- **GitHub Actions**: CI workflow for Python 3.9, 3.10, 3.11, 3.12
- **Documentation**: 3 docx files covering developer guide, technical build, and problem-solution scenarios
- **Standalone test runner**: `run_tests.py` — no pytest required

[Unreleased]: https://github.com/your-org/rag-doctor/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/your-org/rag-doctor/releases/tag/v1.0.0
