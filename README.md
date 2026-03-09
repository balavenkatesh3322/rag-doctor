<div align="center">

# 🩺 rag-doctor

**Agentic RAG pipeline failure diagnosis tool**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-62%20passing-brightgreen)](run_tests.py)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

Stop guessing why your RAG pipeline returned the wrong answer.  
**rag-doctor** runs a six-tool agentic diagnosis loop and pinpoints the exact root cause — plus a concrete fix.

[Quick Start](#quick-start) · [How It Works](#how-it-works) · [CLI Reference](#cli-reference) · [Python SDK](#python-sdk) · [Contributing](#contributing)

</div>

---

## The Problem

RAG pipelines fail silently. You get a wrong answer and have no idea if it's a chunking problem, a retrieval miss, a position bias, or a hallucination. Existing evaluation tools give you a score — they don't tell you *why*.

**rag-doctor** tells you why.

```
Root Cause : chunk_fragmentation (RC-3)
Severity   : HIGH
Finding    : 2/3 chunks appear truncated mid-sentence.
Fix        : Switch to recursive chunking, reduce chunk_size to 256.
Config     : { "chunking.strategy": "recursive", "chunking.chunk_size": 256 }
```

---

## Quick Start

```bash
pip install rag-doctor

rag-doctor diagnose \
  --query    "What is the Acme Corp termination notice period?" \
  --answer   "30 days notice required." \
  --expected "Enterprise contracts require 90 days written notice."
```

---

## How It Works

rag-doctor runs a deterministic agent loop across six specialized tools:

| Tool | Root Cause ID | Detects |
|---|---|---|
| `ChunkAnalyzer` | RC-3 | Mid-sentence truncation, low intra-chunk coherence |
| `RetrievalAuditor` | RC-1 | Recall@k failure — correct doc not in top-k |
| `PositionTester` | RC-2 | Lost-in-the-middle: correct doc in middle position |
| `HallucinationTracer` | RC-4 | Claims not grounded in retrieved documents |
| `ChunkOptimizer` | RC-3 | Best chunking strategy via grid search |
| `QueryRewriter` | RC-5 | Vocabulary mismatch via HyDE rewriting |

### Root Cause IDs

| ID | `root_cause` | Symptom |
|---|---|---|
| RC-1 | `retrieval_miss` | Correct doc never retrieved |
| RC-2 | `context_position_bias` | Correct doc retrieved but ignored (middle position) |
| RC-3 | `chunk_fragmentation` | Key context split across chunks |
| RC-4 | `hallucination` | Answer not grounded in retrieved docs |
| RC-5 | `query_mismatch` | Query vocabulary doesn't match doc vocabulary |
| RC-0 | `healthy` | All tools passed |

---

## CLI Reference

### Diagnose a single query

```bash
rag-doctor diagnose \
  --query    "Your user query" \
  --answer   "The generated answer" \
  --expected "The ground-truth answer"  # optional but improves diagnosis
  --config   rag-doctor.yaml            # optional
  --output   text                       # text | json
```

### Batch diagnose from a JSONL file

```bash
# Each line: {"query": "...", "answer": "...", "expected": "..."}
rag-doctor batch \
  --input              test_set.jsonl \
  --output             report.json \
  --fail-on-severity   high
```

### Find the best chunking strategy

```bash
rag-doctor optimize-chunks \
  --corpus    ./docs/ \
  --test-set  test_set.jsonl
```

---

## Python SDK

```python
from rag_doctor import Doctor

doctor = Doctor.default()

report = doctor.diagnose(
    query    = "What is the refund policy?",
    answer   = "30 day returns are available.",
    docs     = retrieved_docs,   # List[Document] — optional
    expected = "Full refunds within 30 days of purchase.",
)

print(report.root_cause)      # → "chunk_fragmentation"
print(report.severity)        # → "medium"
print(report.fix_suggestion)  # → "Reduce chunk_size or switch to recursive..."
print(report.config_patch)    # → {"chunking.strategy": "recursive", ...}
print(report.to_json())       # → Full JSON report
```

### Bring your own connector

```python
from rag_doctor import Doctor
from rag_doctor.connectors.base import PipelineConnector, Document

class MyConnector(PipelineConnector):
    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        # your retrieval logic
        ...
    def generate(self, query: str, docs: list[Document]) -> str:
        # your generation logic
        ...

doctor = Doctor.default(connector=MyConnector())
report = doctor.diagnose(query="...", answer="...")
```

---

## Configuration

Create `rag-doctor.yaml` in your project root:

```yaml
pipeline_type: langchain     # or llamaindex | custom
vector_db: chroma            # chroma | pinecone | weaviate | pgvector
embedding_model: all-MiniLM-L6-v2
llm: gpt-4o

retrieval:
  top_k: 5
  reranker: false

chunking:
  strategy: fixed            # fixed | recursive | semantic | hierarchical
  chunk_size: 512
  chunk_overlap: 64

diagnosis:
  recall_threshold: 0.75
  faithfulness_threshold: 0.70
  coherence_threshold: 0.65
  severity_threshold: medium
```

---

## GitHub Action

```yaml
# .github/workflows/rag-quality.yml
name: RAG Quality Gate
on: [pull_request]

jobs:
  rag-doctor:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install rag-doctor
      - run: |
          rag-doctor batch \
            --input test_set.jsonl \
            --output report.json \
            --fail-on-severity high
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: rag-diagnosis-report
          path: report.json
```

---

## MCP Integration (Cursor / Claude Code)

```bash
# Start the MCP server
rag-doctor mcp-server --port 3100

# Add to Cursor → Settings → MCP Servers:
# { "rag-doctor": { "url": "http://localhost:3100" } }
```

Then ask Cursor: *"Why did my RAG query return the wrong answer?"*

---

## Running Tests

```bash
# No external dependencies required
python run_tests.py

# Expected output:
# Results: 62/62 passed in 0.02s
```

---

## Documentation

All documentation lives in the [`docs/`](docs/) folder:

| File | Contents |
|---|---|
| [`01-developer-guide.docx`](docs/01-developer-guide.docx) | Install, configure, CLI & SDK reference |
| [`02-technical-build.docx`](docs/02-technical-build.docx) | Phase-by-phase engineering spec |
| [`03-problem-solution.docx`](docs/03-problem-solution.docx) | 4 real-world failure scenarios with fixes |

---

## Project Structure

```
rag-doctor/
├── rag_doctor/
│   ├── __init__.py
│   ├── config.py              # Dataclass config schema
│   ├── doctor.py              # Agent orchestrator
│   ├── report.py              # DiagnosisReport
│   ├── cli.py                 # CLI (argparse)
│   ├── tools/                 # 6 diagnostic tools
│   │   ├── chunk_analyzer.py
│   │   ├── retrieval_auditor.py
│   │   ├── position_tester.py
│   │   ├── hallucination_tracer.py
│   │   ├── chunk_optimizer.py
│   │   └── query_rewriter.py
│   └── connectors/
│       ├── base.py            # PipelineConnector ABC
│       └── mock.py            # MockConnector (offline testing)
├── tests/
│   ├── fixtures.py
│   ├── test_tools.py          # 24 unit tests
│   ├── test_connectors.py     # 11 unit tests
│   ├── test_doctor.py         # 15 integration tests
│   └── test_end_to_end.py     # 12 end-to-end tests
├── docs/                      # 3 docx documentation files
├── .github/workflows/         # CI/CD workflows
├── run_tests.py               # Standalone test runner
├── rag-doctor.yaml            # Example config
└── pyproject.toml
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). We welcome:

- New connector implementations (LangChain, LlamaIndex, Haystack)
- New diagnostic tools
- Better embedding strategies
- Bug reports and fixes

---

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">
Built with ❤️ by the open-source community
</div>
