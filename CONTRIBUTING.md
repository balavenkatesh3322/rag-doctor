# Contributing to rag-doctor

Thank you for your interest in contributing! This document covers everything you need to get started.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [How to Contribute](#how-to-contribute)
- [Adding a New Tool](#adding-a-new-tool)
- [Adding a New Connector](#adding-a-new-connector)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Commit Message Format](#commit-message-format)

---

## Code of Conduct

Be kind. Be constructive. We follow the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

---

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/rag-doctor.git
   cd rag-doctor
   ```
3. **Set up** the development environment (see below)
4. **Create a branch** for your change:
   ```bash
   git checkout -b feat/your-feature-name
   ```

---

## Development Setup

### Requirements
- Python 3.9 or higher
- No external API keys needed for running tests

### Install in editable mode

```bash
pip install -e ".[dev]"
```

### Run tests

```bash
# Standalone runner — no pytest required
python run_tests.py

# Or with pytest if installed
pytest tests/ -v
```

All 62 tests should pass in under 1 second.

---

## Project Structure

```
rag_doctor/
├── config.py          # Configuration dataclasses
├── doctor.py          # Agent orchestrator — the main entry point
├── report.py          # DiagnosisReport and serialization
├── cli.py             # argparse CLI
├── tools/             # Diagnostic tools (one file per tool)
│   ├── base.py        # BaseTool and ToolResult
│   ├── chunk_analyzer.py
│   ├── retrieval_auditor.py
│   ├── position_tester.py
│   ├── hallucination_tracer.py
│   ├── chunk_optimizer.py
│   └── query_rewriter.py
└── connectors/        # Pipeline connectors
    ├── base.py        # Abstract base class
    └── mock.py        # Offline mock connector
```

---

## How to Contribute

### 🐛 Bug Reports

Open an issue using the **Bug Report** template. Include:
- Python version and OS
- Minimal reproduction case
- Expected vs. actual behavior

### 💡 Feature Requests

Open an issue using the **Feature Request** template. Describe:
- The problem you're solving
- Your proposed solution
- Alternatives you considered

### 🔧 Code Contributions

Good first contributions:
- New connector (LangChain, LlamaIndex, Haystack, raw OpenAI)
- New diagnostic tool
- Improved embedding strategy
- Better chunking heuristics
- Documentation improvements

---

## Adding a New Tool

1. Create `rag_doctor/tools/your_tool.py`:

```python
from .base import BaseTool, ToolResult
from ..connectors.base import Document
from typing import List

class YourTool(BaseTool):
    """One-line description of what this tool detects."""

    name = "your_tool"

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold

    def run(self, docs: List[Document], **kwargs) -> ToolResult:
        # Your logic here
        passed = True  # or False
        return ToolResult(
            tool_name=self.name,
            passed=passed,
            severity="low",   # low | medium | high | critical
            finding="Description of what was found.",
            details={"key": "value"},
            recommendation="What the developer should do to fix it." if not passed else None,
        )
```

2. Export from `rag_doctor/tools/__init__.py`:

```python
from .your_tool import YourTool
```

3. Register in `rag_doctor/config.py` under `DiagnosisConfig`:

```python
your_tool: bool = True
```

4. Wire into `rag_doctor/doctor.py` following the existing pattern.

5. Write at least **4 unit tests** in `tests/test_tools.py`:
   - Happy path (tool passes)
   - Failure detection path
   - Edge case (empty input)
   - Threshold/boundary test

---

## Adding a New Connector

1. Create `rag_doctor/connectors/your_connector.py`:

```python
from .base import Document, PipelineConnector
from typing import List

class YourConnector(PipelineConnector):
    """Connector for <framework name>."""

    def __init__(self, ...):
        ...

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        # Call your vector DB / retrieval system
        # Return List[Document] with position and score set
        ...

    def generate(self, query: str, docs: List[Document]) -> str:
        # Call your LLM
        ...
```

2. Export from `rag_doctor/connectors/__init__.py`.

3. Write tests using `MockConnector` patterns as reference.

4. Add connector to the supported integrations table in `README.md`.

---

## Testing Guidelines

### Philosophy
- **Every tool** must have at minimum: happy path, failure path, empty input, boundary test
- **No external API calls** in tests — use `MockConnector`
- **No network** — all tests run fully offline
- Tests must complete in **< 5 seconds** total

### Running specific tests

```bash
# Run tests for a specific class
python -c "
import sys; sys.path.insert(0, '.')
from run_tests import *
collect_and_run(TestChunkAnalyzer)
"

# Run with pytest
pytest tests/test_tools.py::TestChunkAnalyzer -v
```

### Writing good test data

The `MockConnector` uses a character-frequency embedding. When testing retrieval accuracy:

```python
# ✅ GOOD: Use actually distinct character sets for miss tests
doc = Document(content="ZZZZ XXXX WWWW BBBB", position=0, score=0.2)

# ❌ BAD: Normal English shares too many characters
doc = Document(content="The weather is nice today", position=0, score=0.2)
# This will score unexpectedly high against unrelated expected text
```

Use the fixtures in `tests/fixtures.py` for standard corpora.

---

## Pull Request Process

1. **Ensure tests pass**: `python run_tests.py` → all green
2. **Follow the code style**: run `ruff check .` (if ruff is installed)
3. **Update documentation** if you changed behavior
4. **Write a clear PR description** using the template
5. **Link any related issues** in the PR description

### PR Title Format

```
type(scope): short description

Examples:
feat(tools): add SemanticGapDetector tool
fix(retrieval_auditor): handle empty docs list
docs: update connector integration guide
test(e2e): add Haystack connector scenario
refactor(doctor): simplify root cause decision tree
```

---

## Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `perf`

---

## Questions?

Open a [Discussion](https://github.com/your-org/rag-doctor/discussions) or ping us in the issue tracker.
