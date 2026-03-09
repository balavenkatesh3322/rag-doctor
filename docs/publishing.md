# Publishing to PyPI

This document covers how to build, test, and publish rag-doctor to PyPI.

---

## Prerequisites

```bash
pip install build twine

# Optional but recommended: test on TestPyPI first
# Get an API token at https://test.pypi.org/manage/account/token/
# And at https://pypi.org/manage/account/token/
```

---

## Step-by-Step Release

### 1. Update version

In `rag_doctor/__init__.py`:
```python
__version__ = "1.1.0"
```

In `pyproject.toml`:
```toml
[project]
version = "1.1.0"
```

### 2. Update CHANGELOG.md

Add a new section:
```markdown
## [1.1.0] — 2025-XX-XX

### Added
- ...

### Fixed
- ...
```

### 3. Run tests

```bash
python run_tests.py
# Should show: 62/62 passed
```

### 4. Build

```bash
python -m build
# Creates: dist/rag_doctor-1.1.0-py3-none-any.whl
#          dist/rag_doctor-1.1.0.tar.gz
```

### 5. Test on TestPyPI (recommended)

```bash
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ rag-doctor==1.1.0
python -c "import rag_doctor; print(rag_doctor.__version__)"
```

### 6. Publish to PyPI

```bash
twine upload dist/*
# Enter your PyPI API token when prompted
# Or set: export TWINE_PASSWORD=pypi-...token...
```

### 7. Tag the release

```bash
git tag v1.1.0
git push origin v1.1.0
```

The GitHub Actions publish workflow (`.github/workflows/publish.yml`) will automatically publish to PyPI when you create a GitHub Release from the tag.

---

## GitHub Actions (Automated)

Publishing is automated via `.github/workflows/publish.yml`:

1. Go to GitHub → Releases → Draft a new release
2. Choose tag `v1.1.0`
3. Write release notes
4. Click **Publish release**
5. The workflow runs `python -m build` → `pypa/gh-action-pypi-publish`

**Required setup:**
- Add `PYPI_API_TOKEN` to GitHub repository secrets
- Or use [Trusted Publisher](https://docs.pypi.org/trusted-publishers/) (recommended — no token needed)

---

## PyPI Trusted Publisher Setup (No Tokens)

1. Go to [PyPI](https://pypi.org) → Your project → Publishing → Add a publisher
2. Fill in:
   - Owner: `your-org`
   - Repository: `rag-doctor`
   - Workflow: `publish.yml`
   - Environment: `pypi`
3. The workflow already has `permissions: id-token: write` — no secrets needed

---

## Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- `MAJOR.MINOR.PATCH`
- Patch: bug fixes, threshold adjustments
- Minor: new tools, new connector templates, new example scripts
- Major: breaking changes to Doctor API, DiagnosisReport schema, or tool interfaces

---

## pyproject.toml Reference

```toml
[project]
name = "rag-doctor"
version = "1.0.0"
description = "Agentic RAG pipeline failure diagnosis tool"
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.9"
dependencies = ["pyyaml>=6.0"]  # only hard dependency

[project.optional-dependencies]
embeddings = ["sentence-transformers>=2.2.0"]
full = ["sentence-transformers>=2.2.0", "pyyaml"]
dev = ["pytest>=7.0", "pytest-cov>=4.0", "ruff>=0.4", "build>=1.0", "twine>=5.0"]

[project.scripts]
rag-doctor = "rag_doctor.cli:main"
```

---

## Checking Your Package

```bash
# Check dist before uploading
twine check dist/*

# Install locally for testing
pip install dist/rag_doctor-1.0.0-py3-none-any.whl

# Verify CLI works
rag-doctor --help
rag-doctor diagnose --query "test" --answer "test"
```
