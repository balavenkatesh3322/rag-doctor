# Publishing rag-doctor

Complete guide to publishing rag-doctor as an open-source project.

---

## Step 1: Create the GitHub Repository

1. Go to https://github.com/new
2. Set:
   - **Repository name**: `rag-doctor`
   - **Description**: `Agentic RAG pipeline failure diagnosis tool`
   - **Visibility**: Public
   - **DO NOT** initialise with README (you already have one)
3. Click **Create repository**

---

## Step 2: Push the Code

```bash
# Clone or navigate to the rag-doctor directory
cd rag-doctor

# Run the setup script with your GitHub URL
./setup.sh https://github.com/YOUR-USERNAME/rag-doctor.git

# Or manually:
git init
git checkout -b main
git add .
git commit -m "feat: initial release of rag-doctor v1.0.0"
git remote add origin https://github.com/YOUR-USERNAME/rag-doctor.git
git push -u origin main
```

---

## Step 3: Configure the Repository

### Topics (for discoverability)
Go to your repo → ⚙️ gear icon next to "About" → add topics:
```
rag  llm  evaluation  hallucination  retrieval  diagnosis  ai  python  open-source
```

### Branch Protection
Settings → Branches → Add rule for `main`:
- ✅ Require a pull request before merging
- ✅ Require status checks to pass (CI)
- ✅ Require linear history

### Secrets (for CI/CD)
Settings → Secrets and variables → Actions:
- `CODECOV_TOKEN` — from https://codecov.io (optional, for coverage badges)
- PyPI publishing uses OIDC trusted publishing (no token needed)

---

## Step 4: Set Up PyPI Publishing (Optional)

### Using Trusted Publishing (recommended, no API key needed)

1. Create account at https://pypi.org
2. Go to https://pypi.org/manage/account/publishing/
3. Add a new trusted publisher:
   - **PyPI Project Name**: `rag-doctor`
   - **Owner**: your GitHub username
   - **Repository name**: `rag-doctor`
   - **Workflow filename**: `publish.yml`
   - **Environment name**: `pypi`
4. Go to GitHub → Settings → Environments → New environment → name it `pypi`

### First publish
```bash
pip install build twine
python -m build
twine check dist/*

# Upload to TestPyPI first
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ rag-doctor

# If all good, upload to PyPI
twine upload dist/*
```

### Auto-publish on release
The `publish.yml` workflow triggers automatically when you create a GitHub Release.

---

## Step 5: Create the First Release

1. Go to your repo → Releases → **Draft a new release**
2. Tag: `v1.0.0`
3. Title: `rag-doctor v1.0.0 — Initial Release`
4. Description (paste this):

```markdown
## 🩺 rag-doctor v1.0.0

First public release of **rag-doctor** — an agentic RAG pipeline failure diagnosis tool.

### What's Included

- **6 diagnostic tools**: ChunkAnalyzer, RetrievalAuditor, PositionTester, HallucinationTracer, ChunkOptimizer, QueryRewriter
- **5 root cause IDs**: retrieval_miss (RC-1), context_position_bias (RC-2), chunk_fragmentation (RC-3), hallucination (RC-4), query_mismatch (RC-5)
- **CLI**: `rag-doctor diagnose` and `rag-doctor batch`
- **Python SDK**: `Doctor`, `RagDoctorConfig`, `DiagnosisReport`
- **62 tests** — 100% passing, no external APIs required
- **GitHub Actions**: CI across Python 3.9–3.12, PyPI publish workflow
- **Documentation**: developer guide, technical build spec, problem-solution scenarios

### Install

pip install rag-doctor

### Quick Start

rag-doctor diagnose \
  --query    "What is the termination notice period?" \
  --answer   "30 days." \
  --expected "Enterprise requires 90 days written notice."
```

5. Click **Publish release** → this triggers the PyPI publish workflow

---

## Step 6: Update README Badges

After publishing, update the badge URLs in `README.md`:

```markdown
[![PyPI version](https://badge.fury.io/py/rag-doctor.svg)](https://badge.fury.io/py/rag-doctor)
[![Downloads](https://pepy.tech/badge/rag-doctor)](https://pepy.tech/project/rag-doctor)
[![codecov](https://codecov.io/gh/YOUR-USERNAME/rag-doctor/branch/main/graph/badge.svg)](https://codecov.io/gh/YOUR-USERNAME/rag-doctor)
```

---

## Step 7: Promote the Project

### Post to these communities:

**Hacker News**
- Title: `Show HN: rag-doctor – Open-source tool to diagnose why your RAG pipeline failed`
- URL: your GitHub repo

**Reddit**
- r/MachineLearning
- r/LocalLLaMA
- r/Python

**Twitter/X hashtags**: `#RAG #LLM #OpenSource #AI`

**Dev.to / Hashnode** — write a technical post about the 5 root causes

---

## Maintenance Checklist

- [ ] Respond to issues within 48 hours
- [ ] Review and merge PRs weekly
- [ ] Update `CHANGELOG.md` for every release
- [ ] Run `python run_tests.py` before every merge to main
- [ ] Bump version in `pyproject.toml` before each release
