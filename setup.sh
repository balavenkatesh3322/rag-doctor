#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# rag-doctor GitHub Push Script
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh https://github.com/YOUR-USERNAME/rag-doctor.git
#
# What this does:
#   1. Initialises a git repo (if not already)
#   2. Creates a proper initial commit
#   3. Pushes to GitHub
# ─────────────────────────────────────────────────────────────────────────────

set -e

REMOTE_URL="${1}"

if [ -z "$REMOTE_URL" ]; then
  echo "❌  Usage: ./setup.sh https://github.com/YOUR-USERNAME/rag-doctor.git"
  exit 1
fi

echo ""
echo "🩺  rag-doctor — GitHub Setup"
echo "================================"
echo "Remote: $REMOTE_URL"
echo ""

# Init git if needed
if [ ! -d ".git" ]; then
  git init
  echo "✓  git init"
else
  echo "✓  git already initialised"
fi

# Set main as default branch
git checkout -b main 2>/dev/null || git checkout main

# Stage everything
git add .

# Initial commit
git commit -m "feat: initial release of rag-doctor v1.0.0

- Six diagnostic tools: ChunkAnalyzer, RetrievalAuditor, PositionTester,
  HallucinationTracer, ChunkOptimizer, QueryRewriter
- Doctor orchestrator with deterministic agent loop
- 5 root cause IDs: RC-1 through RC-5
- CLI: diagnose + batch commands
- Python SDK: Doctor, RagDoctorConfig, DiagnosisReport
- MockConnector for full offline testing
- 62 tests, 100% passing, no external APIs required
- GitHub Actions: CI, publish to PyPI, RAG quality gate
- Documentation: 3 docx files in docs/"

echo "✓  initial commit created"

# Add remote and push
git remote add origin "$REMOTE_URL" 2>/dev/null || git remote set-url origin "$REMOTE_URL"
git push -u origin main

echo ""
echo "✅  Pushed to $REMOTE_URL"
echo ""
echo "Next steps:"
echo "  1. Go to your repo → Settings → About → add topics:"
echo "     rag, llm, evaluation, hallucination, retrieval, open-source"
echo "  2. Create a release: v1.0.0"
echo "  3. Enable GitHub Actions in the Actions tab"
echo "  4. Optional: add CODECOV_TOKEN secret for coverage reports"
echo "  5. Publish to PyPI: python -m build && twine upload dist/*"
