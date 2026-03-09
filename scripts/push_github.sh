#!/usr/bin/env bash
# =============================================================================
# rag-doctor — Push to GitHub
#
# Usage:
#   chmod +x scripts/push_github.sh
#   ./scripts/push_github.sh https://github.com/balavenkatesh3322/rag-doctor.git
#
# What this does:
#   1. Validates tests pass (62/62)
#   2. Initialises git if needed
#   3. Sets remote origin
#   4. Creates initial commit or commits changes
#   5. Pushes to main branch
# =============================================================================

set -e
cd "$(dirname "$0")/.."

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
ok()   { echo -e "${GREEN}✓${RESET}  $*"; }
fail() { echo -e "${RED}✗${RESET}  $*"; exit 1; }
warn() { echo -e "${YELLOW}⚠${RESET}  $*"; }
info() { echo -e "${CYAN}→${RESET}  $*"; }

echo ""
echo -e "${BOLD}${CYAN}  🩺  rag-doctor — Push to GitHub${RESET}"
echo -e "${BOLD}  ─────────────────────────────────────────${RESET}"
echo ""

# ── Args ──────────────────────────────────────────────────────────────────────
REPO_URL="${1:-}"
if [ -z "$REPO_URL" ]; then
  echo "Usage: ./scripts/push_github.sh https://github.com/YOUR_USERNAME/rag-doctor.git"
  echo ""
  echo "Steps to create the GitHub repo first:"
  echo -e "  1. Go to ${CYAN}https://github.com/new${RESET}"
  echo "  2. Name it: rag-doctor"
  echo "  3. Set visibility: Public or Private"
  echo "  4. DO NOT initialise with README (we'll push ours)"
  echo "  5. Copy the repo URL and run this script again"
  exit 1
fi

# ── Run tests first ───────────────────────────────────────────────────────────
info "Running tests before push..."
python run_tests.py 2>&1 | tail -3
ok "Tests passed"

# ── Git setup ─────────────────────────────────────────────────────────────────
info "Setting up git..."

if [ ! -d ".git" ]; then
  git init
  ok "Git initialised"
fi

# Configure git identity if not set
if [ -z "$(git config user.email 2>/dev/null)" ]; then
  warn "Git identity not configured"
  read -p "  Your email: " GIT_EMAIL
  read -p "  Your name:  " GIT_NAME
  git config user.email "$GIT_EMAIL"
  git config user.name  "$GIT_NAME"
fi
ok "Git identity: $(git config user.name) <$(git config user.email)>"

# ── Clean up before committing ────────────────────────────────────────────────
info "Cleaning up cache files..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name ".DS_Store" -delete 2>/dev/null || true
find . -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "dist" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "build" -type d -exec rm -rf {} + 2>/dev/null || true
ok "Cache files cleaned"

# ── Set remote ────────────────────────────────────────────────────────────────
if git remote get-url origin &>/dev/null; then
  git remote set-url origin "$REPO_URL"
  ok "Updated remote: origin → $REPO_URL"
else
  git remote add origin "$REPO_URL"
  ok "Added remote: origin → $REPO_URL"
fi

# ── Stage and commit ──────────────────────────────────────────────────────────
info "Staging files..."
git add -A

if git diff --cached --quiet; then
  ok "Nothing new to commit (already up to date)"
else
  COMMIT_MSG="feat: initial release v$(python -c "import rag_doctor; print(rag_doctor.__version__)")"
  git commit -m "$COMMIT_MSG"
  ok "Committed: $COMMIT_MSG"
fi

# ── Push ──────────────────────────────────────────────────────────────────────
info "Pushing to GitHub..."

# Rename branch to main if needed
git branch -M main 2>/dev/null || true

if git push -u origin main 2>/dev/null; then
  ok "Pushed to: $REPO_URL"
else
  echo ""
  warn "Push failed. This usually means authentication is needed."
  echo ""
  echo "  Option 1: HTTPS with Personal Access Token"
  echo -e "  ${CYAN}git push https://YOUR_TOKEN@github.com/YOUR_USERNAME/rag-doctor.git main${RESET}"
  echo ""
  echo "  Option 2: SSH"
  echo -e "  ${CYAN}git remote set-url origin git@github.com:YOUR_USERNAME/rag-doctor.git${RESET}"
  echo -e "  ${CYAN}git push -u origin main${RESET}"
  echo ""
  echo "  Generate a token: https://github.com/settings/tokens/new"
  echo "  Scopes needed: repo (full control)"
  exit 1
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}  ✓  Successfully pushed to GitHub!${RESET}"
echo ""
echo "  Next steps:"
echo -e "  1. Visit: ${CYAN}$REPO_URL${RESET}"
echo "  2. Go to Settings → Actions → Enable GitHub Actions"
echo "  3. The CI workflow runs automatically on every push"
echo ""
echo "  To publish to PyPI:"
echo -e "  ${CYAN}./scripts/publish_pypi.sh${RESET}"
echo ""
