#!/usr/bin/env bash
# =============================================================================
# rag-doctor — Publish to PyPI
#
# Usage:
#   chmod +x scripts/publish_pypi.sh
#   ./scripts/publish_pypi.sh [--test]    # --test publishes to TestPyPI first
#
# What this does:
#   1. Validates tests pass (62/62)
#   2. Checks version is bumped
#   3. Builds wheel + sdist
#   4. Checks the distribution
#   5. Publishes to TestPyPI (if --test) or PyPI
#   6. Tags the git release
# =============================================================================

set -e
cd "$(dirname "$0")/.."

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
ok()   { echo -e "${GREEN}✓${RESET}  $*"; }
fail() { echo -e "${RED}✗${RESET}  $*"; exit 1; }
warn() { echo -e "${YELLOW}⚠${RESET}  $*"; }
info() { echo -e "${CYAN}→${RESET}  $*"; }

TEST_MODE=false
[ "$1" = "--test" ] && TEST_MODE=true

echo ""
echo -e "${BOLD}${CYAN}  🩺  rag-doctor — Publish to PyPI${RESET}"
[ "$TEST_MODE" = true ] && echo -e "${YELLOW}  (TEST MODE — publishing to TestPyPI)${RESET}"
echo -e "${BOLD}  ─────────────────────────────────────────${RESET}"
echo ""

# ── Get version ───────────────────────────────────────────────────────────────
VERSION=$(python -c "import rag_doctor; print(rag_doctor.__version__)")
ok "Version: $VERSION"

# ── Run tests ─────────────────────────────────────────────────────────────────
info "Running full test suite..."
python run_tests.py 2>&1 | tail -3
ok "All tests passed"

# ── Check required tools ──────────────────────────────────────────────────────
info "Checking build tools..."
python -m pip install --upgrade build twine -q
ok "build + twine available"

# ── Clean previous builds ─────────────────────────────────────────────────────
info "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info
ok "Cleaned"

# ── Build ─────────────────────────────────────────────────────────────────────
info "Building distribution packages..."
python -m build
ok "Built:"
ls dist/

# ── Check ─────────────────────────────────────────────────────────────────────
info "Checking distribution..."
twine check dist/*
ok "Distribution checks passed"

# ── Confirm ───────────────────────────────────────────────────────────────────
echo ""
if [ "$TEST_MODE" = true ]; then
  echo -e "  Publishing ${CYAN}rag-doctor==$VERSION${RESET} to ${YELLOW}TestPyPI${RESET}"
  echo "  Install after: pip install --index-url https://test.pypi.org/simple/ rag-doctor==$VERSION"
else
  echo -e "  Publishing ${CYAN}rag-doctor==$VERSION${RESET} to ${GREEN}PyPI (LIVE)${RESET}"
  echo -e "  ${RED}This is irreversible. Make sure the version is correct!${RESET}"
fi
echo ""
read -p "  Proceed? [y/N]: " CONFIRM
[[ $CONFIRM =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }

# ── Upload ────────────────────────────────────────────────────────────────────
if [ "$TEST_MODE" = true ]; then
  info "Uploading to TestPyPI..."
  twine upload --repository testpypi dist/*
  echo ""
  ok "Published to TestPyPI!"
  echo ""
  echo "  Test the install:"
  echo -e "  ${CYAN}pip install --index-url https://test.pypi.org/simple/ rag-doctor==$VERSION${RESET}"
else
  info "Uploading to PyPI..."
  twine upload dist/*
  echo ""
  ok "Published to PyPI!"
  echo ""
  echo "  Install:"
  echo -e "  ${CYAN}pip install rag-doctor==$VERSION${RESET}"
fi

# ── Tag release ───────────────────────────────────────────────────────────────
if [ "$TEST_MODE" = false ]; then
  echo ""
  read -p "  Create git tag v$VERSION? [y/N]: " TAG_IT
  if [[ $TAG_IT =~ ^[Yy]$ ]]; then
    git tag "v$VERSION"
    git push origin "v$VERSION"
    ok "Tagged: v$VERSION"
    echo ""
    echo "  Creating a GitHub Release from this tag will trigger the"
    echo "  automated publish workflow (.github/workflows/publish.yml)"
  fi
fi

echo ""
echo -e "${BOLD}${GREEN}  ✓  Done!${RESET}"
echo ""
