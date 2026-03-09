#!/usr/bin/env bash
# =============================================================================
# rag-doctor — Local Mac Setup & Test Script
#
# Usage:
#   chmod +x scripts/test_local_mac.sh
#   ./scripts/test_local_mac.sh
#
# What this does:
#   1.  Checks Python 3.9+
#   2.  Creates a virtualenv (.venv-rag-doctor)
#   3.  Installs rag-doctor in editable mode
#   4.  Optionally installs sentence-transformers
#   5.  Runs 62 offline unit tests
#   6.  Runs all 6 sample scripts
#   7.  Runs all example scripts
#   8.  CLI smoke test (diagnose + batch)
#   9.  Optional Ollama integration test
# =============================================================================

set -e
cd "$(dirname "$0")/.."   # always run from repo root

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
ok()   { echo -e "${GREEN}✓${RESET}  $*"; }
fail() { echo -e "${RED}✗${RESET}  $*"; exit 1; }
warn() { echo -e "${YELLOW}⚠${RESET}  $*"; }
info() { echo -e "${CYAN}→${RESET}  $*"; }
hr()   { echo -e "${BOLD}  ─────────────────────────────────────────${RESET}"; }

echo ""
echo -e "${BOLD}${CYAN}  🩺  rag-doctor — Local Mac Setup & Test${RESET}"
hr
echo ""

# =============================================================================
# 1. Python version check
# =============================================================================
info "Checking Python version..."
PYTHON=""
for cmd in python3.12 python3.11 python3.10 python3.9 python3 python; do
  if command -v "$cmd" &>/dev/null; then
    version=$("$cmd" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
    major=$(echo "$version" | cut -d. -f1)
    minor=$(echo "$version" | cut -d. -f2)
    if [ "$major" -ge 3 ] && [ "$minor" -ge 9 ]; then
      PYTHON=$cmd; break
    fi
  fi
done
[ -z "$PYTHON" ] && fail "Python 3.9+ not found. Install with: brew install python"
ok "Python: $($PYTHON --version)"

# =============================================================================
# 2. Virtual environment
# =============================================================================
info "Setting up virtual environment..."
VENV=".venv-rag-doctor"
if [ ! -d "$VENV" ]; then
  $PYTHON -m venv "$VENV"
  ok "Created: $VENV"
else
  ok "Exists: $VENV"
fi
# shellcheck disable=SC1090
source "$VENV/bin/activate"
ok "Activated: $VENV"

# =============================================================================
# 3. Install dependencies
# =============================================================================
info "Installing rag-doctor (editable)..."
pip install --upgrade pip -q
pip install -e "." -q
ok "rag-doctor installed"

echo ""
echo -e "  Install ${CYAN}sentence-transformers${RESET} for semantic embeddings? (~90MB, one-time)"
echo -e "  (Without it, TF-IDF is used — all 62 tests still pass)"
read -p "  [y/N]: " INSTALL_ST
if [[ $INSTALL_ST =~ ^[Yy]$ ]]; then
  pip install sentence-transformers -q && ok "sentence-transformers installed"
else
  info "Skipping. TF-IDF fallback will be used."
fi

# =============================================================================
# 4. Unit tests (62 tests, fully offline)
# =============================================================================
echo ""; hr
echo -e "${BOLD}  Step 4 — Unit Tests (offline, no Ollama needed)${RESET}"; hr; echo ""
python run_tests.py || fail "Unit tests failed"

# =============================================================================
# 5. Samples
# =============================================================================
echo ""; hr
echo -e "${BOLD}  Step 5 — Samples${RESET}"; hr; echo ""

run_sample() {
  local script="$1"
  local allow_nonzero="${2:-false}"
  echo -n "  "
  info "Running $script ..."
  if python "$script" > /tmp/rout.txt 2>&1; then
    ok "$script"
  elif [ "$allow_nonzero" = "true" ]; then
    ok "$script (exit 1 = issues detected, correct for CI gate)"
  else
    warn "$script FAILED"
    head -8 /tmp/rout.txt
  fi
}

run_sample "samples/01_basic_diagnosis.py"
run_sample "samples/02_from_logs.py"
run_sample "samples/03_batch_ci_gate.py"  "true"   # exits 1 when issues found — correct
run_sample "samples/04_custom_connector.py"
run_sample "samples/05_all_root_causes.py"
run_sample "samples/06_json_report.py"

# =============================================================================
# 6. Example scripts
# =============================================================================
echo ""; hr
echo -e "${BOLD}  Step 6 — Example Scripts${RESET}"; hr; echo ""

run_sample "examples/quickstart.py"
run_sample "examples/debug_from_logs.py"
run_sample "examples/batch_diagnose.py"   "true"   # exits 1 when issues found
run_sample "examples/ci_quality_gate.py"
run_sample "examples/custom_connector.py"

# =============================================================================
# 7. CLI smoke test
# =============================================================================
echo ""; hr
echo -e "${BOLD}  Step 7 — CLI Smoke Test${RESET}"; hr; echo ""

info "rag-doctor diagnose ..."
rag-doctor diagnose \
  --query    "What is the termination notice period?" \
  --answer   "30 days required." \
  --expected "Enterprise requires 90 days written notice." \
  > /tmp/cli_out.txt 2>&1 || true   # exit 1 is correct — issue found
ok "rag-doctor diagnose (exit 1 = issue detected, correct)"
head -4 /tmp/cli_out.txt | sed 's/^/    /'

echo ""
info "rag-doctor batch ..."
rag-doctor batch \
  --input examples/batch_example.jsonl \
  --fail-on-severity high \
  > /tmp/batch_out.txt 2>&1 || true
ok "rag-doctor batch"
head -3 /tmp/batch_out.txt | sed 's/^/    /'

# =============================================================================
# 8. Ollama integration test (optional)
# =============================================================================
echo ""; hr
echo -e "${BOLD}  Step 8 — Ollama Integration (optional)${RESET}"; hr; echo ""

if curl -s --max-time 2 http://localhost:11434/api/tags > /dev/null 2>&1; then
  ok "Ollama is running"
  MODEL_COUNT=$(curl -s http://localhost:11434/api/tags | \
    $PYTHON -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('models',[])))" 2>/dev/null || echo "0")
  if [ "$MODEL_COUNT" = "0" ]; then
    warn "No models installed."
    echo -e "    ${CYAN}ollama pull llama3.2${RESET}  (~2GB)"
    echo -e "    ${CYAN}ollama pull nomic-embed-text${RESET}  (~274MB, better embeddings)"
  else
    read -p "  Run Ollama integration tests? [y/N]: " RUN_OLLAMA
    if [[ $RUN_OLLAMA =~ ^[Yy]$ ]]; then
      python test_with_ollama.py
    else
      info "Skipped. Run manually: python test_with_ollama.py"
    fi
  fi
else
  warn "Ollama not running"
  echo -e "    Install: ${CYAN}brew install ollama${RESET}"
  echo -e "    Start  : ${CYAN}ollama serve${RESET}  (separate terminal)"
  echo -e "    Model  : ${CYAN}ollama pull llama3.2${RESET}"
fi

# =============================================================================
# Done
# =============================================================================
echo ""
echo -e "${BOLD}${GREEN}  ✓  All done!${RESET}"
echo ""
echo "  Quick reference:"
echo -e "  ${CYAN}source $VENV/bin/activate${RESET}            # activate venv next time"
echo -e "  ${CYAN}python run_tests.py${RESET}                  # 62 offline unit tests"
echo -e "  ${CYAN}python samples/05_all_root_causes.py${RESET} # see all 6 root causes"
echo -e "  ${CYAN}python samples/03_batch_ci_gate.py${RESET}   # CI quality gate demo"
echo -e "  ${CYAN}rag-doctor diagnose --help${RESET}           # CLI help"
echo -e "  ${CYAN}./scripts/push_github.sh REPO_URL${RESET}   # push to GitHub"
echo -e "  ${CYAN}./scripts/publish_pypi.sh${RESET}           # publish to PyPI"
echo ""
