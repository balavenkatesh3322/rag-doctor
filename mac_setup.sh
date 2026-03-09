#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# rag-doctor Mac Setup & Test Script
#
# Usage:
#   chmod +x mac_setup.sh
#   ./mac_setup.sh
#
# What this does:
#   1. Checks Python 3.9+
#   2. Creates a virtual environment
#   3. Installs dependencies
#   4. Runs the offline test suite (no Ollama needed)
#   5. Checks if Ollama is available and offers to run Ollama tests
# ─────────────────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")"

# ─── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
ok()   { echo -e "${GREEN}✓${RESET}  $1"; }
err()  { echo -e "${RED}✗${RESET}  $1"; exit 1; }
warn() { echo -e "${YELLOW}⚠${RESET}  $1"; }
info() { echo -e "${CYAN}→${RESET}  $1"; }

echo ""
echo -e "${BOLD}${CYAN}  🩺  rag-doctor — Mac Setup & Test${RESET}"
echo -e "${BOLD}  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""

# ─── Step 1: Python version ───────────────────────────────────────────────────
info "Checking Python version..."
PYTHON=""
for cmd in python3.12 python3.11 python3.10 python3.9 python3 python; do
  if command -v $cmd &>/dev/null; then
    version=$($cmd --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
    major=$(echo $version | cut -d. -f1)
    minor=$(echo $version | cut -d. -f2)
    if [ "$major" -ge 3 ] && [ "$minor" -ge 9 ]; then
      PYTHON=$cmd
      break
    fi
  fi
done

if [ -z "$PYTHON" ]; then
  err "Python 3.9+ not found. Install with: brew install python"
fi
ok "Python: $($PYTHON --version)"

# ─── Step 2: Virtual environment ──────────────────────────────────────────────
info "Setting up virtual environment..."
VENV=".venv-rag-doctor"

if [ ! -d "$VENV" ]; then
  $PYTHON -m venv $VENV
  ok "Created virtualenv: $VENV"
else
  ok "Virtualenv exists: $VENV"
fi

source $VENV/bin/activate
ok "Activated: $VENV"

# ─── Step 3: Install dependencies ────────────────────────────────────────────
info "Installing dependencies..."
pip install --upgrade pip -q
pip install pyyaml -q
ok "Dependencies installed (pyyaml)"

# ─── Step 4: Offline unit tests ───────────────────────────────────────────────
echo ""
echo -e "${BOLD}  ── Offline Tests (no Ollama needed) ──${RESET}"
echo ""

python run_tests.py
echo ""

# ─── Step 5: Check Ollama ─────────────────────────────────────────────────────
echo -e "${BOLD}  ── Ollama Integration Test ──${RESET}"
echo ""

OLLAMA_RUNNING=false
if curl -s --max-time 2 http://localhost:11434/api/tags > /dev/null 2>&1; then
  OLLAMA_RUNNING=true
  ok "Ollama is running"
else
  warn "Ollama is not running"
  echo ""
  echo "  To install and start Ollama on Mac:"
  echo -e "  ${CYAN}brew install ollama${RESET}"
  echo -e "  ${CYAN}ollama serve${RESET}            # run in a separate terminal"
  echo -e "  ${CYAN}ollama pull llama3.2${RESET}    # or: mistral, phi3, gemma2"
  echo ""
fi

if [ "$OLLAMA_RUNNING" = true ]; then
  echo "  Checking installed models..."
  MODELS=$(curl -s http://localhost:11434/api/tags | python3 -c "
import sys, json
data = json.load(sys.stdin)
models = data.get('models', [])
if models:
    for m in models:
        print('   •', m['name'])
else:
    print('   (none installed)')
" 2>/dev/null || echo "   (could not parse)")
  echo "$MODELS"
  echo ""

  # Check if any models installed
  MODEL_COUNT=$(curl -s http://localhost:11434/api/tags | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(len(data.get('models', [])))
" 2>/dev/null || echo "0")

  if [ "$MODEL_COUNT" = "0" ]; then
    warn "No models installed. Pull one to run Ollama tests:"
    echo -e "  ${CYAN}ollama pull llama3.2${RESET}"
    echo -e "  ${CYAN}ollama pull mistral${RESET}"
    echo -e "  ${CYAN}ollama pull phi3${RESET}"
  else
    echo -e "  Run Ollama integration test? (benchmarks models + runs 5 scenarios)"
    read -p "  [y/N]: " RUN_OLLAMA
    if [[ $RUN_OLLAMA =~ ^[Yy]$ ]]; then
      echo ""
      python test_with_ollama.py
    else
      echo ""
      info "Skipped. Run manually: ${CYAN}python test_with_ollama.py${RESET}"
    fi
  fi
fi

# ─── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}  ✓ Setup complete!${RESET}"
echo ""
echo "  Activate the venv for future use:"
echo -e "  ${CYAN}source $VENV/bin/activate${RESET}"
echo ""
echo "  Available commands:"
echo -e "  ${CYAN}python run_tests.py${RESET}                    # offline tests"
echo -e "  ${CYAN}python test_with_ollama.py${RESET}             # Ollama tests (auto model select)"
echo -e "  ${CYAN}python test_with_ollama.py --benchmark-only${RESET}  # benchmark models only"
echo -e "  ${CYAN}python test_with_ollama.py --quick${RESET}     # skip benchmark"
echo -e "  ${CYAN}python examples/quickstart.py${RESET}          # run examples"
echo -e "  ${CYAN}./setup.sh https://github.com/YOU/rag-doctor.git${RESET}  # push to GitHub"
echo ""
