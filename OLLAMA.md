# Ollama Integration Guide

Test rag-doctor locally using a completely free, offline LLM via Ollama.

---

## Quick Setup (Mac)

```bash
# 1. Install Ollama
brew install ollama

# 2. Start Ollama server (keep this terminal open)
ollama serve

# 3. Pull a model (pick one)
ollama pull llama3.2         # recommended — fast, good quality (~2GB)
ollama pull mistral          # alternative (~4GB)
ollama pull phi3             # smallest, fastest (~2GB)
ollama pull gemma2:9b        # high quality (~5GB)

# 4. Run the Mac setup + test script
./mac_setup.sh
```

---

## Manual Ollama Test

```bash
# Auto-benchmark all installed models and run all scenarios
python test_with_ollama.py

# Use a specific model (skip benchmark)
python test_with_ollama.py --model llama3.2

# Just benchmark models, don't run scenarios
python test_with_ollama.py --benchmark-only

# Quick mode — skip benchmark, use first available model
python test_with_ollama.py --quick

# Run only one scenario (1-5)
python test_with_ollama.py --scenario 2

# List installed models
python test_with_ollama.py --list-models
```

---

## Model Recommendations

| Model | Size | Speed | Quality | Best For |
|---|---|---|---|---|
| `llama3.2:3b` | 2GB | ⚡⚡⚡ | ★★★★ | Default — best balance |
| `llama3.1:8b` | 5GB | ⚡⚡ | ★★★★★ | Best quality |
| `mistral:7b` | 4GB | ⚡⚡ | ★★★★ | Good all-rounder |
| `phi3:mini` | 2GB | ⚡⚡⚡ | ★★★ | Fastest |
| `gemma2:9b` | 5GB | ⚡⚡ | ★★★★★ | Best quality |

For **embeddings** (better retrieval):
```bash
ollama pull nomic-embed-text    # best embedding model
ollama pull mxbai-embed-large   # alternative
```

---

## Python SDK with Ollama

```python
from rag_doctor import Doctor
from rag_doctor.connectors.ollama_connector import OllamaConnector

# Auto-select best installed model
connector = OllamaConnector()

# Or specify explicitly
connector = OllamaConnector(
    model="llama3.2",
    embed_model="nomic-embed-text",  # optional, better embeddings
    temperature=0.1,                  # low = more deterministic
)

# Load your documents
connector.load_corpus([
    {"id": "doc1", "content": "Your document text here...", "metadata": {}},
])

# Use with rag-doctor
doctor = Doctor.default(connector=connector)
docs = connector.retrieve("your query", top_k=5)
answer = connector.generate("your query", docs)
report = doctor.diagnose(query="your query", answer=answer, docs=docs)
print(report.to_text())
```

---

## Model Benchmarking

The `select_and_benchmark()` function tests every installed model on a RAG task and ranks them:

```python
from rag_doctor.connectors.model_selector import select_and_benchmark
best_model = select_and_benchmark(verbose=True)
# Output:
# Testing llama3.2:latest...  ✅ quality=1.0 speed=42t/s latency=2.1s score=0.968
# Testing phi3:mini...        ⚠️  quality=0.5 speed=61t/s latency=1.4s score=0.544
# 🏆 Best model: llama3.2:latest
```

Scoring: 60% quality + 40% speed (normalized to 30 tokens/s ceiling).

---

## Troubleshooting

**`ConnectionError: Cannot reach Ollama`**
```bash
ollama serve    # make sure this is running in another terminal
```

**`No models found`**
```bash
ollama pull llama3.2
```

**Slow responses**
- Use a smaller model: `phi3:mini` or `llama3.2:1b`
- Make sure Ollama has GPU access (Mac M-series chips work automatically)

**Poor diagnosis quality**
- Use a larger model: `llama3.1:8b` or `gemma2:9b`
- Add an embedding model: `ollama pull nomic-embed-text`
