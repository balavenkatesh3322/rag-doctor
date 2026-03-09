# Architecture

This document explains the internal design of rag-doctor: how embeddings work, how the VectorStore avoids shared state, and how the agent loop makes decisions.

---

## Module Map

```
rag_doctor/
├── __init__.py             # Public API: Doctor, RagDoctorConfig, DiagnosisReport
├── doctor.py               # Orchestrator — runs the six-tool agent loop
├── config.py               # Dataclass config with calibrated defaults
├── report.py               # DiagnosisReport dataclass + serialization
├── embeddings.py           # 4-level embedding chain (auto-selected)
├── vector_store.py         # Ephemeral in-memory VectorStore
├── cli.py                  # argparse CLI (diagnose + batch commands)
├── connectors/
│   ├── base.py             # Document dataclass + PipelineConnector ABC
│   ├── mock.py             # MockConnector — in-memory, real embeddings
│   └── ollama_connector.py # OllamaConnector — local LLM pipeline
└── tools/
    ├── _embed_utils.py     # fit_and_embed() — correct TF-IDF vocab for all tools
    ├── base.py             # BaseTool ABC + ToolResult dataclass
    ├── chunk_analyzer.py   # RC-3: truncation + intra-chunk coherence
    ├── retrieval_auditor.py# RC-1: recall@k with ground truth
    ├── position_tester.py  # RC-2: lost-in-the-middle detection
    ├── hallucination_tracer.py # RC-4: claim grounding
    ├── chunk_optimizer.py  # RC-3 sub-tool: grid search chunking strategies
    └── query_rewriter.py   # RC-5 sub-tool: HyDE + step-back + synonyms
```

---

## Embedding Chain

The embedding engine auto-selects the best available backend on first use:

```
SentenceTransformerEmbedder   ← pip install sentence-transformers
        │  (not available)
        ▼
OllamaEmbedder                ← ollama serve + ollama pull nomic-embed-text
        │  (not available)
        ▼
TFIDFEmbedder                 ← stdlib + numpy, no install needed
        │  (numpy not available)
        ▼
CharFreqEmbedder              ← stdlib only, always available
```

### Critical Design: TF-IDF State Isolation

Each `VectorStore` instance owns its own `TFIDFEmbedder`. They are **never shared** via the global cache.

Why: TF-IDF vocabulary is built from the corpus it's fitted on. If two connectors with different corpora shared an embedder, the second would use the first's vocabulary and produce wrong similarity scores. This was the root cause of a test interference bug fixed in v1.1.

```python
# CORRECT (what we do): fresh TFIDFEmbedder per VectorStore
class VectorStore:
    def _get_or_create_embedder(self, corpus_texts=None):
        cached = get_embedder(...)
        if isinstance(cached, TFIDFEmbedder):
            self._embedder = TFIDFEmbedder(dim=cached._dim)  # fresh copy
        ...

# WRONG (what we don't do): sharing global TFIDFEmbedder
self._embedder = get_embedder()  # ← breaks between tests
```

### Critical Design: fit_and_embed() in Tools

Every tool that compares texts (HallucinationTracer, ChunkAnalyzer, RetrievalAuditor, etc.) must embed **all** texts it will compare in a **single batch**, so TF-IDF vocabulary covers both sides.

```python
# CORRECT: all texts embedded together in one call
all_texts = [expected] + [d.content for d in docs] + claim_sentences
e, all_vecs = fit_and_embed(all_texts)
exp_vec   = all_vecs[0]
doc_vecs  = all_vecs[1:len(docs)+1]

# WRONG: embedding sentence-by-sentence
exp_vec = embedder.embed(expected)   # ← fitted on expected vocab only
for doc in docs:
    doc_vec = embedder.embed(doc.content)  # ← different vocab, wrong similarity
```

---

## VectorStore

`VectorStore` is a simple list of `(Document, embedding)` pairs with cosine similarity search.

```python
store = VectorStore()
store.add_batch([{"id": "d1", "content": "..."}, ...])
results = store.search("my query", top_k=5)  # → List[Document] sorted by score
```

**add_batch() flow:**
1. Collect all texts from items
2. `_get_or_create_embedder(corpus_texts=texts)` — creates fresh TFIDFEmbedder and fits it on all texts at once
3. `embedder.embed_batch(texts)` — batch embed for efficiency (sentence-transformers uses GPU batching)
4. Store `(Document, embedding)` pairs

---

## Agent Loop

`Doctor.diagnose()` runs tools in priority order:

```python
def diagnose(self, query, answer, docs, expected, corpus_texts):
    # Step 1: always run
    retrieval = RetrievalAuditor.run(docs, expected, query)

    # Step 2: only if retrieval passed
    if not retrieval_miss and len(docs) >= 3:
        position = PositionTester.run(docs, query, expected)

    # Step 3: always run
    chunk = ChunkAnalyzer.run(docs, query)

    # Step 4: always run
    hallucination = HallucinationTracer.run(answer, docs)

    # Step 5: only if retrieval missed
    if retrieval_miss:
        rewrite = QueryRewriter.run(query, docs, connector, expected)

    # Step 6: only if fragmentation AND corpus_texts provided
    if chunk_failed and corpus_texts:
        optimizer = ChunkOptimizer.run(corpus_texts, test_pairs)

    # Root cause attribution (priority order)
    # RC-1/RC-5 → RC-2 → RC-3 → RC-4 → RC-0
```

**Root cause priority:**
1. Retrieval miss always wins — if the right doc isn't in context, nothing else matters
2. Position bias — doc was retrieved but placed where LLMs ignore it
3. Chunk fragmentation — but only if severity != low (minor quality issues don't override)
4. Hallucination — answer not grounded in what was retrieved
5. Healthy — all tools passed

---

## Configuration and Threshold Calibration

Thresholds are calibrated for **TF-IDF** by default (the most constrained backend). TF-IDF cosine similarity tops out at ~0.60 for paraphrases, so thresholds are set below that.

| Threshold | TF-IDF default | sentence-transformers recommended |
|---|---|---|
| `recall_threshold` | 0.35 | 0.65 |
| `faithfulness_threshold` | 0.40 | 0.65 |
| `coherence_threshold` | 0.25 | 0.55 |

With sentence-transformers, paraphrase similarity reaches 0.85–0.95, so thresholds can be raised significantly for sharper diagnosis.

---

## Data Flow Diagram

```
User Input
  query: str
  answer: str
  docs: List[Document]  ← may come from user's DB, logs, or MockConnector
  expected: str (optional)
  corpus_texts: List[str] (optional, for ChunkOptimizer)
        │
        ▼
  Doctor.diagnose()
        │
        ├─ VectorStore (ephemeral, built from docs if passed)
        │    └─ TFIDFEmbedder (per-instance, fitted on corpus)
        │
        ├─ fit_and_embed(all_comparison_texts) per tool
        │    └─ fresh TFIDFEmbedder per tool call
        │
        └─ DiagnosisReport
             ├─ root_cause + severity + finding
             ├─ fix_suggestion (human-readable)
             ├─ config_patch (machine-readable YAML patch)
             └─ tool_results (one ToolResult per tool that ran)
```
