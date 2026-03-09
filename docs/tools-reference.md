# Tools Reference

All six diagnostic tools share the same interface: `tool.run(**kwargs) → ToolResult`.

---

## ToolResult

```python
@dataclass
class ToolResult:
    tool_name: str
    passed: bool
    finding: str
    severity: str          # "low" | "medium" | "high" | "critical"
    details: dict          # tool-specific metrics
    recommendation: str | None
```

All tool details dicts include an `"embedder"` key indicating which backend was used.

---

## RetrievalAuditor

**Root Cause: RC-1 `retrieval_miss`**

Checks whether the expected answer is semantically present in retrieved documents.

```python
from rag_doctor.tools.retrieval_auditor import RetrievalAuditor

tool = RetrievalAuditor(recall_threshold=0.35)  # raise to 0.65 with sentence-transformers
result = tool.run(docs=docs, expected="expected answer text", query="user query")
```

**Parameters:**
- `recall_threshold` (float, default 0.35) — minimum cosine similarity between expected answer and best-matching doc sentence

**Details dict:**
```json
{
  "recall_hit": true,
  "best_match_position": 0,
  "best_match_score": 0.71,
  "scores_by_position": {0: 0.71, 1: 0.42, 2: 0.08},
  "recall_threshold": 0.35,
  "embedder": "tfidf"
}
```

**Without `expected`:** falls back to checking whether `max(d.score for d in docs) >= threshold`.

---

## PositionTester

**Root Cause: RC-2 `context_position_bias`**

Detects whether the most relevant document is in a middle "danger zone" where LLMs statistically underperform (Liu et al. 2023).

```python
from rag_doctor.tools.position_tester import PositionTester

result = PositionTester().run(docs=docs, query=query, expected=expected)
```

**Logic:**
- Finds best document by: expected-answer similarity > retrieval score > query similarity
- With `n >= 3` docs, any position `0 < pos < n-1` is the danger zone
- Risk score = `min(pos, n-1-pos) / ((n-1)/2)` — peaks at exact middle

**Details dict:**
```json
{
  "best_position": 1,
  "total_docs": 3,
  "in_danger_zone": true,
  "position_risk_score": 1.0
}
```

**Only runs** when `retrieval_auditor` passed and `len(docs) >= 3`.

---

## ChunkAnalyzer

**Root Cause: RC-3 `chunk_fragmentation`**

Analyzes retrieved chunks for truncation, incoherence, and size issues.

```python
from rag_doctor.tools.chunk_analyzer import ChunkAnalyzer

tool = ChunkAnalyzer(coherence_threshold=0.25)  # raise to 0.55 with sentence-transformers
result = tool.run(docs=docs, query=query)
```

**Checks:**
1. **Truncation detection** — starts lowercase or ends without `.!?` → mid-sentence cut
2. **Intra-chunk coherence** — average cosine similarity between adjacent sentences in each chunk
3. **Size analysis** — ideal range 80–800 tokens; flags too-small and too-large chunks

**Details dict:**
```json
{
  "avg_coherence": 0.41,
  "avg_tokens": 145.3,
  "truncated_chunks": 1,
  "total_chunks": 3,
  "coherence_threshold": 0.25,
  "issues": ["pos=0: too small (12 tokens)"],
  "embedder": "tfidf"
}
```

---

## HallucinationTracer

**Root Cause: RC-4 `hallucination`**

Checks whether each claim in the generated answer is grounded in at least one source sentence.

```python
from rag_doctor.tools.hallucination_tracer import HallucinationTracer

tool = HallucinationTracer(faithfulness_threshold=0.40)
result = tool.run(answer=answer, docs=docs)
```

**Algorithm:**
1. Split answer into atomic claims at sentence boundaries
2. Embed all claims + all document sentences **together** in one batch (critical for TF-IDF correctness)
3. Per claim: find maximum cosine similarity to any doc sentence
4. If similarity >= threshold → grounded; else → hallucinated
5. Faithfulness = grounded_claims / total_claims

**Severity:**
- `< 0.35` → critical
- `< 0.60` → high
- `< 0.80` → medium
- `>= 0.80` → low (passed at 0.80)

**Details dict:**
```json
{
  "faithfulness_score": 0.33,
  "total_claims": 3,
  "grounded_claims": 1,
  "hallucinated_claims": [
    {"claim": "This supports --quantum flag.", "best_similarity": 0.12}
  ],
  "embedder": "tfidf"
}
```

---

## ChunkOptimizer

**Root Cause sub-tool: RC-3 `chunk_fragmentation`**

Grid-searches five chunking strategies and measures recall@5 for each.

```python
from rag_doctor.tools.chunk_optimizer import ChunkOptimizer

result = ChunkOptimizer().run(
    corpus_texts  = ["Full document text 1...", "Full document text 2..."],
    test_pairs    = [{"query": "q1", "expected": "a1"}, ...],
    current_strategy = {"strategy": "fixed", "chunk_size": 512, "chunk_overlap": 64},
)
```

**Strategies tested:**
| Strategy | chunk_size | chunk_overlap |
|---|---|---|
| fixed | 128 | 16 |
| fixed | 256 | 32 |
| fixed | 512 | 64 |
| fixed | 1024 | 128 |
| recursive | 256 | 32 |

**Details dict:**
```json
{
  "ranked_strategies": [
    {"strategy": "recursive", "chunk_size": 256, "chunk_overlap": 32, "recall_at_5": 0.82, "num_chunks": 18},
    {"strategy": "fixed",     "chunk_size": 512, "chunk_overlap": 64, "recall_at_5": 0.71, "num_chunks": 9}
  ],
  "current_recall": 0.71,
  "best_improvement": 0.11
}
```

**Only runs** when `ChunkAnalyzer` failed and `corpus_texts` were provided.

---

## QueryRewriter

**Root Cause sub-tool: RC-5 `query_mismatch`**

Rewrites the query using three strategies and checks if retrieval improves.

```python
from rag_doctor.tools.query_rewriter import QueryRewriter

result = QueryRewriter().run(
    query        = "How many weeks maternity leave?",
    original_docs = docs,
    connector    = connector,  # optional — if provided, actually re-retrieves
    expected     = "16 weeks paid parental leave.",
)
```

**Strategies:**
- **HyDE** — turns question into a hypothetical answer document (improves when doc vocabulary is declarative but query is interrogative)
- **Step-back** — broadens to higher abstraction level
- **Synonym expansion** — replaces domain terms: maternity→parental, fired→terminated, pto→paid time off, etc.

**Details dict:**
```json
{
  "original_query": "How many weeks maternity leave?",
  "strategy_results": {
    "original": {"query": "...", "top_score": 0.21},
    "synonym":  {"query": "How many weeks maternity parental leave?", "top_score": 0.68}
  },
  "best_strategy": "synonym",
  "best_score": 0.68,
  "improvement": 0.47,
  "embedder": "tfidf"
}
```

**Only runs** when `RetrievalAuditor` failed.
