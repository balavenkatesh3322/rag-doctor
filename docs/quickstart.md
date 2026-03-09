# Quick Start

Get your first diagnosis in 5 minutes.

---

## Step 1: Install

```bash
# Minimum — works with TF-IDF, no API keys needed
pip install rag-doctor

# Recommended — enables semantic embeddings
pip install rag-doctor sentence-transformers
```

Requires Python 3.9+. The only hard dependency is `pyyaml` (for config). `numpy` is used automatically if available (required for TF-IDF).

---

## Step 2: Your First Diagnosis (CLI)

```bash
rag-doctor diagnose \
  --query  "What is the maximum acetaminophen dose for liver disease?" \
  --answer "The maximum daily dose is 4000mg."
```

Output:

```
══════════════════════════════════════════════════════════════
  RAG-DOCTOR ✗ ISSUES FOUND
══════════════════════════════════════════════════════════════
  Root Cause : hallucination (RC-4)
  Severity   : HIGH
  Finding    : 1/2 answer claims not grounded in retrieved docs.
──────────────────────────────────────────────────────────────
  Tool Results:
    ✓ [low     ] ChunkAnalyzer: chunk structure looks healthy
    ✓ [low     ] RetrievalAuditor: top documents match query
    ✗ [high    ] HallucinationTracer: claim "4000mg" not found in source docs
  Fix: Strengthen system prompt: 'Answer ONLY from provided sources.'
══════════════════════════════════════════════════════════════
```

---

## Step 3: Your First Diagnosis (Python SDK)

```python
from rag_doctor import Doctor
from rag_doctor.connectors.base import Document

# Pass documents from your RAG pipeline (or leave empty)
docs = [
    Document(
        content="For patients with liver disease, the maximum acetaminophen dose is 2000mg per day.",
        score=0.91,
        position=0,
    ),
    Document(
        content="Standard adult dosing is 325-650mg every 4-6 hours, not exceeding 3000mg/day.",
        score=0.74,
        position=1,
    ),
]

report = Doctor.default().diagnose(
    query    = "What is the maximum acetaminophen dose for liver disease?",
    answer   = "The maximum daily dose is 4000mg.",
    docs     = docs,
    expected = "For liver disease patients the maximum dose is 2000mg per day.",
)

print(report.root_cause)      # → "hallucination"
print(report.severity)        # → "high"
print(report.fix_suggestion)  # → "Strengthen system prompt..."
print(report.to_json())       # → full JSON
```

---

## Step 4: Debug from Your Logs

If you have the query, answer, and retrieved doc IDs in your logs, you can diagnose any production incident without touching the database:

```python
from rag_doctor import Doctor
from rag_doctor.connectors.base import Document

# Reconstruct from your stored logs
docs = [
    Document(content=row["text"], score=row["score"], position=i)
    for i, row in enumerate(my_log_rows)
]

report = Doctor.default().diagnose(
    query=logged_query,
    answer=logged_answer,
    docs=docs,
)
print(report.root_cause)
```

No database connection required. rag-doctor works entirely from the data you pass.

---

## Step 5: Batch Diagnose (JSONL)

Create a `test_set.jsonl` file:

```json
{"query": "What is the refund policy?", "answer": "30 day returns.", "expected": "Full refunds within 30 days of purchase."}
{"query": "How do I cancel my subscription?", "answer": "Call support.", "expected": "Cancellations can be done online in account settings."}
```

Run:

```bash
rag-doctor batch --input test_set.jsonl --fail-on-severity high
```

---

## Next Steps

- [User Guide](user-guide.md) — real user journeys and end-to-end approach
- [Root Causes](root-causes.md) — what each root cause means and how to fix it
- [Writing Connectors](connectors.md) — connect your production vector DB
- [Configuration](configuration.md) — tune thresholds for your domain
