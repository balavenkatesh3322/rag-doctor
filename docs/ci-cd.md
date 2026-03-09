# CI/CD Integration

rag-doctor is designed to run as a quality gate in CI pipelines. No GPU, no cloud, no API keys required.

---

## GitHub Actions

### Basic Quality Gate

```yaml
# .github/workflows/rag-quality.yml
name: RAG Quality Gate
on: [pull_request, push]

jobs:
  rag-doctor:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install rag-doctor
        run: pip install rag-doctor

      - name: Run RAG quality gate
        run: |
          rag-doctor batch \
            --input      tests/golden_set.jsonl \
            --output     rag-report.json \
            --fail-on-severity high

      - name: Upload diagnosis report
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: rag-diagnosis-report
          path: rag-report.json
```

### With Semantic Embeddings

```yaml
      - name: Install rag-doctor with embeddings
        run: pip install rag-doctor sentence-transformers

      - name: Update thresholds for sentence-transformers
        run: |
          cat > rag-doctor.yaml << EOF
          diagnosis:
            recall_threshold: 0.65
            faithfulness_threshold: 0.65
            coherence_threshold: 0.55
          EOF

      - name: Run quality gate
        run: rag-doctor batch --input tests/golden_set.jsonl --config rag-doctor.yaml
```

---

## Creating Your Test Set

Your `tests/golden_set.jsonl` should contain representative queries with known-good answers:

```json
{"query": "What is the enterprise SLA uptime guarantee?", "answer": "99.9% uptime.", "expected": "Enterprise plan guarantees 99.9% uptime SLA with 24/7 support."}
{"query": "What is the cancellation policy?", "answer": "Cancel anytime.", "expected": "Month-to-month plans can be cancelled at any time. Annual plans can be cancelled with 30 days notice."}
{"query": "What payment methods are accepted?", "answer": "Credit cards and PayPal.", "expected": "We accept Visa, Mastercard, American Express, and PayPal."}
```

**Recommended test set size:** 20–50 queries covering:
- Happy path queries (should be `healthy`)
- Known edge cases from production incidents
- Queries at vocabulary boundaries (domain-specific terms)

---

## Python-Based Tests

```python
# tests/test_rag_quality.py
import pytest
from rag_doctor import Doctor
from rag_doctor.connectors.mock import MockConnector

# Load from your actual corpus
CORPUS = [
    "Enterprise plan guarantees 99.9% uptime SLA with 24/7 support.",
    "Month-to-month plans can be cancelled at any time.",
    "Annual plans can be cancelled with 30 days written notice.",
    "We accept Visa, Mastercard, American Express, and PayPal.",
    # ... more documents
]

@pytest.fixture(scope="module")
def connector():
    return MockConnector(corpus=CORPUS)


@pytest.mark.parametrize("query, expected", [
    ("enterprise SLA uptime", "99.9% uptime SLA"),
    ("how to cancel subscription", "cancelled at any time"),
    ("payment methods accepted", "Visa, Mastercard"),
])
def test_retrieval_quality(connector, query, expected):
    docs = connector.retrieve(query, top_k=5)
    report = Doctor.default().diagnose(
        query=query,
        answer=connector.generate(query, docs),
        docs=docs,
        expected=expected,
    )
    assert report.severity != "critical", (
        f"Critical failure for query '{query}': {report.finding}"
    )


def test_no_regressions():
    """Ensure no severe regressions across the full test set."""
    cases = [
        {"query": "enterprise SLA", "answer": "99.9% uptime", "expected": "99.9% uptime SLA"},
        {"query": "cancellation policy", "answer": "cancel anytime", "expected": "cancel at any time"},
    ]
    reports = Doctor.default().batch_diagnose(cases)
    critical = [r for r in reports if r.severity == "critical"]
    assert not critical, f"Critical failures: {[r.root_cause for r in critical]}"
```

---

## Detecting Regressions After Model Updates

When you update your embedding model or LLM, run the quality gate and compare:

```bash
# Before update
rag-doctor batch --input golden_set.jsonl --output before.json

# After updating your model
rag-doctor batch --input golden_set.jsonl --output after.json

# Compare (Python)
python compare_reports.py before.json after.json
```

```python
# compare_reports.py
import json, sys

before = json.load(open(sys.argv[1]))
after = json.load(open(sys.argv[2]))

regressions = []
for b, a in zip(before, after):
    sev = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    if sev[a["severity"]] > sev[b["severity"]]:
        regressions.append({
            "query": a["query"],
            "before": b["severity"],
            "after": a["severity"],
            "root_cause": a["root_cause"],
        })

if regressions:
    print(f"REGRESSIONS DETECTED: {len(regressions)}")
    for r in regressions:
        print(f"  {r['query'][:50]}: {r['before']} → {r['after']} ({r['root_cause']})")
    sys.exit(1)
else:
    print("No regressions detected.")
```
