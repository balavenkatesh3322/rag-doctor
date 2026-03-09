# CLI Reference

```bash
pip install rag-doctor
rag-doctor --help
```

---

## `rag-doctor diagnose`

Diagnose a single query.

```bash
rag-doctor diagnose \
  --query    "What is the termination notice period?" \
  --answer   "30 days notice is required." \
  --expected "Enterprise contracts require 90 days written notice."  # optional
  --config   rag-doctor.yaml  # optional
  --output   text             # text | json
```

### Options

| Flag | Short | Default | Description |
|---|---|---|---|
| `--query` | `-q` | required | The user query |
| `--answer` | `-a` | required | The generated answer |
| `--expected` | `-e` | none | Ground-truth answer (improves diagnosis) |
| `--config` | `-c` | none | Path to `rag-doctor.yaml` |
| `--output` | `-o` | `text` | Output format: `text` or `json` |

### Exit Codes

| Code | Meaning |
|---|---|
| `0` | Diagnosis passed (healthy) |
| `1` | Issues found |

### JSON Output

```bash
rag-doctor diagnose -q "..." -a "..." --output json
```

```json
{
  "root_cause": "hallucination",
  "root_cause_id": "RC-4",
  "severity": "high",
  "passed": false,
  "finding": "2/3 answer claims not grounded in retrieved docs.",
  "fix_suggestion": "Strengthen system prompt: Answer ONLY from provided sources.",
  "config_patch": {"llm.temperature": 0.0},
  "faithfulness_score": 0.33,
  "retrieval_score": 0.71,
  "tool_results": [...]
}
```

---

## `rag-doctor batch`

Batch diagnose from a JSONL file.

```bash
rag-doctor batch \
  --input              test_set.jsonl \
  --output             report.json \
  --fail-on-severity   high \
  --config             rag-doctor.yaml
```

### Input Format

One JSON object per line:

```json
{"query": "What is the refund policy?", "answer": "30 day returns.", "expected": "Full refunds within 30 days."}
{"query": "How do I cancel?", "answer": "Call support.", "expected": "Cancel online in account settings."}
{"query": "What are the shipping times?", "answer": "5-7 business days."}
```

### Options

| Flag | Short | Default | Description |
|---|---|---|---|
| `--input` | `-i` | required | Path to JSONL file |
| `--output` | `-o` | none | Save results to JSON file |
| `--config` | `-c` | none | Path to `rag-doctor.yaml` |
| `--fail-on-severity` | — | `high` | Exit with code 1 if any result has this severity or worse |

### Output

```
  #  Query                                     Root Cause                 Severity
──────────────────────────────────────────────────────────────────────────────────────
  1  What is the refund policy?                hallucination              high    ✗
  2  How do I cancel?                          query_mismatch             medium  ✓
  3  What are the shipping times?              healthy                    low     ✓

Total: 3 | Failures: 1
```

### Exit Codes

| Code | Meaning |
|---|---|
| `0` | All results below `--fail-on-severity` threshold |
| `1` | One or more results at or above `--fail-on-severity` |

---

## Using in CI (GitHub Actions)

```yaml
- name: RAG Quality Gate
  run: |
    rag-doctor batch \
      --input test_set.jsonl \
      --output diagnosis-report.json \
      --fail-on-severity high
  
- name: Upload report
  uses: actions/upload-artifact@v4
  if: always()
  with:
    name: rag-diagnosis-report
    path: diagnosis-report.json
```

The step fails (`exit 1`) if any case has `severity: high` or `severity: critical`, blocking the deployment.

---

## Programmatic Use of CLI Exit Codes

```bash
#!/bin/bash
rag-doctor batch --input test_set.jsonl --fail-on-severity high
if [ $? -ne 0 ]; then
    echo "RAG quality gate failed. Check report.json."
    exit 1
fi
echo "RAG quality gate passed."
```
