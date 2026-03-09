# rag-doctor Samples

Six working examples you can run immediately.
**No API keys. No database. No configuration.**

---

## Run All Samples

```bash
git clone https://github.com/your-org/rag-doctor
cd rag-doctor

# No pip install needed — samples add the parent dir to sys.path automatically
python samples/01_basic_diagnosis.py
python samples/02_from_logs.py
python samples/03_batch_ci_gate.py
python samples/04_custom_connector.py
python samples/05_all_root_causes.py
python samples/06_json_report.py

# Or run everything at once (Mac):
chmod +x scripts/test_local_mac.sh
./scripts/test_local_mac.sh
```

---

## Sample Index

| File | What It Demonstrates | Exit Code |
|---|---|---|
| `01_basic_diagnosis.py` | Simplest possible diagnosis with `Document` objects | 0 |
| `02_from_logs.py` | Reproduce a production incident from stored log data | 0 |
| `03_batch_ci_gate.py` | CI quality gate — exits 1 if any case has high/critical severity | 0 or 1 |
| `04_custom_connector.py` | Build your own `PipelineConnector` | 0 |
| `05_all_root_causes.py` | Trigger and inspect all 6 root causes (RC-0 through RC-5) | 0 |
| `06_json_report.py` | Export `DiagnosisReport` as JSON for monitoring/alerting | 0 |

---

## Expected Output

### Sample 01 — Basic Diagnosis

```
============================================================
Sample 1: Basic Diagnosis
============================================================

Root Cause   : hallucination (RC-4)
Severity     : HIGH
Finding      : 1/2 answer claims not grounded in retrieved docs.
Fix          : Strengthen system prompt: 'Answer ONLY from provided sources.'

Tool Results:
  ✓ [low     ] ChunkAnalyzer: chunk structure looks healthy
  ✓ [low     ] RetrievalAuditor: best match score 0.91 passes threshold 0.35
  ✗ [high    ] HallucinationTracer: 1/2 claims not grounded (score 0.50)
```

### Sample 05 — All Root Causes

```
  RC-0: healthy — pipeline working correctly
  ────────────────────────────────────────────────────────
  Root Cause : healthy (RC-0)
  Severity   : LOW

  RC-1: retrieval_miss — correct doc not in top-k
  ────────────────────────────────────────────────────────
  Root Cause : retrieval_miss (RC-1)
  Severity   : HIGH
  ...
```

---

## Using Samples as Tests

Sample 03 doubles as a pytest test when pytest is installed:

```bash
pip install pytest
pytest samples/03_batch_ci_gate.py -v
```

Or run standalone as a CI gate:
```bash
python samples/03_batch_ci_gate.py
echo "Exit: $?"   # 0 = all passed, 1 = failures detected
```

---

## Adapting Samples for Your Stack

1. **Replace `CORPUS`** in samples 03, 04, 05, 06 with your actual documents
2. **Replace `MockConnector`** with your `PipelineConnector` (see `samples/04_custom_connector.py`)
3. **Add your golden test pairs** to sample 03 for regression testing
4. **Wire sample 06's JSON output** to your log aggregator (Datadog, Splunk, CloudWatch)
