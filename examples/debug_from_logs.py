"""
Example: Debugging a production incident from logs.

Scenario: a user reported that the chatbot told them the wrong
acetaminophen dose for liver disease patients. You have the query,
the generated answer, and the retrieved documents saved in your logs.

This example shows how to reproduce and diagnose the incident
WITHOUT re-querying your production database.

Run:
    python examples/debug_from_logs.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rag_doctor import Doctor
from rag_doctor.connectors.base import Document


# ── Simulate production log data ──────────────────────────────────────────────
# In real use, these would come from your log store (Datadog, Elasticsearch, etc.)

LOGGED_QUERY  = "What is the maximum acetaminophen dose for a patient with liver disease?"
LOGGED_ANSWER = "The maximum daily dose of acetaminophen is 4000mg."

# The documents retrieved by your production vector DB, saved to logs
LOGGED_DOCS = [
    {
        "text":     "General drug overview: Acetaminophen is an analgesic and antipyretic.",
        "score":    0.72,
        "metadata": {"source": "drug_overview.pdf", "page": 1},
    },
    {
        "text":     "For patients with liver disease or hepatic impairment, maximum daily dose is 2000mg. Consult physician before use.",
        "score":    0.91,
        "metadata": {"source": "clinical_guidelines.pdf", "page": 14},
    },
    {
        "text":     "Standard adult dose: up to 4000mg per day. Do not exceed recommended dose.",
        "score":    0.68,
        "metadata": {"source": "drug_label.pdf", "page": 2},
    },
]

# ── Reconstruct Document objects (no DB connection needed) ────────────────────
docs = [
    Document(
        content  = row["text"],
        score    = row["score"],
        position = i,
        metadata = row["metadata"],
    )
    for i, row in enumerate(LOGGED_DOCS)
]

print("=" * 62)
print("  Reproducing production incident from logs")
print("=" * 62)
print(f"  Query  : {LOGGED_QUERY}")
print(f"  Answer : {LOGGED_ANSWER}")
print(f"  Docs   : {len(docs)} retrieved")
print()

# ── Diagnose ──────────────────────────────────────────────────────────────────
report = Doctor.default().diagnose(
    query    = LOGGED_QUERY,
    answer   = LOGGED_ANSWER,
    docs     = docs,
    expected = "For liver disease patients maximum dose is 2000mg per day.",
)

print(report.to_text())

# ── Programmatic access ───────────────────────────────────────────────────────
print("\n--- Programmatic Access ---")
print(f"root_cause    : {report.root_cause}")
print(f"severity      : {report.severity}")
print(f"fix_suggestion: {report.fix_suggestion}")
print(f"config_patch  : {report.config_patch}")

# For PagerDuty / Slack alerting:
if not report.passed:
    print(f"\n[ALERT] RAG failure detected: {report.root_cause} ({report.severity.upper()})")
    print(f"        Fix: {report.fix_suggestion}")
