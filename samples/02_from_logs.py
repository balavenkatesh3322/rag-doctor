#!/usr/bin/env python3
"""
Sample 2: Debug From Production Logs
=====================================
Reconstruct a diagnosis entirely from stored log data.
No database connection required.

Run:
    python samples/02_from_logs.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_doctor import Doctor
from rag_doctor.connectors.base import Document


# Simulate what you'd have in your production logs
MOCK_LOG = {
    "request_id": "req_abc123",
    "timestamp":  "2025-03-01T14:32:11Z",
    "query":      "What is the enterprise plan SLA uptime guarantee?",
    "answer":     "Our enterprise plan comes with a 99% uptime SLA.",
    "retrieved_docs": [
        {"text": "Standard plans offer 99.5% uptime.",                                          "score": 0.62, "doc_id": "doc_001"},
        {"text": "The Enterprise plan includes 24/7 support and a 99.99% uptime guarantee.",    "score": 0.58, "doc_id": "doc_002"},
        {"text": "Uptime is measured as a monthly rolling average excluding maintenance.",       "score": 0.45, "doc_id": "doc_003"},
    ],
}


def diagnose_from_log(log: dict):
    """Reconstruct and diagnose from a stored log entry."""
    docs = [
        Document(
            content = row["text"],
            score   = row["score"],
            position = i,
            doc_id  = row.get("doc_id"),
        )
        for i, row in enumerate(log["retrieved_docs"])
    ]

    return Doctor.default().diagnose(
        query    = log["query"],
        answer   = log["answer"],
        docs     = docs,
        expected = "Enterprise plan includes 99.99% uptime guarantee.",
    )


def main():
    print("=" * 60)
    print("Sample 2: Debug From Production Logs")
    print("=" * 60)
    print(f"\nRequest ID : {MOCK_LOG['request_id']}")
    print(f"Query      : {MOCK_LOG['query']}")
    print(f"Answer     : {MOCK_LOG['answer']}")

    report = diagnose_from_log(MOCK_LOG)

    print("\nDiagnosis:")
    print(f"  Root Cause : {report.root_cause} ({report.root_cause_id})")
    print(f"  Severity   : {report.severity.upper()}")
    print(f"  Finding    : {report.finding}")
    print(f"  Fix        : {report.fix_suggestion}")

    print("\n  Note: doc_002 has the correct 99.99% answer but sits at position 1")
    print("  with a LOWER score (0.58) than doc_001 (0.62). Classic position bias.")


if __name__ == "__main__":
    main()
