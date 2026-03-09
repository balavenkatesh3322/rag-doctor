#!/usr/bin/env python3
"""
Sample 6: JSON Reports and Automation
=======================================
Export diagnosis results as structured JSON for logging,
alerting, or integration with monitoring systems (Datadog, PagerDuty, etc.).

Run:
    python samples/06_json_report.py
    python samples/06_json_report.py | jq '.[].root_cause'
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import datetime
from rag_doctor import Doctor
from rag_doctor.connectors.base import Document
from rag_doctor.connectors.mock import MockConnector


CORPUS = [
    {"content": "Enterprise plan guarantees 99.99% uptime SLA with 24/7 dedicated support."},
    {"content": "All customer data is encrypted at rest using AES-256."},
    {"content": "GDPR compliance is maintained through data processing agreements."},
    {"content": "Users can export all their data at any time from account settings."},
    {"content": "Data is retained for 7 years after account closure per financial regulations."},
]


def diagnose_and_log(doctor, query: str, answer: str, docs, expected: str = None) -> dict:
    """Run diagnosis and return a structured log-ready dict."""
    report = doctor.diagnose(query=query, answer=answer, docs=docs, expected=expected)

    return {
        "timestamp":       datetime.datetime.utcnow().isoformat() + "Z",
        "query":           query,
        "root_cause":      report.root_cause,
        "root_cause_id":   report.root_cause_id,
        "severity":        report.severity,
        "passed":          report.passed,
        "finding":         report.finding,
        "fix_suggestion":  report.fix_suggestion,
        "config_patch":    report.config_patch,
        "faithfulness_score": report.faithfulness_score,
        "retrieval_score":    report.retrieval_score,
        "tool_results": [
            {
                "tool":    tr.tool_name,
                "passed":  tr.passed,
                "severity": tr.severity,
                "finding": tr.finding,
            }
            for tr in report.tool_results
        ],
    }


def main():
    print("=" * 60)
    print("Sample 6: JSON Reports")
    print("=" * 60)

    conn   = MockConnector(corpus=CORPUS, quiet=True)
    doctor = Doctor.default(conn)

    test_cases = [
        {"query": "What is the enterprise uptime SLA?",      "expected": "99.99% uptime SLA with 24/7 support"},
        {"query": "How is customer data protected?",          "expected": "encrypted at rest using AES-256"},
        {"query": "What are the GDPR compliance measures?",   "expected": "GDPR compliance through data processing agreements"},
    ]

    results = []
    for case in test_cases:
        docs   = conn.retrieve(case["query"], top_k=5)
        answer = conn.generate(case["query"], docs)
        entry  = diagnose_and_log(doctor, case["query"], answer, docs, case.get("expected"))
        results.append(entry)

    # Pretty-print the JSON report
    print("\nJSON Report:")
    print(json.dumps(results, indent=2))

    # Operational summary
    failures = [r for r in results if r["severity"] in ("high", "critical")]
    print(f"\nSummary: {len(results)} cases | Failures: {len(failures)} | Passed: {len(results)-len(failures)}")

    # In production you'd ship this to your logging system:
    # import logging
    # for entry in results:
    #     if entry["severity"] in ("high", "critical"):
    #         logging.warning("RAG quality issue: %s", json.dumps(entry))


if __name__ == "__main__":
    main()
