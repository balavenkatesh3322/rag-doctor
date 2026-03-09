#!/usr/bin/env python3
"""
Sample 3: Batch CI Quality Gate
================================
Run a batch diagnosis over a golden test set and exit with code 1
if any case has severity >= high. Use this in CI pipelines.

Run:
    python samples/03_batch_ci_gate.py
    echo "Exit code: $?"
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_doctor import Doctor
from rag_doctor.connectors.mock import MockConnector


# Your document corpus (replace with your actual documents in production)
CORPUS = [
    {"content": "Enterprise plan guarantees 99.99% uptime SLA with 24/7 dedicated support."},
    {"content": "Standard plan offers 99.5% uptime with email support during business hours."},
    {"content": "Annual subscriptions include a 20% discount compared to monthly billing."},
    {"content": "Month-to-month plans can be cancelled at any time with no cancellation fee."},
    {"content": "Annual plans require 30 days written notice for cancellation."},
    {"content": "We accept Visa, Mastercard, American Express, and PayPal for payment."},
    {"content": "Custom enterprise pricing is available for accounts with 500+ seats."},
    {"content": "Data exports are available in CSV and JSON formats from account settings."},
    {"content": "Two-factor authentication is required for all enterprise accounts."},
    {"content": "The API rate limit is 1000 requests per minute for enterprise accounts."},
]

# Golden test set: queries that should all resolve with low/medium severity
GOLDEN_SET = [
    {
        "query":    "What is the enterprise SLA uptime guarantee?",
        "expected": "Enterprise plan guarantees 99.99% uptime",
    },
    {
        "query":    "How do I cancel a month-to-month subscription?",
        "expected": "cancelled at any time with no cancellation fee",
    },
    {
        "query":    "What payment methods are accepted?",
        "expected": "Visa, Mastercard, American Express, and PayPal",
    },
    {
        "query":    "What is the API rate limit for enterprise accounts?",
        "expected": "1000 requests per minute for enterprise",
    },
]

SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def run_quality_gate(fail_on_severity: str = "high") -> bool:
    """Run quality gate. Returns True if all pass, False if any failures."""
    threshold = SEVERITY_RANK[fail_on_severity]
    conn   = MockConnector(corpus=CORPUS, quiet=True)
    doctor = Doctor.default(conn)

    print(f"\n{'#':>3}  {'Query':50}  {'Root Cause':25}  Severity")
    print("-" * 95)

    failures = 0
    for i, case in enumerate(GOLDEN_SET, 1):
        docs   = conn.retrieve(case["query"], top_k=5)
        answer = conn.generate(case["query"], docs)

        report = doctor.diagnose(
            query    = case["query"],
            answer   = answer,
            docs     = docs,
            expected = case.get("expected"),
        )

        status = "✓" if SEVERITY_RANK.get(report.severity, 0) < threshold else "✗"
        print(f"{i:>3}  {case['query'][:50]:50}  {report.root_cause:25}  {report.severity.upper():8}  {status}")

        if SEVERITY_RANK.get(report.severity, 0) >= threshold:
            failures += 1

    print(f"\nTotal: {len(GOLDEN_SET)} | Failures (>={fail_on_severity}): {failures}")
    return failures == 0


def main():
    print("=" * 60)
    print("Sample 3: Batch CI Quality Gate")
    print("=" * 60)

    passed = run_quality_gate(fail_on_severity="high")

    if passed:
        print("\n✓ RAG quality gate PASSED")
        sys.exit(0)
    else:
        print("\n✗ RAG quality gate FAILED — check the diagnosis above")
        sys.exit(1)


if __name__ == "__main__":
    main()
