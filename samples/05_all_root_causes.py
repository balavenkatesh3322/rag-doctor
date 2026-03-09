#!/usr/bin/env python3
"""
Sample 5: Trigger All Root Causes
==================================
Demonstrates each of the 6 root causes with a concrete example.
Useful for understanding what each root cause looks and feels like.

Run:
    python samples/05_all_root_causes.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_doctor import Doctor
from rag_doctor.connectors.base import Document
from rag_doctor.connectors.mock import MockConnector


CORPUS = [
    "The enterprise plan includes 99.99% uptime SLA and 24/7 dedicated support.",
    "Standard plans offer 99.5% uptime with business hours support only.",
    "Annual subscriptions include a 20% discount compared to monthly pricing.",
    "Cancellation requires 30 days written notice for annual plans.",
    "Month-to-month plans can be cancelled at any time without penalty.",
    "All payments are processed securely via Stripe and stored encrypted.",
    "Two-factor authentication is mandatory for all admin accounts.",
    "The API supports REST and GraphQL with rate limits of 1000 req/min.",
]

RC_COLOR = {
    "healthy":               "\033[32m",
    "retrieval_miss":        "\033[91m",
    "context_position_bias": "\033[93m",
    "chunk_fragmentation":   "\033[93m",
    "hallucination":         "\033[31m",
    "query_mismatch":        "\033[95m",
}
SEV_COLOR = {"low": "\033[32m", "medium": "\033[33m", "high": "\033[91m", "critical": "\033[31m"}
RESET = "\033[0m"


def show(title: str, report):
    c = RC_COLOR.get(report.root_cause, "")
    s = SEV_COLOR.get(report.severity, "")
    print(f"\n  {title}")
    print(f"  {'─' * 56}")
    print(f"  Root Cause : {c}{report.root_cause} ({report.root_cause_id}){RESET}")
    print(f"  Severity   : {s}{report.severity.upper()}{RESET}")
    print(f"  Finding    : {report.finding[:90]}")
    if report.fix_suggestion:
        print(f"  Fix        : {report.fix_suggestion[:90]}")


def main():
    print("=" * 62)
    print("  Sample 5: All Root Causes")
    print("=" * 62)

    doctor = Doctor.default()
    conn   = MockConnector(corpus=[{"content": c} for c in CORPUS], quiet=True)

    # ── RC-0: Healthy ──────────────────────────────────────────────────────────
    docs   = conn.retrieve("What is the enterprise SLA uptime guarantee?", top_k=5)
    answer = conn.generate("What is the enterprise SLA uptime guarantee?", docs)
    report = doctor.diagnose(
        query    = "What is the enterprise SLA uptime guarantee?",
        answer   = answer,
        docs     = docs,
        expected = "Enterprise plan includes 99.99% uptime SLA",
    )
    show("RC-0: healthy — pipeline working correctly", report)

    # ── RC-1: Retrieval Miss ───────────────────────────────────────────────────
    # Ask about something NOT in the corpus at all
    docs = conn.retrieve("quantum computing error correction algorithm", top_k=5)
    report = doctor.diagnose(
        query    = "What is the quantum computing error correction algorithm?",
        answer   = "The algorithm uses surface codes for error correction.",
        docs     = docs,
        expected = "surface codes quantum error correction",
    )
    show("RC-1: retrieval_miss — correct doc not in top-k", report)

    # ── RC-2: Context Position Bias ────────────────────────────────────────────
    # Best doc is manually placed in the MIDDLE position (pos 2 of 5)
    docs = [
        Document(content="Standard plans offer 99.5% uptime.",                         score=0.80, position=0),
        Document(content="Month-to-month can be cancelled anytime.",                    score=0.72, position=1),
        Document(content="Enterprise plan includes 99.99% uptime SLA.",                 score=0.91, position=2),  # correct but middle
        Document(content="Two-factor auth is mandatory for admin accounts.",             score=0.45, position=3),
        Document(content="API rate limit is 1000 req/min.",                             score=0.31, position=4),
    ]
    report = doctor.diagnose(
        query    = "What uptime guarantee does the enterprise plan have?",
        answer   = "The enterprise plan offers 99.5% uptime.",   # wrong — took pos 0
        docs     = docs,
        expected = "Enterprise plan includes 99.99% uptime SLA.",
    )
    show("RC-2: context_position_bias — best doc buried in middle", report)

    # ── RC-3: Chunk Fragmentation ──────────────────────────────────────────────
    # Simulate truncated mid-sentence chunks
    frag_docs = [
        Document(content="and the enterprise plan also includes dedicated account",                  score=0.85, position=0),
        Document(content="management and priority escalation for critical issues as well as",        score=0.72, position=1),
        Document(content="24/7 phone support which is not available on",                             score=0.61, position=2),
    ]
    report = doctor.diagnose(
        query  = "What support is included in the enterprise plan?",
        answer = "The enterprise plan includes some support options.",
        docs   = frag_docs,
    )
    show("RC-3: chunk_fragmentation — chunks start/end mid-sentence", report)

    # ── RC-4: Hallucination ────────────────────────────────────────────────────
    conn_hal = MockConnector(corpus=[{"content": c} for c in CORPUS], inject_hallucination=True, quiet=True)
    docs   = conn_hal.retrieve("What is the cancellation policy?", top_k=5)
    answer = conn_hal.generate("What is the cancellation policy?", docs)
    report = doctor.diagnose(
        query  = "What is the cancellation policy?",
        answer = answer,
        docs   = docs,
    )
    show("RC-4: hallucination — answer not grounded in retrieved docs", report)

    # ── RC-5: Query Mismatch ───────────────────────────────────────────────────
    # Query uses "sacking" — corpus uses "cancellation"
    docs = conn.retrieve("employee sacking procedure", top_k=5)
    report = doctor.diagnose(
        query    = "What is the employee sacking procedure?",
        answer   = "No specific procedure was found.",
        docs     = docs,
        expected = "cancellation requires 30 days written notice",
    )
    show("RC-5: query_mismatch — vocabulary mismatch (sacking vs cancellation)", report)

    print("\n" + "=" * 62)
    print("  All 6 root causes demonstrated.")
    print("=" * 62)


if __name__ == "__main__":
    main()
