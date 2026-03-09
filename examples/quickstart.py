"""
rag-doctor quickstart example.

Demonstrates diagnosing four classic RAG failure patterns
using only the offline MockConnector — no API keys needed.

Run:
    python examples/quickstart.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rag_doctor import Doctor
from rag_doctor.connectors.mock import MockConnector


def separator(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ─────────────────────────────────────────────────────────────
# Example 1: Chunk Fragmentation (RC-3)
# ─────────────────────────────────────────────────────────────
separator("Example 1: Chunk Fragmentation (Legal Contract)")

connector = MockConnector(corpus=[
    {"id": "c1", "content": "Acme Corp Agreement Section 12: Termination. Either party may terminate"},
    {"id": "c2", "content": "with 90 days written notice to the other party. Notice via certified mail."},
    {"id": "c3", "content": "Standard contracts: 30 days notice for monthly billing."},
])
doctor = Doctor.default(connector)
docs = connector.retrieve("Acme termination notice", top_k=3)

report = doctor.diagnose(
    query    = "What is the Acme Corp termination notice period?",
    answer   = "30 days notice required.",
    docs     = docs,
    expected = "Acme Corp requires 90 days written notice.",
)
print(report.to_text())


# ─────────────────────────────────────────────────────────────
# Example 2: Position Bias (RC-2)
# ─────────────────────────────────────────────────────────────
separator("Example 2: Context Position Bias (Medical)")

from rag_doctor.connectors.base import Document

# Simulate: correct doc at middle position
docs_biased = [
    Document(content="General drug information for common pain relievers.", position=0, score=0.72),
    Document(content="For liver disease patients maximum acetaminophen dose is 2000mg per day.", position=1, score=0.88),
    Document(content="Ibuprofen dosing: 400-800mg every 4-6 hours.", position=2, score=0.68),
]
doctor2 = Doctor.default()
report2 = doctor2.diagnose(
    query    = "What is the max acetaminophen dose for liver disease patients?",
    answer   = "The maximum daily dose is 4000mg.",
    docs     = docs_biased,
    expected = "For liver disease patients maximum dose is 2000mg per day.",
)
print(report2.to_text())


# ─────────────────────────────────────────────────────────────
# Example 3: Hallucination (RC-4)
# ─────────────────────────────────────────────────────────────
separator("Example 3: Hallucination (Dev Docs)")

connector3 = MockConnector(
    corpus=[
        {"id": "api", "content": "uploadFile(file, metadata=None) — Uploads a file. Returns UploadResponse. Does not support streaming."},
    ],
    inject_hallucination=True,  # Forces a hallucinated answer
)
doctor3 = Doctor.default(connector3)
docs3 = connector3.retrieve("uploadFile chunked encoding", top_k=2)
answer3 = connector3.generate("Does uploadFile support chunked transfer encoding?", docs3)

report3 = doctor3.diagnose(
    query  = "Does uploadFile() support chunked transfer encoding?",
    answer = answer3,
    docs   = docs3,
)
print(report3.to_text())


# ─────────────────────────────────────────────────────────────
# Example 4: Healthy Pipeline (RC-0)
# ─────────────────────────────────────────────────────────────
separator("Example 4: Healthy Pipeline")

connector4 = MockConnector(corpus=[
    {"id": "p1", "content": "Return policy: Customers may return any item within 30 days of purchase for a full refund. Items must be in original condition. Refunds processed within 5 business days."},
])
doctor4 = Doctor.default(connector4)
docs4 = connector4.retrieve("return policy refund", top_k=3)

report4 = doctor4.diagnose(
    query    = "What is the return policy?",
    answer   = "Customers can return items within 30 days for a full refund.",
    docs     = docs4,
    expected = "Items returnable within 30 days for a full refund.",
)
print(report4.to_text())


# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
separator("Summary")
for label, r in [
    ("Legal (fragmentation)", report),
    ("Medical (position bias)", report2),
    ("Dev docs (hallucination)", report3),
    ("E-commerce (healthy)", report4),
]:
    status = "✓" if r.passed else "✗"
    print(f"  {status}  {label:30}  {r.root_cause:30}  [{r.severity}]")
print()
