"""
rag-doctor quickstart example
==============================
Demonstrates diagnosing four classic RAG failure patterns
using only the offline MockConnector — no API keys needed.

Run:
    python examples/quickstart.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rag_doctor import Doctor
from rag_doctor.connectors.mock import MockConnector
from rag_doctor.connectors.base import Document


def sep(title):
    print(f"\n{'='*62}")
    print(f"  {title}")
    print('='*62)


# ── Example 1: Chunk Fragmentation (RC-3) ────────────────────────────────────
sep("Example 1: Chunk Fragmentation  (Legal Contract)")

connector1 = MockConnector(corpus=[
    {"id": "c1", "content": "Acme Corp Agreement Section 12: Termination. Either party may terminate"},
    {"id": "c2", "content": "with 90 days written notice to the other party. Notice via certified mail."},
    {"id": "c3", "content": "Standard monthly contracts: 30 days notice required for cancellation."},
    {"id": "c4", "content": "All notices must be sent to the registered office address of the party."},
])
doctor1 = Doctor.default(connector1)
docs1   = connector1.retrieve("Acme termination notice period", top_k=4)

report1 = doctor1.diagnose(
    query    = "What is the Acme Corp termination notice period?",
    answer   = "30 days notice required.",
    docs     = docs1,
    expected = "Acme Corp requires 90 days written notice.",
)
print(report1.to_text())


# ── Example 2: Context Position Bias (RC-2) ──────────────────────────────────
sep("Example 2: Context Position Bias  (Medical Dosing)")

# The correct document is at position 1 (middle of 3) — danger zone
docs2 = [
    Document(content="General analgesic overview: acetaminophen and ibuprofen are common OTC pain relievers.", position=0, score=0.74),
    Document(content="For liver disease patients maximum acetaminophen dose is 2000mg per day. Consult a physician.", position=1, score=0.91),
    Document(content="Standard adult acetaminophen dose: up to 4000mg per day. Do not exceed label dosage.", position=2, score=0.68),
]
report2 = Doctor.default().diagnose(
    query    = "What is the max acetaminophen dose for liver disease patients?",
    answer   = "The maximum daily dose is 4000mg.",
    docs     = docs2,
    expected = "For liver disease patients maximum dose is 2000mg per day.",
)
print(report2.to_text())


# ── Example 3: Hallucination (RC-4) ──────────────────────────────────────────
sep("Example 3: Hallucination  (Developer Docs)")

connector3 = MockConnector(
    corpus=[
        {"id": "api1", "content": "uploadFile(file, metadata=None) — Uploads a file. Returns UploadResponse. Does not support streaming or chunked encoding."},
        {"id": "api2", "content": "streamUpload(file) — Use for files over 100MB. Supports chunked transfer encoding."},
        {"id": "api3", "content": "deleteFile(file_id) — Permanently deletes a file. This action is irreversible."},
    ],
    inject_hallucination=True,   # Forces a hallucinated answer
)
doctor3 = Doctor.default(connector3)
docs3   = connector3.retrieve("uploadFile chunked encoding support", top_k=3)
answer3 = connector3.generate("Does uploadFile support chunked transfer encoding?", docs3)

report3 = doctor3.diagnose(
    query  = "Does uploadFile() support chunked transfer encoding?",
    answer = answer3,
    docs   = docs3,
)
print(report3.to_text())


# ── Example 4: Healthy Pipeline (RC-0) ───────────────────────────────────────
sep("Example 4: Healthy Pipeline  (E-commerce)")

connector4 = MockConnector(corpus=[
    {"id": "pol1", "content": "Return policy: Customers may return any item within 30 days of purchase for a full refund. Items must be in original condition."},
    {"id": "pol2", "content": "Shipping policy: Standard delivery takes 3-5 business days. Express shipping available."},
    {"id": "pol3", "content": "Privacy policy: We do not share personal data with third parties without explicit consent."},
])
doctor4 = Doctor.default(connector4)
docs4   = connector4.retrieve("return policy refund", top_k=3)

report4 = doctor4.diagnose(
    query    = "What is the return policy?",
    answer   = "Customers can return items within 30 days for a full refund.",
    docs     = docs4,
    expected = "Items returnable within 30 days for a full refund.",
)
print(report4.to_text())


# ── Summary ───────────────────────────────────────────────────────────────────
sep("Summary")
for label, r in [
    ("Legal (fragmentation)",   report1),
    ("Medical (position bias)", report2),
    ("Dev docs (hallucination)",report3),
    ("E-commerce (healthy)",    report4),
]:
    icon = "✓" if r.passed else "✗"
    print(f"  {icon}  {label:30}  {r.root_cause:28}  [{r.severity}]")
print()
