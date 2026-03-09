"""
Example: Batch Diagnosis

Useful for:
  - Post-incident retrospectives (diagnose a week of bad answers at once)
  - Nightly regression testing against a golden dataset
  - Comparing before/after a deployment

Run:
    python examples/batch_diagnose.py
    python examples/batch_diagnose.py --jsonl examples/batch_example.jsonl

Or with the CLI:
    rag-doctor batch --input examples/batch_example.jsonl --fail-on-severity high
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
from rag_doctor import Doctor
from rag_doctor.connectors.mock import MockConnector


# ── Shared corpus for the built-in test cases ─────────────────────────────────
CORPUS = [
    # Legal
    {"id": "l1", "content": "Acme Corp Agreement: Either party may terminate with 90 days written notice via certified mail."},
    {"id": "l2", "content": "Standard contracts: 30 days notice for month-to-month billing."},
    # Medical
    {"id": "m1", "content": "Acetaminophen: standard adult dose up to 4000mg/day. Liver disease patients: maximum 2000mg/day."},
    {"id": "m2", "content": "Ibuprofen: 400-800mg every 4-6 hours. Maximum 3200mg/day."},
    # HR
    {"id": "h1", "content": "Parental leave (2024): all full-time employees are entitled to 16 weeks paid parental leave."},
    {"id": "h2", "content": "Parental leave (2022): 12 weeks paid parental leave."},
    # Tech
    {"id": "t1", "content": "uploadFile(): uploads file, returns UploadResponse. Does NOT support streaming or chunked encoding."},
    {"id": "t2", "content": "streamUpload(): use for files over 100MB. Supports chunked transfer encoding."},
    # Policy
    {"id": "p1", "content": "Return policy: full refund within 30 days of purchase. Items must be unused and in original packaging."},
]

TEST_CASES = [
    {"query": "What is the Acme Corp termination notice period?",      "answer": "30 days notice required.",          "expected": "Acme Corp requires 90 days written notice."},
    {"query": "Max acetaminophen dose for liver disease?",              "answer": "The maximum daily dose is 4000mg.", "expected": "For liver disease patients maximum dose is 2000mg per day."},
    {"query": "How many weeks of parental leave do employees get?",     "answer": "12 weeks paid parental leave.",    "expected": "As of 2024, employees get 16 weeks paid parental leave."},
    {"query": "Does uploadFile support chunked transfer encoding?",     "answer": "Yes, use the --chunked flag.",      "expected": "No, uploadFile does not support chunked encoding."},
    {"query": "What is the return policy?",                             "answer": "Full refund within 30 days.",       "expected": "Full refund within 30 days. Items must be unused."},
]

SEVERITY_COLOR = {"low": "\033[32m", "medium": "\033[33m", "high": "\033[91m", "critical": "\033[31m"}
RESET = "\033[0m"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_batch(cases, connector):
    doctor  = Doctor.default(connector)
    reports = doctor.batch_diagnose(cases)

    print(f"\n{'#':>3}  {'Query':42}  {'Root Cause':26}  Severity  Status")
    print("-" * 96)

    failures = 0
    for i, (case, report) in enumerate(zip(cases, reports), 1):
        c  = SEVERITY_COLOR.get(report.severity, "")
        st = "✓" if report.passed else "✗"
        q  = case["query"][:42]
        print(f"{i:>3}  {q:42}  {report.root_cause:26}  {c}{report.severity:<8}{RESET}  {st}")
        if not report.passed:
            failures += 1

    print(f"\nTotal: {len(reports)} cases | Issues: {failures} | Passed: {len(reports)-failures}")
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default=None, help="Path to JSONL file")
    args = parser.parse_args()

    connector = MockConnector(corpus=CORPUS, quiet=True)
    cases     = load_jsonl(args.jsonl) if args.jsonl else TEST_CASES

    label = args.jsonl or "built-in test cases"
    print(f"\n🩺  Batch Diagnosis — {label}")
    print("=" * 70)

    failures = run_batch(cases, connector)
    # Exit 1 when issues found (correct for CI), but also informative standalone
    sys.exit(1 if failures > 0 else 0)
