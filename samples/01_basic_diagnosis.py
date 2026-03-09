#!/usr/bin/env python3
"""
Sample 1: Basic Diagnosis
=========================
The simplest possible use of rag-doctor.
No database, no LLM, no configuration needed.

Run:
    python samples/01_basic_diagnosis.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_doctor import Doctor
from rag_doctor.connectors.base import Document


def main():
    print("=" * 60)
    print("Sample 1: Basic Diagnosis")
    print("=" * 60)

    # Simulate retrieved documents from your RAG pipeline.
    # In production these come from your vector database query.
    docs = [
        Document(
            content=(
                "For patients with hepatic impairment, the maximum recommended "
                "daily dose of acetaminophen is 2000 mg. Chronic heavy alcohol "
                "users should not exceed 2000 mg per day."
            ),
            score=0.91,
            position=0,
        ),
        Document(
            content=(
                "Standard adult dosing for acetaminophen is 325-650 mg every "
                "4-6 hours as needed, not exceeding 3000 mg per day in healthy adults."
            ),
            score=0.74,
            position=1,
        ),
        Document(
            content="Ibuprofen is contraindicated in patients with renal impairment.",
            score=0.32,
            position=2,
        ),
    ]

    report = Doctor.default().diagnose(
        query    = "What is the maximum acetaminophen dose for someone with liver disease?",
        answer   = "The maximum daily dose of acetaminophen is 4000mg.",
        docs     = docs,
        expected = "For hepatic impairment patients the maximum dose is 2000mg per day.",
    )

    print(f"\nRoot Cause   : {report.root_cause} ({report.root_cause_id})")
    print(f"Severity     : {report.severity.upper()}")
    print(f"Finding      : {report.finding}")
    print(f"Fix          : {report.fix_suggestion}")

    print("\nTool Results:")
    for tr in report.tool_results:
        status = "✓" if tr.passed else "✗"
        print(f"  {status} [{tr.severity:8}] {tr.tool_name}: {tr.finding}")

    print("\nConfig Patch      :", report.config_patch)
    print("Faithfulness Score:", report.faithfulness_score)
    print("Retrieval Score   :", report.retrieval_score)


if __name__ == "__main__":
    main()
