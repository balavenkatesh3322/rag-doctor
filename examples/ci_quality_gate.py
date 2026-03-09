"""
Example: CI Quality Gate

Drop this into your test suite to automatically catch RAG regressions.
Designed to run in CI with zero external dependencies.

As a standalone script (no pytest needed):
    python examples/ci_quality_gate.py

As a pytest test (if pytest is installed):
    pytest examples/ci_quality_gate.py -v

In your CI workflow:
    - name: RAG quality gate
      run: python examples/ci_quality_gate.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_doctor import Doctor
from rag_doctor.connectors.mock import MockConnector
from rag_doctor.config import RagDoctorConfig


# ── Your corpus (replace with your actual documents) ─────────────────────────
MEDICAL_CORPUS = [
    {"id": "drug_001", "content": "Acetaminophen dosing: standard adult max 4000mg/day. Liver disease patients: max 2000mg/day. Consult physician."},
    {"id": "drug_002", "content": "Ibuprofen: 400-800mg every 4-6 hours. Max 3200mg/day for OTC use."},
    {"id": "drug_003", "content": "Aspirin: not recommended for children under 12 due to Reye's syndrome risk."},
]

LEGAL_CORPUS = [
    {"id": "contract_001", "content": "Acme Corp MSA: termination requires 90 days written notice via certified mail."},
    {"id": "contract_002", "content": "Standard monthly contracts: 30 days notice for termination."},
    {"id": "policy_001",   "content": "Enterprise refund policy: full refund within 90 days of purchase."},
    {"id": "policy_002",   "content": "Standard refund policy: 30 days return window. Items must be in original condition."},
]

HR_CORPUS = [
    {"id": "leave_2024", "content": "Parental leave (effective Jan 2024): all full-time employees entitled to 16 weeks paid parental leave."},
    {"id": "leave_2022", "content": "Parental leave (2022 policy): 12 weeks paid parental leave for full-time employees."},
]

# ── Golden test pairs: (query, expected_answer, corpus) ─────────────────────
GOLDEN_TESTS = [
    (
        "Max acetaminophen dose for liver disease patients?",
        "For liver disease patients maximum dose is 2000mg per day.",
        MEDICAL_CORPUS,
    ),
    (
        "What is the Acme Corp termination notice period?",
        "Acme Corp requires 90 days written notice.",
        LEGAL_CORPUS,
    ),
    (
        "How many weeks parental leave as of 2024?",
        "Effective January 2024 employees get 16 weeks paid parental leave.",
        HR_CORPUS,
    ),
]

SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def run_quality_gate() -> int:
    """Run all quality checks. Returns 0 on pass, 1 on failure."""
    doctor   = Doctor.default()
    failures = []

    print(f"\n{'Status':<8} {'Severity':<10} {'Root Cause':<30} Query")
    print("-" * 90)

    for query, expected, corpus in GOLDEN_TESTS:
        connector = MockConnector(corpus=corpus, quiet=True)
        docs      = connector.retrieve(query, top_k=5)
        answer    = connector.generate(query, docs)

        report = doctor.diagnose(
            query=query, answer=answer, docs=docs, expected=expected
        )

        ok     = report.severity in ("low", "medium")
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"{status:<8} {report.severity.upper():<10} {report.root_cause:<30} {query[:45]}")

        if not ok:
            failures.append(report)

    print(f"\nTotal: {len(GOLDEN_TESTS)} | Failed: {len(failures)} | Passed: {len(GOLDEN_TESTS)-len(failures)}")

    if failures:
        print("\nFailed tests detail:")
        for r in failures:
            print(f"  • {r.query[:55]}")
            print(f"    Root Cause : {r.root_cause} ({r.severity.upper()})")
            print(f"    Fix        : {r.fix_suggestion}")
    return 1 if failures else 0


# ── Optional pytest-style tests (only registered when pytest imports this) ──
try:
    import pytest  # noqa: F401

    @pytest.fixture(scope="module")
    def doctor_fixture():
        return Doctor.default()

    @pytest.mark.parametrize("query,expected,corpus", GOLDEN_TESTS)
    def test_rag_quality(doctor_fixture, query, expected, corpus):
        connector = MockConnector(corpus=corpus, quiet=True)
        docs      = connector.retrieve(query, top_k=5)
        answer    = connector.generate(query, docs)
        report    = doctor_fixture.diagnose(
            query=query, answer=answer, docs=docs, expected=expected
        )
        assert report.severity in ("low", "medium"), (
            f"\nRAG quality gate FAILED\n"
            f"Root Cause : {report.root_cause} ({report.severity.upper()})\n"
            f"Finding    : {report.finding}\n"
            f"Fix        : {report.fix_suggestion}"
        )

except ImportError:
    pass  # pytest not installed — standalone mode only


if __name__ == "__main__":
    print("\n🩺  RAG Quality Gate\n" + "=" * 50)
    sys.exit(run_quality_gate())
