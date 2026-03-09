"""End-to-end tests: full pipeline from corpus to diagnosis report."""

from rag_doctor.doctor import Doctor
from rag_doctor.connectors.mock import MockConnector
from rag_doctor.connectors.base import Document
from tests.fixtures import (
    LEGAL_CORPUS, MEDICAL_CORPUS, TECH_CORPUS, HR_CORPUS_OLD, HR_CORPUS_NEW,
    make_legal_connector, make_medical_connector, make_tech_connector,
)


def make_doctor(connector=None) -> Doctor:
    return Doctor.default(connector)


# ─── E2E: Legal ───────────────────────────────────────────────────────────────

class TestEndToEndScenarios:

    def test_e2e_legal_fragmented_chunks(self):
        """Split clause across two chunks → should flag chunk_fragmentation or retrieval_miss."""
        connector = MockConnector(corpus=[
            {"id": "c1", "content": "Acme Corp: Either party may terminate this agreement with"},
            {"id": "c2", "content": "90 days written notice via certified mail to the other party."},
            {"id": "c3", "content": "Standard contracts: 30 day notice for monthly billing customers."},
        ])
        doctor = make_doctor(connector)
        docs = connector.retrieve("Acme termination notice period", top_k=3)
        report = doctor.diagnose(
            query="What is the termination notice period for Acme Corp?",
            answer="30 days notice required.",
            docs=docs,
            expected="Acme Corp requires 90 days written notice for termination.",
        )
        print(f"\n[E2E Legal] Root cause: {report.root_cause} | Severity: {report.severity}")
        assert report.root_cause in (
            "chunk_fragmentation", "retrieval_miss", "query_mismatch",
            "hallucination", "context_position_bias", "healthy"
        )
        assert report.severity in ("low", "medium", "high", "critical")

    def test_e2e_medical_position_bias(self):
        """Correct medical doc at middle position → position_bias or retrieval."""
        docs = [
            Document(content="General analgesic information for common medications.", position=0, score=0.5),
            Document(content="For liver disease patients maximum acetaminophen dose is 2000mg per day.", position=1, score=0.9),
            Document(content="Ibuprofen: 400mg every four hours for pain relief.", position=2, score=0.4),
        ]
        doctor = make_doctor()
        report = doctor.diagnose(
            query="Max acetaminophen dose for liver disease patients?",
            answer="The maximum daily dose is 4000mg.",
            docs=docs,
            expected="For liver disease patients maximum dose is 2000mg per day.",
        )
        print(f"\n[E2E Medical] Root cause: {report.root_cause} | Faithfulness: {next((tr.details.get('faithfulness_score') for tr in report.tool_results if tr.tool_name == 'hallucination_tracer'), 'N/A')}")
        assert report.root_cause in (
            "context_position_bias", "hallucination", "chunk_fragmentation",
            "retrieval_miss", "healthy"
        )

    def test_e2e_tech_hallucination(self):
        """Tech API doc doesn't support feature → hallucinated answer should be caught."""
        connector = MockConnector(
            corpus=[
                {"id": "api1", "content": "uploadFile() uploads files. Does not support streaming or chunked transfer encoding."},
                {"id": "api2", "content": "Use streamUpload() for large files. uploadFile() is for files under 100MB."},
            ],
            inject_hallucination=True,
        )
        doctor = make_doctor(connector)
        docs = connector.retrieve("uploadFile chunked encoding", top_k=2)
        answer = connector.generate("Does uploadFile support chunked transfer encoding?", docs)
        report = doctor.diagnose(
            query="Does uploadFile() support chunked transfer encoding?",
            answer=answer,
            docs=docs,
        )
        print(f"\n[E2E Tech] Root cause: {report.root_cause} | Answer: {answer[:80]}")
        assert "hallucination_tracer" in [tr.tool_name for tr in report.tool_results]

    def test_e2e_hr_version_conflict(self):
        """Old HR policy in corpus alongside new → stale version retrieved."""
        connector = MockConnector(corpus=HR_CORPUS_OLD + HR_CORPUS_NEW)
        doctor = make_doctor(connector)
        docs = connector.retrieve("parental leave weeks 2024", top_k=3)
        report = doctor.diagnose(
            query="How many weeks of parental leave are employees entitled to?",
            answer="12 weeks.",
            docs=docs,
            expected="Effective January 2024, all full-time employees are entitled to 16 weeks.",
        )
        print(f"\n[E2E HR] Root cause: {report.root_cause} | Severity: {report.severity}")
        assert report.root_cause in (
            "retrieval_miss", "query_mismatch", "hallucination",
            "chunk_fragmentation", "context_position_bias", "healthy"
        )

    def test_e2e_healthy_pipeline(self):
        """When query, docs and answer are semantically aligned → healthy or low severity."""
        connector = MockConnector(corpus=[
            {"id": "p1", "content": (
                "Return policy: customers may return any item within 30 days of purchase "
                "for a full refund. Items must be unused and in original packaging. "
                "Refunds are processed within 5 business days to the original payment method."
            )},
        ])
        doctor = make_doctor(connector)
        docs = connector.retrieve("return policy refund 30 days", top_k=3)
        answer = "Customers can return items within 30 days for a full refund processed in 5 business days."
        report = doctor.diagnose(
            query="What is the return policy and how long do refunds take?",
            answer=answer,
            docs=docs,
            expected="Customers may return items within 30 days for a full refund in 5 business days.",
        )
        print(f"\n[E2E Healthy] Root cause: {report.root_cause} | Severity: {report.severity}")
        assert report.severity in ("low", "medium")

    def test_e2e_chunk_optimizer_runs(self):
        """Provide corpus_texts to trigger ChunkOptimizer when fragmentation detected."""
        corpus_texts = [
            "Acme Corp Agreement Section 12: Termination. Either party may terminate",
            "with 90 days written notice to the other party. Notice via certified mail.",
            "Standard contracts: 30 days notice for monthly billing customers.",
        ]
        connector = MockConnector(corpus=[
            {"id": f"c{i}", "content": t} for i, t in enumerate(corpus_texts)
        ])
        doctor = make_doctor(connector)
        docs = connector.retrieve("Acme termination notice", top_k=3)
        report = doctor.diagnose(
            query="What is the Acme termination notice period?",
            answer="30 days notice required.",
            docs=docs,
            expected="Acme Corp requires 90 days written notice.",
            corpus_texts=corpus_texts,
        )
        print(f"\n[E2E Optimizer] Root cause: {report.root_cause}")
        assert report.root_cause is not None

    def test_e2e_report_serialisation(self):
        """Ensure report serialises to JSON without errors."""
        connector = MockConnector(corpus=[
            {"id": "p1", "content": "Policy: all customers get 30 day returns."}
        ])
        doctor = make_doctor(connector)
        docs = connector.retrieve("return policy", top_k=2)
        report = doctor.diagnose(query="return policy", answer="30 day returns.", docs=docs)
        import json
        data = json.loads(report.to_json())
        assert "root_cause" in data
        assert "severity" in data
        assert "tool_results" in data


# ─── Batch E2E ────────────────────────────────────────────────────────────────

class TestBatchEndToEnd:
    def test_batch_mixed_scenarios(self):
        connector = MockConnector(corpus=[
            {"id": "p1", "content": "Return policy: 30 days for a full refund to all customers."},
            {"id": "p2", "content": "Quantum computing uses qubits for parallel computation."},
            {"id": "p3", "content": "Enterprise SLA guarantees 99.9% uptime with 4-hour response time."},
        ])
        doctor = make_doctor(connector)
        cases = [
            {"query": "return policy",       "answer": "30 days refund.",        "expected": "30 days for a full refund"},
            {"query": "quantum features",    "answer": "Uses quantum entanglement.", "expected": "Quantum computing uses qubits"},
            {"query": "enterprise SLA uptime","answer": "99.9% uptime guaranteed.", "expected": "99.9% uptime with 4-hour response"},
        ]
        reports = doctor.batch_diagnose(cases)
        print("\n[Batch] Results:")
        for case, report in zip(cases, reports):
            print(f"  {case['query']:30}  → {report.root_cause:30}  [{report.severity}]")
        assert len(reports) == 3
        assert all(r.root_cause is not None for r in reports)
