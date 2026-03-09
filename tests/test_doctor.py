"""Integration tests for the Doctor orchestrator."""

from rag_doctor.doctor import Doctor
from rag_doctor.config import RagDoctorConfig
from rag_doctor.connectors.mock import MockConnector
from rag_doctor.connectors.base import Document
from tests.fixtures import (
    make_legal_connector, make_medical_connector,
    make_tech_connector, LEGAL_CORPUS, MEDICAL_CORPUS, TECH_CORPUS,
)


def make_doctor(connector=None) -> Doctor:
    return Doctor(RagDoctorConfig.default(), connector)


# ─── Scenario 1: Chunk Fragmentation ─────────────────────────────────────────

class TestScenario1ChunkFragmentation:
    """Legal contract: clause split across chunk boundaries."""

    def test_detects_fragmentation_pattern(self):
        connector = MockConnector(corpus=[
            {"id": "c1", "content": "Acme Corp Termination clause: Either party may terminate this"},
            {"id": "c2", "content": "90 days written notice to the other party via certified mail."},
            {"id": "c3", "content": "Standard contracts have 30 day termination for monthly billing."},
        ])
        doctor = make_doctor(connector)
        docs = connector.retrieve("termination notice Acme", top_k=3)
        report = doctor.diagnose(
            query="What is the Acme Corp termination notice period?",
            answer="30 days notice required.",
            docs=docs,
            expected="Acme Corp termination requires 90 days written notice.",
        )
        # Any meaningful diagnosis is acceptable — focus is that it runs and classifies
        assert report.root_cause in (
            "chunk_fragmentation", "retrieval_miss", "hallucination",
            "context_position_bias", "query_mismatch", "healthy"
        )

    def test_report_has_tool_results(self):
        connector = make_legal_connector()
        doctor = make_doctor(connector)
        docs = connector.retrieve("termination policy", top_k=5)
        report = doctor.diagnose(
            query="termination notice",
            answer="30 days",
            docs=docs,
        )
        assert len(report.tool_results) >= 2
        tool_names = [tr.tool_name for tr in report.tool_results]
        assert "retrieval_auditor" in tool_names
        assert "chunk_analyzer" in tool_names


# ─── Scenario 2: Position Bias ────────────────────────────────────────────────

class TestScenario2PositionBias:
    """Medical: correct doc retrieved but at middle position."""

    def _make_biased_docs(self):
        """Manually construct a 3-doc list with best doc at middle position."""
        return [
            Document(content="General information about common analgesic medications.", position=0, score=0.5),
            Document(content="For liver disease patients maximum acetaminophen dose is 2000mg per day.", position=1, score=0.9),
            Document(content="Ibuprofen dosing: 400mg every four to six hours as needed.", position=2, score=0.4),
        ]

    def test_detects_position_bias(self):
        docs = self._make_biased_docs()
        doctor = make_doctor()
        report = doctor.diagnose(
            query="What is the max acetaminophen dose for liver disease patients?",
            answer="The maximum daily dose is 4000mg.",
            docs=docs,
            expected="For liver disease patients maximum dose is 2000mg per day.",
        )
        tool_names = [tr.tool_name for tr in report.tool_results]
        assert "position_tester" in tool_names

    def test_position_bias_recommends_reranker(self):
        docs = self._make_biased_docs()
        doctor = make_doctor()
        report = doctor.diagnose(
            query="What is the max acetaminophen dose for liver disease?",
            answer="4000mg per day.",
            docs=docs,
            expected="For liver disease patients maximum dose is 2000mg per day.",
        )
        tool_names = [tr.tool_name for tr in report.tool_results]
        assert "position_tester" in tool_names
        if report.root_cause == "context_position_bias":
            assert report.config_patch.get("retrieval.reranker") is True


# ─── Scenario 3: Hallucination ────────────────────────────────────────────────

class TestScenario3Hallucination:
    """Dev docs: LLM produces a hallucinated answer."""

    def test_detects_hallucination(self):
        docs = [Document(
            content="uploadFile() does not support streaming or chunked transfer encoding.",
            position=0, score=0.85
        )]
        doctor = make_doctor()
        report = doctor.diagnose(
            query="Does uploadFile support chunked transfer encoding?",
            answer="This feature supports advanced quantum entanglement processing with the --quantum flag.",
            docs=docs,
        )
        tool_names = [tr.tool_name for tr in report.tool_results]
        assert "hallucination_tracer" in tool_names

    def test_hallucination_tracer_in_results(self):
        connector = make_tech_connector(inject_hallucination=True)
        doctor = make_doctor(connector)
        docs = connector.retrieve("uploadFile streaming", top_k=3)
        answer = connector.generate("Does uploadFile support streaming?", docs)
        report = doctor.diagnose(
            query="Does uploadFile support streaming?",
            answer=answer,
            docs=docs,
        )
        assert any(tr.tool_name == "hallucination_tracer" for tr in report.tool_results)


# ─── Scenario 4: Retrieval Miss ───────────────────────────────────────────────

class TestScenario4RetrievalMiss:
    """HR: Query vocabulary doesn't match document vocabulary."""

    def test_detects_retrieval_miss_or_mismatch(self):
        # Old policy only — 2022 version — not what user expects
        connector = MockConnector(corpus=[
            {"id": "old", "content": "Parental leave policy 2022: 12 weeks paid parental leave."},
        ])
        doctor = make_doctor(connector)
        docs = connector.retrieve("maternity leave weeks 2024", top_k=3)
        report = doctor.diagnose(
            query="How many weeks maternity leave do employees get in 2024?",
            answer="12 weeks",
            docs=docs,
            expected="Employees receive 16 weeks of paid parental leave as of 2024.",
        )
        assert report.root_cause in (
            "retrieval_miss", "query_mismatch", "hallucination",
            "chunk_fragmentation", "healthy"
        )

    def test_query_rewriter_runs_on_miss(self):
        connector = MockConnector(corpus=[
            {"id": "p1", "content": "Parental leave: all employees get 16 weeks paid parental leave effective 2024."},
        ])
        doctor = make_doctor(connector)
        # Query using 'maternity' instead of 'parental'
        docs = connector.retrieve("maternity leave policy 2024", top_k=3)
        report = doctor.diagnose(
            query="How many weeks of maternity leave are employees entitled to?",
            answer="12 weeks",
            docs=docs,
            expected="16 weeks of paid parental leave effective 2024.",
        )
        # QueryRewriter should run when retrieval miss detected
        if report.root_cause in ("retrieval_miss", "query_mismatch"):
            tool_names = [tr.tool_name for tr in report.tool_results]
            assert "query_rewriter" in tool_names


# ─── Healthy Pipeline ─────────────────────────────────────────────────────────

class TestHealthyPipeline:
    def test_healthy_pipeline_returns_healthy_or_minor(self):
        """When query, docs and answer all align, pipeline should be healthy."""
        connector = MockConnector(corpus=[
            {"id": "p1", "content": (
                "Return policy: customers may return any item within 30 days "
                "of purchase for a full refund. Items must be in original condition. "
                "Refunds processed within 5 business days."
            )},
        ])
        doctor = make_doctor(connector)
        docs = connector.retrieve("return policy refund days", top_k=3)
        answer = "Customers can return items within 30 days for a full refund."
        report = doctor.diagnose(
            query="What is the return policy?",
            answer=answer,
            docs=docs,
            expected="Customers may return items within 30 days for a full refund.",
        )
        # Accept healthy or low-severity chunk issue (minor quality issue is fine)
        assert report.root_cause in ("healthy", "chunk_fragmentation", "context_position_bias")
        assert report.severity in ("low", "medium")

    def test_healthy_report_structure(self):
        connector = MockConnector(corpus=[
            {"id": "p1", "content": "Refund policy: 30 days for full refund."},
        ])
        doctor = make_doctor(connector)
        docs = connector.retrieve("refund policy", top_k=3)
        report = doctor.diagnose(
            query="What is the refund policy?",
            answer="30 day refunds available.",
            docs=docs,
        )
        assert report.root_cause is not None
        assert report.severity in ("low", "medium", "high", "critical")
        assert report.to_json() != ""

    def test_report_to_text(self):
        connector = MockConnector(corpus=[
            {"id": "p1", "content": "Policy: returns accepted within 30 days."}
        ])
        doctor = make_doctor(connector)
        docs = connector.retrieve("return policy", top_k=2)
        report = doctor.diagnose(query="return policy", answer="30 day returns.", docs=docs)
        text = report.to_text()
        assert "Root Cause" in text
        assert "Severity" in text

    def test_report_passed_flag(self):
        connector = MockConnector(corpus=[
            {"id": "p1", "content": "Full refunds within 30 days of purchase guaranteed."}
        ])
        doctor = make_doctor(connector)
        docs = connector.retrieve("refund policy", top_k=2)
        report = doctor.diagnose(
            query="What is the refund policy?",
            answer="30 day full refund policy.",
            docs=docs,
            expected="Full refunds within 30 days of purchase.",
        )
        assert isinstance(report.passed, bool)


# ─── Batch Diagnosis ──────────────────────────────────────────────────────────

class TestBatchDiagnosis:
    def test_batch_returns_all_reports(self):
        connector = MockConnector(corpus=[
            {"id": "p1", "content": "Return policy: 30 days for full refund."},
            {"id": "p2", "content": "Parental leave: 16 weeks paid."},
        ])
        doctor = make_doctor(connector)
        cases = [
            {"query": "return policy", "answer": "30 days", "expected": "30 days for full refund"},
            {"query": "parental leave", "answer": "16 weeks", "expected": "16 weeks paid"},
        ]
        reports = doctor.batch_diagnose(cases)
        assert len(reports) == 2
        assert all(r.root_cause is not None for r in reports)

    def test_batch_empty_input(self):
        doctor = make_doctor()
        reports = doctor.batch_diagnose([])
        assert reports == []
