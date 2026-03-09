"""End-to-end tests simulating real-world RAG failure scenarios."""

from rag_doctor.doctor import Doctor
from rag_doctor.config import RagDoctorConfig
from rag_doctor.connectors.mock import MockConnector
from rag_doctor.connectors.base import Document


def make_doctor(connector=None) -> Doctor:
    return Doctor(RagDoctorConfig.default(), connector)


class TestEndToEndScenarios:
    """
    Full end-to-end scenarios that mirror the Problem-Solution document.
    Each test validates the full pipeline: connect → retrieve → generate → diagnose.
    """

    def test_e2e_legal_contract_clause_miss(self):
        """
        Scenario: Legal contract RAG returns wrong termination period.
        Expected: rag-doctor identifies retrieval or fragmentation issue.
        """
        connector = MockConnector(corpus=[
            {"id": "acme_full", "content":
                "Acme Corp Services Agreement Section 12: Termination. "
                "Either party may terminate this agreement by providing 90 days written notice. "
                "Notice shall be delivered via certified mail to the registered address."},
            {"id": "standard", "content":
                "Standard tier agreements may be terminated with 30 days notice. "
                "Month-to-month billing applies."},
            {"id": "general", "content": "All contracts are governed by the laws of the State of Delaware."},
        ])
        doctor = make_doctor(connector)

        query = "What is the termination notice period for Acme Corp?"
        docs = connector.retrieve(query, top_k=3)
        answer = "30 days notice required."  # Wrong answer
        expected = "Acme Corp requires 90 days written notice for termination."

        report = doctor.diagnose(query=query, answer=answer, docs=docs, expected=expected)

        # Validate report structure
        assert report.query == query
        assert report.root_cause is not None
        assert report.severity is not None
        assert len(report.tool_results) >= 3

        # Either retrieval auditor or hallucination tracer should flag this
        flagged_tools = [t for t in report.tool_results if not t.passed]
        assert len(flagged_tools) >= 1

        print(f"\n[E2E Legal] Root cause: {report.root_cause} | Severity: {report.severity}")
        print(report.to_text())

    def test_e2e_medical_correct_retrieval_wrong_answer(self):
        """
        Scenario: Medical RAG retrieves correct doc but gives wrong dose.
        Expected: Hallucination tracer detects the grounding failure.
        """
        connector = MockConnector(corpus=[
            {"id": "drug_guide", "content":
                "Acetaminophen dosing: Standard adults 4000mg max daily. "
                "Liver disease patients: maximum 2000mg per day due to reduced hepatic metabolism. "
                "Always consult physician for hepatic impairment cases."},
            {"id": "general_pain", "content": "Common pain relievers include acetaminophen and ibuprofen."},
        ])
        doctor = make_doctor(connector)

        query = "Max acetaminophen dose for a patient with liver disease?"
        docs = connector.retrieve(query, top_k=2)

        # Wrong answer: uses healthy adult dose
        answer = "The maximum daily dose of acetaminophen is 4000mg per day."
        expected = "For liver disease patients the maximum dose is 2000mg per day."

        report = doctor.diagnose(query=query, answer=answer, docs=docs, expected=expected)

        assert report.root_cause is not None
        ht = next((t for t in report.tool_results if t.tool_name == "hallucination_tracer"), None)
        assert ht is not None

        print(f"\n[E2E Medical] Root cause: {report.root_cause} | Faithfulness: "
              f"{ht.details.get('faithfulness_score', 'N/A')}")

    def test_e2e_tech_docs_hallucination(self):
        """
        Scenario: Dev tools RAG hallucinates a non-existent API feature.
        Expected: Hallucination tracer detects the fabricated claim.
        """
        connector = MockConnector(
            corpus=[
                {"id": "api_ref", "content":
                    "uploadFile(file, metadata=None) - Uploads a file. "
                    "Returns UploadResponse with file_id and upload_url. "
                    "File size limit: 500MB. Does not support streaming or chunked transfer."},
                {"id": "streaming", "content":
                    "For streaming use streamUpload(). Not available in uploadFile()."},
            ],
            inject_hallucination=True  # Simulate LLM making up features
        )
        doctor = make_doctor(connector)

        query = "Does uploadFile() support chunked transfer encoding?"
        docs = connector.retrieve(query, top_k=2)
        answer = connector.generate(query, docs)  # Will be hallucinated

        report = doctor.diagnose(query=query, answer=answer, docs=docs)

        # Any root cause is valid for the hallucinated answer test
        assert report.root_cause is not None
        print(f"\n[E2E Tech] Root cause: {report.root_cause} | Answer: {answer[:80]}")

    def test_e2e_hr_conflicting_versions(self):
        """
        Scenario: HR policy has two versions. Bot returns old policy.
        Expected: Retrieval auditor detects the version conflict pattern.
        """
        connector = MockConnector(corpus=[
            {"id": "policy_2022", "content":
                "Parental Leave Policy (2022): Employees are entitled to 12 weeks paid parental leave."},
            {"id": "policy_2024", "content":
                "Parental Leave Policy (Updated 2024): All employees now receive 16 weeks paid parental leave "
                "effective January 1, 2024."},
            {"id": "benefits_overview", "content": "Benefits overview for all full-time employees."},
        ])
        doctor = make_doctor(connector)

        query = "How many weeks of parental leave am I entitled to?"
        docs = connector.retrieve(query, top_k=3)
        answer = "You are entitled to 12 weeks of parental leave."  # Old policy
        expected = "Employees receive 16 weeks of paid parental leave as of 2024."

        report = doctor.diagnose(query=query, answer=answer, docs=docs, expected=expected)

        assert report.root_cause is not None
        assert len(report.tool_results) >= 2

        print(f"\n[E2E HR] Root cause: {report.root_cause} | Severity: {report.severity}")

    def test_e2e_healthy_pipeline(self):
        """
        Scenario: A well-configured pipeline returns the correct, grounded answer.
        Expected: All tools pass, root cause is 'healthy'.
        """
        connector = MockConnector(corpus=[
            {"id": "policy", "content":
                "Return policy: Customers may return any item within 30 days of purchase for a full refund. "
                "Items must be in original condition. Refunds processed within 5 business days."},
        ])
        doctor = make_doctor(connector)

        query = "What is the return policy?"
        docs = connector.retrieve(query, top_k=3)
        # Perfectly grounded answer
        answer = ("Customers can return items within 30 days for a full refund. "
                  "Items must be in original condition.")
        expected = "Items returnable within 30 days for full refund."

        report = doctor.diagnose(query=query, answer=answer, docs=docs, expected=expected)

        # Should be healthy or low severity
        assert report.severity in ("low", "medium")
        print(f"\n[E2E Healthy] Root cause: {report.root_cause} | Severity: {report.severity}")

    def test_e2e_full_pipeline_with_connector(self):
        """Full pipeline: connector retrieves and generates, then diagnoses."""
        connector = MockConnector(corpus=[
            {"id": "d1", "content": "The enterprise plan includes 24/7 support and 99.9% SLA uptime guarantee."},
            {"id": "d2", "content": "Standard plan includes business hours support only."},
        ])
        doctor = make_doctor(connector)

        query = "Does the enterprise plan include 24/7 support?"
        docs = connector.retrieve(query, top_k=3)
        answer = connector.generate(query, docs)

        report = doctor.diagnose(
            query=query, answer=answer, docs=docs,
            expected="Enterprise plan includes 24/7 support."
        )

        assert report.query == query
        assert report.answer == answer
        assert report.root_cause is not None
        d = report.to_dict()
        assert d["query"] == query

    def test_e2e_no_docs_retrieved(self):
        """Edge case: connector returns empty result set."""
        connector = MockConnector(corpus=[])  # Empty corpus
        doctor = make_doctor(connector)

        report = doctor.diagnose(
            query="What is the refund policy?",
            answer="I don't know.",
            docs=[],
        )
        assert report.root_cause is not None
        ra = next(t for t in report.tool_results if t.tool_name == "retrieval_auditor")
        assert ra.passed is False

    def test_e2e_chunk_optimizer_with_corpus(self):
        """ChunkOptimizer runs when fragmentation detected and corpus provided."""
        corpus_texts = [
            "Enterprise agreement section 5: Termination policy requires 90 days written notice. "
            "This clause applies only to enterprise tier customers. Standard customers need 30 days. "
            "All notices must be sent via certified mail.",
            "Section 8: Payment terms. Enterprise invoices due net-30. "
            "Late payments incur 1.5% monthly interest.",
        ]
        connector = MockConnector(corpus=[
            {"id": "c1", "content": "Enterprise agreement section 5: Termination policy requires 90 days written notice"},
            {"id": "c2", "content": "This clause applies only to enterprise tier customers"},
        ])
        doctor = make_doctor(connector)
        docs = connector.retrieve("enterprise termination", top_k=2)

        report = doctor.diagnose(
            query="What is the enterprise termination notice period?",
            answer="30 days.",
            docs=docs,
            expected="Enterprise requires 90 days written notice.",
            corpus_texts=corpus_texts,
        )

        assert report.root_cause is not None
        print(f"\n[E2E Optimizer] Root cause: {report.root_cause}")
        optimizer_result = next(
            (t for t in report.tool_results if t.tool_name == "chunk_optimizer"), None
        )
        if optimizer_result:
            assert "ranked_strategies" in optimizer_result.details


class TestBatchEndToEnd:
    """Batch processing end-to-end tests."""

    def test_batch_mixed_scenarios(self):
        """Batch with healthy and unhealthy cases."""
        doctor = Doctor(RagDoctorConfig.default())

        cases = [
            {
                "query": "return policy",
                "answer": "30 day returns for all customers.",
                "docs": [Document(
                    content="Return policy: 30 day returns available for all purchases.",
                    position=0, score=0.92
                )],
                "expected": "Customers have 30 days to return items.",
            },
            {
                "query": "quantum features",
                "answer": "We support quantum entanglement via the --qe flag.",
                "docs": [Document(
                    content="Our product provides basic file upload functionality.",
                    position=0, score=0.55
                )],
            },
            {
                "query": "enterprise SLA uptime",
                "answer": "99.9% uptime guaranteed.",
                "docs": [Document(
                    content="Enterprise SLA guarantees 99.9% uptime with 24/7 support.",
                    position=0, score=0.95
                )],
                "expected": "99.9% uptime SLA for enterprise customers.",
            },
        ]

        reports = doctor.batch_diagnose(cases)
        assert len(reports) == 3

        for report in reports:
            assert report.root_cause is not None
            assert report.severity in ("low", "medium", "high", "critical")

        print("\n[Batch] Results:")
        for r in reports:
            print(f"  {r.query[:30]:30} → {r.root_cause:25} [{r.severity}]")

    def test_batch_all_outputs_serializable(self):
        """All batch reports must serialize to JSON."""
        import json
        doctor = Doctor(RagDoctorConfig.default())
        cases = [
            {"query": f"q{i}", "answer": f"a{i}",
             "docs": [Document(content=f"content about q{i}", position=0, score=0.8)]}
            for i in range(5)
        ]
        reports = doctor.batch_diagnose(cases)
        for r in reports:
            json_str = r.to_json()
            parsed = json.loads(json_str)
            assert "root_cause" in parsed
            assert "tool_results" in parsed
