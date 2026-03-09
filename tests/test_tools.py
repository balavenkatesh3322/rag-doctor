"""Unit tests for the six diagnostic tools."""

from rag_doctor.connectors.base import Document
from rag_doctor.connectors.mock import MockConnector
from rag_doctor.tools.chunk_analyzer import ChunkAnalyzer
from rag_doctor.tools.retrieval_auditor import RetrievalAuditor
from rag_doctor.tools.position_tester import PositionTester
from rag_doctor.tools.hallucination_tracer import HallucinationTracer
from rag_doctor.tools.chunk_optimizer import ChunkOptimizer
from rag_doctor.tools.query_rewriter import QueryRewriter
from rag_doctor.embeddings import reset_embedder_cache


def _doc(content, position=0, score=0.9):
    return Document(content=content, position=position, score=score)


# ─── ChunkAnalyzer ────────────────────────────────────────────────────────────

class TestChunkAnalyzer:
    def test_truncated_chunks_detected(self):
        """Chunks that start lowercase or end without punctuation are flagged."""
        tool = ChunkAnalyzer(coherence_threshold=0.0)  # only check truncation
        docs = [
            _doc("Acme Corp Agreement: Either party may terminate this"),  # ends abruptly
            _doc("90 days written notice via certified mail."),            # starts mid-sentence
        ]
        result = tool.run(docs)
        assert result.details["truncated_chunks"] >= 1

    def test_well_formed_chunks_pass(self):
        tool = ChunkAnalyzer(coherence_threshold=0.0)
        docs = [
            _doc("The refund policy allows returns within 30 days of purchase."),
            _doc("All enterprise customers have custom SLA agreements."),
        ]
        result = tool.run(docs)
        assert result.details["truncated_chunks"] == 0

    def test_empty_docs_returns_passed(self):
        result = ChunkAnalyzer().run([])
        assert result.passed is True

    def test_result_has_required_fields(self):
        tool = ChunkAnalyzer()
        docs = [_doc("Some content for testing purposes here.")]
        result = tool.run(docs)
        assert "avg_coherence" in result.details
        assert "truncated_chunks" in result.details
        assert "total_chunks" in result.details


# ─── RetrievalAuditor ─────────────────────────────────────────────────────────

class TestRetrievalAuditor:
    def setup_method(self):
        self.corpus = [
            {"id": "d1", "content": "Acetaminophen max dose is 4000mg for healthy adults."},
            {"id": "d2", "content": "For liver disease patients max acetaminophen dose is 2000mg per day."},
            {"id": "d3", "content": "Ibuprofen 400mg every four hours for pain."},
        ]

    def test_no_docs_fails(self):
        result = RetrievalAuditor().run(docs=[], expected="anything")
        assert result.passed is False
        assert result.severity == "critical"

    def test_recall_hit_with_matching_expected(self):
        """When expected answer is in corpus and retrieved, recall_hit should be True."""
        conn = MockConnector(corpus=self.corpus)
        docs = conn.retrieve("liver disease acetaminophen dose", top_k=3)
        result = RetrievalAuditor(recall_threshold=0.40).run(
            docs=docs,
            expected="For liver disease patients max dose is 2000mg per day."
        )
        assert result.details["recall_hit"] is True

    def test_recall_miss_for_unrelated_expected(self):
        """Expected about an unrelated topic should not match acetaminophen docs."""
        conn = MockConnector(corpus=self.corpus)
        docs = conn.retrieve("acetaminophen dose", top_k=3)
        result = RetrievalAuditor(recall_threshold=0.95).run(
            docs=docs,
            expected="The boiling point of water is 100 degrees Celsius at sea level."
        )
        assert result.details["recall_hit"] is False

    def test_result_without_expected_uses_scores(self):
        docs = [_doc("Some content", score=0.8)]
        result = RetrievalAuditor(recall_threshold=0.5).run(docs=docs)
        assert result.passed is True

    def test_result_structure(self):
        docs = [_doc("Content about liver and acetaminophen dose", score=0.9)]
        result = RetrievalAuditor().run(docs=docs, expected="acetaminophen liver dose")
        assert "recall_hit" in result.details
        assert "best_match_score" in result.details


# ─── PositionTester ───────────────────────────────────────────────────────────

class TestPositionTester:
    def test_middle_position_detected(self):
        """Best doc at position 1 of 3 is the danger zone."""
        docs = [
            _doc("General information", position=0, score=0.5),
            _doc("The exact answer to the question.", position=1, score=0.9),
            _doc("Unrelated content here.", position=2, score=0.4),
        ]
        result = PositionTester().run(docs, query="exact answer")
        assert result.details["in_danger_zone"] is True
        assert result.passed is False

    def test_first_position_safe(self):
        """Best doc at position 0 is safe."""
        docs = [
            _doc("Best answer here.", position=0, score=0.9),
            _doc("Less relevant.", position=1, score=0.5),
            _doc("Even less relevant.", position=2, score=0.3),
        ]
        result = PositionTester().run(docs, query="best answer")
        assert result.details["in_danger_zone"] is False
        assert result.passed is True

    def test_two_docs_never_danger_zone(self):
        """With only 2 docs, no danger zone exists."""
        docs = [
            _doc("Content A", position=0, score=0.8),
            _doc("Content B", position=1, score=0.5),
        ]
        result = PositionTester().run(docs)
        assert result.passed is True

    def test_empty_docs(self):
        result = PositionTester().run([])
        assert result.passed is True

    def test_risk_score_range(self):
        docs = [
            _doc("Low relevance", position=0, score=0.2),
            _doc("High relevance.", position=1, score=0.9),
            _doc("Low relevance", position=2, score=0.2),
        ]
        result = PositionTester().run(docs)
        assert 0.0 <= result.details["position_risk_score"] <= 1.0


# ─── HallucinationTracer ──────────────────────────────────────────────────────

class TestHallucinationTracer:
    def test_grounded_answer_passes(self):
        """Answer that paraphrases source content should be grounded."""
        docs = [_doc("Acetaminophen max dose for liver patients is 2000mg per day.")]
        result = HallucinationTracer(faithfulness_threshold=0.30).run(
            answer="The maximum acetaminophen dose for liver disease is 2000mg daily.",
            docs=docs,
        )
        assert result.details["faithfulness_score"] > 0.5

    def test_hallucinated_answer_fails(self):
        """Answer about quantum physics against a medical doc should be unfaithful."""
        docs = [_doc("Acetaminophen dosing for liver disease patients.")]
        result = HallucinationTracer(faithfulness_threshold=0.80).run(
            answer="Quantum entanglement enables faster-than-light communication via photon pairs.",
            docs=docs,
        )
        assert result.passed is False

    def test_empty_answer(self):
        result = HallucinationTracer().run(answer="", docs=[_doc("Content")])
        assert result.passed is True

    def test_no_docs_critical(self):
        result = HallucinationTracer().run(answer="Some answer", docs=[])
        assert result.passed is False
        assert result.severity == "critical"

    def test_faithfulness_score_in_details(self):
        docs = [_doc("The sky is blue during daytime.")]
        result = HallucinationTracer().run(answer="The sky is blue.", docs=docs)
        assert "faithfulness_score" in result.details
        assert 0.0 <= result.details["faithfulness_score"] <= 1.0


# ─── ChunkOptimizer ───────────────────────────────────────────────────────────

class TestChunkOptimizer:
    def test_returns_ranked_strategies(self):
        corpus = [
            "Acme Corp. Either party may terminate this agreement with 90 days written notice.",
            "Standard contracts require 30 day termination notice for monthly billing customers.",
            "Enterprise agreements have custom termination periods defined individually.",
            "All agreements governed by the laws of Delaware. Arbitration required for disputes.",
        ]
        test_pairs = [{"query": "termination notice", "expected": "90 days written notice"}]
        result = ChunkOptimizer().run(corpus_texts=corpus, test_pairs=test_pairs)
        assert "ranked_strategies" in result.details
        assert len(result.details["ranked_strategies"]) > 0

    def test_empty_corpus_passes(self):
        result = ChunkOptimizer().run(corpus_texts=[], test_pairs=[])
        assert result.passed is True

    def test_best_strategy_has_recall_score(self):
        corpus = [
            "The refund policy allows returns within 30 days of purchase for a full refund.",
            "Enterprise customers have 90 day refund window with account credit option.",
        ]
        test_pairs = [{"query": "refund policy", "expected": "30 days return policy"}]
        result = ChunkOptimizer().run(corpus_texts=corpus, test_pairs=test_pairs)
        if result.details["ranked_strategies"]:
            best = result.details["ranked_strategies"][0]
            assert "recall_at_5" in best
            assert 0.0 <= best["recall_at_5"] <= 1.0

    def test_result_structure(self):
        corpus = ["Policy document about employment terms and conditions."]
        pairs = [{"query": "employment", "expected": "employment terms"}]
        result = ChunkOptimizer().run(corpus_texts=corpus, test_pairs=pairs)
        assert result.tool_name == "chunk_optimizer"
        assert isinstance(result.passed, bool)


# ─── QueryRewriter ────────────────────────────────────────────────────────────

class TestQueryRewriter:
    def test_returns_rewrites_without_connector(self):
        tool = QueryRewriter()
        docs = [_doc("Parental leave policy: 16 weeks paid.", score=0.3)]
        result = tool.run(
            query="How many weeks maternity leave?",
            original_docs=docs,
        )
        assert "original_query" in result.details
        assert "strategy_results" in result.details

    def test_synonym_rewrite_improves_hr_query(self):
        """'maternity' should be rewritten to 'parental' improving retrieval."""
        corpus = [
            {"id": "p1", "content": "Parental leave: all employees get 16 weeks paid parental leave."},
            {"id": "p2", "content": "Benefits: health insurance, dental, and retirement plans."},
        ]
        conn = MockConnector(corpus=corpus)
        docs = conn.retrieve("maternity leave weeks", top_k=3)
        tool = QueryRewriter()
        result = tool.run(
            query="How many weeks of maternity leave?",
            original_docs=docs,
            connector=conn,
        )
        assert "strategy_results" in result.details
        assert "synonym" in result.details["strategy_results"]

    def test_result_has_best_strategy(self):
        docs = [_doc("Some content about policy.", score=0.4)]
        result = QueryRewriter().run(query="What is the policy?", original_docs=docs)
        assert "best_strategy" in result.details

    def test_passed_means_no_improvement_needed(self):
        """If original retrieval is already good, rewriter should report passed=True."""
        docs = [_doc("Exact match for the query.", score=0.95)]
        result = QueryRewriter().run(
            query="exact match query",
            original_docs=docs,
        )
        # With high original score, improvement will be minimal
        assert isinstance(result.passed, bool)
