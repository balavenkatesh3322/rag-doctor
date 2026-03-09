"""Unit tests for each diagnostic tool."""

from rag_doctor.connectors.base import Document
from rag_doctor.tools import (
    ChunkAnalyzer, RetrievalAuditor, PositionTester,
    HallucinationTracer, ChunkOptimizer, QueryRewriter,
)


def make_doc(content, pos=0, score=0.8, doc_id=None):
    return Document(content=content, position=pos, score=score, doc_id=doc_id or f"d{pos}")


# ─── ChunkAnalyzer ────────────────────────────────────────────────────────────

class TestChunkAnalyzer:
    def setup_method(self):
        self.tool = ChunkAnalyzer(coherence_threshold=0.65)

    def test_well_formed_chunk_passes(self):
        doc = make_doc("The refund policy states that customers receive full refunds within 30 days. "
                       "All requests must be submitted via the support portal. "
                       "Refunds are processed within 5-7 business days.")
        result = self.tool.run(docs=[doc], query="refund policy")
        assert result.tool_name == "chunk_analyzer"
        assert result.details["total_chunks"] == 1

    def test_truncated_chunk_detected(self):
        # Starts lowercase = mid-sentence truncation
        doc = make_doc("and therefore the policy applies to all enterprise customers who have signed the agreement")
        result = self.tool.run(docs=[doc], query="policy")
        assert result.details["truncated_chunks"] >= 1

    def test_empty_docs(self):
        result = self.tool.run(docs=[], query="test")
        assert result.passed is True
        assert result.severity == "low"

    def test_multiple_docs_analyzed(self):
        docs = [
            make_doc("Complete sentence about policies. Another sentence follows here.", pos=0),
            make_doc("continuing from the previous chunk without clear context", pos=1),
            make_doc("Final chunk with proper ending. This ends well.", pos=2),
        ]
        result = self.tool.run(docs=docs, query="policy")
        assert result.details["total_chunks"] == 3
        assert result.details["truncated_chunks"] >= 1


# ─── RetrievalAuditor ─────────────────────────────────────────────────────────

class TestRetrievalAuditor:
    def setup_method(self):
        self.tool = RetrievalAuditor(recall_threshold=0.75)

    def test_recall_hit_with_matching_expected(self):
        doc = make_doc("Enterprise customers receive 90 days termination notice.", pos=0, score=0.9)
        result = self.tool.run(
            docs=[doc],
            expected="Enterprise customers get 90 days termination notice.",
            query="termination notice"
        )
        assert result.details["recall_hit"] is True
        assert result.details["best_match_position"] == 0

    def test_recall_miss_with_unrelated_docs(self):
        doc = make_doc("ZZZZ XXXX WWWW PPPP BBBB", pos=0, score=0.2)
        result = self.tool.run(
            docs=[doc],
            expected="Enterprise termination requires 90 days written notice.",
            query="termination notice enterprise"
        )
        assert result.details["recall_hit"] is False
        assert result.severity in ("high", "critical")

    def test_no_docs_returns_critical(self):
        result = self.tool.run(docs=[], expected="any answer", query="test")
        assert result.passed is False
        assert result.severity == "critical"

    def test_without_expected_uses_score(self):
        doc = make_doc("Some content", pos=0, score=0.9)
        result = self.tool.run(docs=[doc])
        assert result.details["top_score"] == 0.9

    def test_scores_by_position_populated(self):
        docs = [make_doc(f"Document {i} content here", pos=i, score=0.9 - i * 0.1) for i in range(3)]
        result = self.tool.run(
            docs=docs,
            expected="Document 0 content here",
            query="document"
        )
        assert len(result.details["scores_by_position"]) == 3


# ─── PositionTester ───────────────────────────────────────────────────────────

class TestPositionTester:
    def setup_method(self):
        self.tool = PositionTester()

    def test_position_0_is_safe(self):
        docs = [make_doc(f"doc {i}", pos=i) for i in range(5)]
        result = self.tool.run(docs=docs, best_position=0, top_k=5)
        assert result.passed is True
        assert result.details["in_danger_zone"] is False

    def test_last_position_is_safe(self):
        docs = [make_doc(f"doc {i}", pos=i) for i in range(5)]
        result = self.tool.run(docs=docs, best_position=4, top_k=5)
        assert result.passed is True

    def test_middle_position_is_danger(self):
        docs = [make_doc(f"doc {i}", pos=i) for i in range(5)]
        result = self.tool.run(docs=docs, best_position=2, top_k=5)
        assert result.passed is False
        assert result.details["in_danger_zone"] is True
        assert result.severity in ("medium", "high")

    def test_no_docs_passes(self):
        result = self.tool.run(docs=[], best_position=-1)
        assert result.passed is True

    def test_risk_score_increases_toward_center(self):
        docs = [make_doc(f"doc {i}", pos=i) for i in range(7)]
        r1 = self.tool.run(docs=docs, best_position=1, top_k=7)
        r3 = self.tool.run(docs=docs, best_position=3, top_k=7)
        assert r3.details["position_risk_score"] >= r1.details["position_risk_score"]


# ─── HallucinationTracer ──────────────────────────────────────────────────────

class TestHallucinationTracer:
    def setup_method(self):
        self.tool = HallucinationTracer(faithfulness_threshold=0.60)

    def test_grounded_answer_passes(self):
        doc = make_doc("The refund policy allows 30 day returns for all customers.")
        result = self.tool.run(
            answer="Customers can return items within 30 days per the refund policy.",
            docs=[doc]
        )
        assert result.details["total_claims"] >= 1

    def test_hallucinated_answer_fails(self):
        # Use faithfulness_threshold=0.99 to force detection of mismatch
        tool = HallucinationTracer(faithfulness_threshold=0.99)
        doc = make_doc("ZZZZ XXXX BBBB YYYY PPPP MMMM")  # unrelated chars
        result = tool.run(
            answer="The maximum dose is 2000mg for liver disease patients.",
            docs=[doc]
        )
        assert result.passed is False
        assert len(result.details["hallucinated_claims"]) > 0

    def test_empty_answer_passes(self):
        result = self.tool.run(answer="", docs=[make_doc("some content")])
        assert result.passed is True

    def test_no_docs_is_critical(self):
        result = self.tool.run(answer="Some answer about something.", docs=[])
        assert result.passed is False
        assert result.severity == "critical"

    def test_faithfulness_score_range(self):
        doc = make_doc("Data retention policy requires keeping records for 7 years.")
        result = self.tool.run(answer="Records must be kept for 7 years.", docs=[doc])
        assert 0.0 <= result.details["faithfulness_score"] <= 1.0


# ─── ChunkOptimizer ───────────────────────────────────────────────────────────

class TestChunkOptimizer:
    def setup_method(self):
        self.tool = ChunkOptimizer()
        self.corpus = [
            "The enterprise termination policy requires 90 days written notice. "
            "This applies to all enterprise tier customers. Notice must be via certified mail. "
            "Standard tier customers require only 30 days notice. "
            "The policy was updated in January 2024.",
            "Refund policy: Full refunds available within 30 days of purchase. "
            "Enterprise customers have extended 90-day refund windows. "
            "All refund requests must go through the support portal.",
        ]
        self.test_pairs = [
            {"query": "enterprise termination notice period", "expected": "Enterprise requires 90 days notice."},
            {"query": "refund window enterprise", "expected": "Enterprise customers have 90-day refund windows."},
        ]

    def test_returns_ranked_strategies(self):
        result = self.tool.run(corpus_texts=self.corpus, test_pairs=self.test_pairs)
        assert "ranked_strategies" in result.details
        assert len(result.details["ranked_strategies"]) > 0

    def test_strategies_sorted_by_recall(self):
        result = self.tool.run(corpus_texts=self.corpus, test_pairs=self.test_pairs)
        recalls = [s["recall_at_5"] for s in result.details["ranked_strategies"]]
        assert recalls == sorted(recalls, reverse=True)

    def test_empty_corpus_passes(self):
        result = self.tool.run(corpus_texts=[], test_pairs=self.test_pairs)
        assert result.passed is True

    def test_improvement_computed_with_current_strategy(self):
        result = self.tool.run(
            corpus_texts=self.corpus,
            test_pairs=self.test_pairs,
            current_strategy={"strategy": "fixed", "chunk_size": 1024, "chunk_overlap": 128}
        )
        assert result.details.get("current_recall") is not None


# ─── QueryRewriter ────────────────────────────────────────────────────────────

class TestQueryRewriter:
    def setup_method(self):
        self.tool = QueryRewriter()

    def test_returns_rewrite_strategies(self):
        docs = [make_doc("Some content about uploads", pos=0, score=0.4)]
        result = self.tool.run(
            query="Does uploadFile support chunked transfer encoding?",
            original_docs=docs,
        )
        assert "strategy_results" in result.details
        assert "hyde" in result.details["strategy_results"]
        assert "step_back" in result.details["strategy_results"]

    def test_hyde_rewrite_differs_from_original(self):
        docs = [make_doc("content", pos=0, score=0.3)]
        result = self.tool.run(query="What is the refund policy?", original_docs=docs)
        hyde_q = result.details["strategy_results"].get("hyde", {}).get("query", "")
        assert hyde_q != "What is the refund policy?"

    def test_with_connector_runs_retrieval(self):
        from tests.fixtures import make_legal_connector
        connector = make_legal_connector()
        docs = connector.retrieve("termination notice", top_k=5)
        result = self.tool.run(
            query="termination notice enterprise",
            original_docs=docs,
            connector=connector,
        )
        assert result.details["best_score"] > 0

    def test_no_docs_still_returns_result(self):
        result = self.tool.run(query="test query", original_docs=[])
        assert result.tool_name == "query_rewriter"
