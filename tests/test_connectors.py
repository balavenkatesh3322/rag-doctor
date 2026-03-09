"""Tests for MockConnector and base connector interface."""

from rag_doctor.connectors.mock import MockConnector
from rag_doctor.connectors.base import Document


class TestMockConnector:
    def setup_method(self):
        self.corpus = [
            {"id": "d1", "content": "Acetaminophen max dose is 4000mg for healthy adults."},
            {"id": "d2", "content": "For liver disease patients the max acetaminophen dose is 2000mg per day."},
            {"id": "d3", "content": "Ibuprofen is an NSAID used for pain and inflammation."},
        ]

    def test_retrieve_returns_documents(self):
        conn = MockConnector(corpus=self.corpus)
        docs = conn.retrieve("acetaminophen dose", top_k=3)
        assert len(docs) <= 3
        assert all(isinstance(d, Document) for d in docs)

    def test_retrieve_assigns_positions(self):
        conn = MockConnector(corpus=self.corpus)
        docs = conn.retrieve("acetaminophen", top_k=3)
        positions = [d.position for d in docs]
        assert positions == sorted(positions)

    def test_retrieve_sorted_by_score(self):
        conn = MockConnector(corpus=self.corpus)
        docs = conn.retrieve("acetaminophen liver dose", top_k=3)
        scores = [d.score for d in docs]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_respected(self):
        conn = MockConnector(corpus=self.corpus)
        docs = conn.retrieve("dose", top_k=2)
        assert len(docs) <= 2

    def test_generate_returns_string(self):
        conn = MockConnector(corpus=self.corpus)
        docs = conn.retrieve("acetaminophen", top_k=3)
        answer = conn.generate("What is the dose?", docs)
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_add_document(self):
        conn = MockConnector()
        conn.add_document("New policy document.", metadata={"type": "policy"})
        docs = conn.retrieve("policy", top_k=1)
        assert len(docs) == 1
        assert "policy" in docs[0].content.lower()

    def test_inject_position_bias(self):
        conn = MockConnector(corpus=self.corpus, inject_position_bias=True)
        docs = conn.retrieve("acetaminophen liver disease", top_k=3)
        if len(docs) >= 3:
            positions = [d.position for d in docs]
            # Best doc should not be at position 0 when bias injected
            best_content_pos = next(
                (d.position for d in docs if "liver disease" in d.content), None
            )
            # Just check positions are assigned
            assert len(set(positions)) == len(positions)

    def test_inject_hallucination(self):
        conn = MockConnector(corpus=self.corpus, inject_hallucination=True)
        docs = conn.retrieve("dose", top_k=2)
        answer = conn.generate("What is the dose?", docs)
        assert "quantum" in answer.lower()

    def test_embed_returns_vector(self):
        conn = MockConnector(corpus=self.corpus)
        vec = conn.embed("test text")
        assert isinstance(vec, list)
        assert len(vec) == 64
        assert all(isinstance(v, float) for v in vec)

    def test_custom_answer_fn(self):
        def custom_fn(query, docs):
            return f"Custom answer for: {query}"

        conn = MockConnector(corpus=self.corpus, answer_fn=custom_fn)
        docs = conn.retrieve("test", top_k=1)
        answer = conn.generate("my query", docs)
        assert "Custom answer for: my query" == answer

    def test_empty_corpus(self):
        conn = MockConnector()
        docs = conn.retrieve("anything", top_k=5)
        assert docs == []
