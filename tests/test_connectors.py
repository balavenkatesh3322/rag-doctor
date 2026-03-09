"""Tests for MockConnector and base connector interface."""

from rag_doctor.connectors.mock import MockConnector
from rag_doctor.connectors.base import Document
from rag_doctor.embeddings import CharFreqEmbedder


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
            assert len(set(positions)) == len(positions)

    def test_inject_hallucination(self):
        conn = MockConnector(corpus=self.corpus, inject_hallucination=True)
        docs = conn.retrieve("dose", top_k=2)
        answer = conn.generate("What is the dose?", docs)
        assert "quantum" in answer.lower()

    def test_embed_returns_fixed_dim_vector(self):
        """embed() must return a fixed-dim vector (CharFreqEmbedder.FIXED_DIM)."""
        conn = MockConnector(corpus=self.corpus)
        vec = conn.embed("test text")
        assert isinstance(vec, list)
        assert len(vec) == CharFreqEmbedder.FIXED_DIM
        assert all(isinstance(v, float) for v in vec)

    def test_custom_answer_fn(self):
        def custom_fn(query, docs):
            return f"Custom answer for: {query}"
        conn = MockConnector(corpus=self.corpus, answer_fn=custom_fn)
        docs = conn.retrieve("dose", top_k=2)
        answer = conn.generate("What is the dose?", docs)
        assert "Custom answer" in answer

    def test_load_corpus_replaces_existing(self):
        conn = MockConnector(corpus=self.corpus)
        conn.load_corpus([{"id": "new", "content": "Completely new corpus."}])
        docs = conn.retrieve("new corpus", top_k=1)
        assert len(docs) == 1

    def test_semantic_retrieval_accuracy(self):
        """Real embedder should rank liver-disease doc higher for liver query."""
        conn = MockConnector(corpus=self.corpus)
        docs = conn.retrieve("liver disease acetaminophen dose", top_k=3)
        assert len(docs) > 0
        # Top doc should be about liver disease, not ibuprofen
        assert "liver" in docs[0].content.lower() or "acetaminophen" in docs[0].content.lower()

    def test_ibuprofen_scores_low_for_acetaminophen_query(self):
        """Ibuprofen doc should score lower than acetaminophen docs for acetaminophen query."""
        conn = MockConnector(corpus=self.corpus)
        docs = conn.retrieve("acetaminophen dosing for liver patients", top_k=3)
        # Find ibuprofen doc
        ibu_doc = next((d for d in docs if "ibuprofen" in d.content.lower()), None)
        ace_doc = next((d for d in docs if "acetaminophen" in d.content.lower()), None)
        if ibu_doc and ace_doc:
            assert ace_doc.score >= ibu_doc.score
