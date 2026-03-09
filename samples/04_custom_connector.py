#!/usr/bin/env python3
"""
Sample 4: Custom Connector
===========================
Connect rag-doctor to your own retrieval stack.
Shows a simple in-memory connector you can adapt for
Chroma, Pinecone, pgvector, LangChain, or any backend.

Run:
    python samples/04_custom_connector.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List
from rag_doctor import Doctor
from rag_doctor.connectors.base import PipelineConnector, Document
from rag_doctor.tools._embed_utils import fit_and_embed, similarity


class SimpleInMemoryConnector(PipelineConnector):
    """
    In-memory connector using rag-doctor's own embedder.
    Replace the internals with calls to your real retrieval system.

    For Chroma:
        results = self.collection.query(query_texts=[query], n_results=top_k)
        return [Document(content=d, score=1-s, position=i) for i,(d,s) in enumerate(...)]

    For Pinecone:
        results = self.index.query(vector=embed(query), top_k=top_k, include_metadata=True)
        return [Document(content=m.metadata["text"], score=m.score, position=i) ...]
    """

    def __init__(self, corpus: List[str]):
        self.corpus = corpus

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        # Embed query + all corpus texts together (critical for TF-IDF correctness)
        embedder, vecs = fit_and_embed([query] + self.corpus)
        q_vec    = vecs[0]
        doc_vecs = vecs[1:]

        scored = sorted(
            [(i, similarity(embedder, q_vec, dv)) for i, dv in enumerate(doc_vecs)],
            key=lambda x: x[1], reverse=True
        )

        return [
            Document(content=self.corpus[idx], score=score, position=pos)
            for pos, (idx, score) in enumerate(scored[:top_k])
        ]

    def generate(self, query: str, docs: List[Document]) -> str:
        """
        Simple extractive generation — returns first sentence of best doc.

        Replace with your LLM call:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role":"system","content":"Answer from sources only."},
                          {"role":"user","content":f"Sources:\n{context}\n\nQ:{query}"}])
            return response.choices[0].message.content
        """
        if not docs:
            return "No documents were retrieved."
        sentences = [s.strip() for s in docs[0].content.split(".") if s.strip()]
        return sentences[0] + "." if sentences else docs[0].content[:200]


LEGAL_CORPUS = [
    "Enterprise software licenses require 90 days written notice for termination.",
    "Standard licenses can be terminated with 30 days notice.",
    "All license agreements are governed by the laws of the State of California.",
    "Breach of license terms may result in immediate termination without notice.",
    "Intellectual property rights remain with the licensor at all times.",
    "Liability is limited to the fees paid in the 12 months preceding the claim.",
]


def main():
    print("=" * 60)
    print("Sample 4: Custom Connector")
    print("=" * 60)

    connector = SimpleInMemoryConnector(corpus=LEGAL_CORPUS)
    query     = "What is the notice period to terminate an enterprise license?"
    docs      = connector.retrieve(query, top_k=5)
    answer    = connector.generate(query, docs)

    print(f"\nQuery : {query}")
    print(f"Answer: {answer}")
    print(f"\nTop 3 retrieved docs:")
    for d in docs[:3]:
        print(f"  [{d.position}] score={d.score:.3f}  {d.content[:80]}")

    report = Doctor.default(connector).diagnose(
        query    = query,
        answer   = answer,
        docs     = docs,
        expected = "Enterprise licenses require 90 days written notice for termination.",
    )

    print(f"\nDiagnosis:")
    print(f"  Root Cause : {report.root_cause} ({report.root_cause_id})")
    print(f"  Severity   : {report.severity.upper()}")
    print(f"  Finding    : {report.finding}")
    if report.fix_suggestion:
        print(f"  Fix        : {report.fix_suggestion}")


if __name__ == "__main__":
    main()
