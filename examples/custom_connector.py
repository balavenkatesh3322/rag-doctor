"""
Example: Building a custom connector for your own RAG stack.

This shows how to wrap any retrieval + generation system
so rag-doctor can diagnose it.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from typing import List
from rag_doctor import Doctor
from rag_doctor.connectors.base import PipelineConnector, Document


class MyRAGConnector(PipelineConnector):
    """
    Drop-in template. Replace the bodies of retrieve() and generate()
    with your own retrieval and LLM calls.
    """

    def __init__(self, vector_db_client=None, llm_client=None):
        self.vector_db = vector_db_client
        self.llm = llm_client

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Call your vector DB here.

        Example with Chroma:
            results = self.vector_db.query(query_texts=[query], n_results=top_k)
            return [
                Document(
                    content=doc,
                    metadata=meta,
                    score=float(dist),
                    position=i,
                )
                for i, (doc, meta, dist) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ))
            ]
        """
        # Stub for this example
        return [
            Document(
                content=f"Stub result {i} for query: {query}",
                metadata={"source": f"doc_{i}"},
                score=0.9 - i * 0.1,
                position=i,
            )
            for i in range(min(top_k, 3))
        ]

    def generate(self, query: str, docs: List[Document]) -> str:
        """
        Call your LLM here.

        Example with OpenAI:
            context = "\n\n".join(d.content for d in docs)
            response = self.llm.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": f"Answer using only this context:\n{context}"},
                    {"role": "user",   "content": query},
                ]
            )
            return response.choices[0].message.content
        """
        # Stub for this example
        context = " | ".join(d.content[:50] for d in docs)
        return f"Based on retrieved docs: {context}"


# Usage
if __name__ == "__main__":
    connector = MyRAGConnector()
    doctor = Doctor.default(connector=connector)

    query = "What is our enterprise refund policy?"
    docs = connector.retrieve(query, top_k=5)
    answer = connector.generate(query, docs)

    report = doctor.diagnose(
        query=query,
        answer=answer,
        docs=docs,
        expected="Enterprise customers receive 90-day refunds.",
    )
    print(report.to_text())
