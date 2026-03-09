# Writing Connectors

A `PipelineConnector` bridges rag-doctor to your production RAG stack.

---

## The Interface

```python
from rag_doctor.connectors.base import PipelineConnector, Document
from typing import List

class PipelineConnector:
    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve top-k documents for query."""
        ...

    def generate(self, query: str, docs: List[Document]) -> str:
        """Generate answer given query and retrieved docs."""
        ...

    def embed(self, text: str) -> List[float]:
        """Optional: embed text with your own model."""
        ...
```

---

## Chroma Connector

```python
import chromadb
from rag_doctor.connectors.base import PipelineConnector, Document

class ChromaConnector(PipelineConnector):
    def __init__(self, collection_name: str, persist_dir: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_collection(collection_name)

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
        )
        docs = []
        for i, (text, distance, meta) in enumerate(zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        )):
            docs.append(Document(
                content=text,
                score=1.0 - distance,  # Chroma returns distance, not similarity
                position=i,
                metadata=meta or {},
            ))
        return docs

    def generate(self, query: str, docs: List[Document]) -> str:
        # Use your LLM here, or a simple extractive answer for testing
        context = "\n\n".join(d.content for d in docs[:3])
        return f"Based on the documents: {context[:200]}..."


# Usage
from rag_doctor import Doctor

connector = ChromaConnector("my_collection")
doctor = Doctor.default(connector)

docs = connector.retrieve("What is the return policy?", top_k=5)
answer = connector.generate("What is the return policy?", docs)

report = doctor.diagnose(query="What is the return policy?", answer=answer, docs=docs)
print(report.root_cause)
```

---

## Pinecone Connector

```python
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rag_doctor.connectors.base import PipelineConnector, Document

class PineconeConnector(PipelineConnector):
    def __init__(self, index_name: str, api_key: str):
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        query_vec = self.embedder.encode(query).tolist()
        results = self.index.query(
            vector=query_vec,
            top_k=top_k,
            include_metadata=True,
        )
        return [
            Document(
                content=match["metadata"].get("text", ""),
                score=match["score"],
                position=i,
                metadata=match["metadata"],
                doc_id=match["id"],
            )
            for i, match in enumerate(results["matches"])
        ]

    def generate(self, query: str, docs: List[Document]) -> str:
        # Integrate with your LLM
        ...
```

---

## pgvector Connector

```python
import psycopg2
from rag_doctor.connectors.base import PipelineConnector, Document
from rag_doctor.embeddings import get_embedder

class PGVectorConnector(PipelineConnector):
    def __init__(self, dsn: str):
        self.conn = psycopg2.connect(dsn)
        self.embedder = get_embedder()

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        query_vec = self.embedder.embed(query)
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT content, metadata,
                       1 - (embedding <=> %s::vector) AS score
                FROM documents
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_vec, query_vec, top_k))
            rows = cur.fetchall()
        return [
            Document(content=row[0], metadata=row[1] or {}, score=row[2], position=i)
            for i, row in enumerate(rows)
        ]

    def generate(self, query: str, docs: List[Document]) -> str:
        ...
```

---

## LangChain Connector

```python
from langchain.schema import Document as LCDocument
from rag_doctor.connectors.base import PipelineConnector, Document

class LangChainConnector(PipelineConnector):
    def __init__(self, retriever, llm):
        self.retriever = retriever  # Any LangChain retriever
        self.llm = llm

    def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        lc_docs: List[LCDocument] = self.retriever.invoke(query)
        return [
            Document(
                content=doc.page_content,
                metadata=doc.metadata,
                score=doc.metadata.get("score", 0.5),
                position=i,
            )
            for i, doc in enumerate(lc_docs[:top_k])
        ]

    def generate(self, query: str, docs: List[Document]) -> str:
        context = "\n\n".join(d.content for d in docs)
        return self.llm.invoke(f"Context:\n{context}\n\nQuestion: {query}")
```

---

## Tips for Writing Connectors

1. **Score normalization**: Different backends use different score ranges. Normalize to 0.0–1.0. For distance metrics (lower = better), use `score = 1.0 - distance`.

2. **Position matters**: Always set `position=i` in order of retrieval. PositionTester relies on this.

3. **Don't over-engineer generate()**: For testing, a simple extractive answer is fine. For production diagnosis, use your real LLM.

4. **Test with MockConnector first**: Validate your diagnosis logic works before connecting to your real stack.

```python
# Always test with Mock first
from rag_doctor.connectors.mock import MockConnector
conn = MockConnector(corpus=["doc 1...", "doc 2..."])
report = Doctor.default(conn).diagnose(query="test", answer="test answer")
assert report  # sanity check

# Then swap in your real connector
conn = MyRealConnector(...)
report = Doctor.default(conn).diagnose(query="test", answer="test answer")
```
