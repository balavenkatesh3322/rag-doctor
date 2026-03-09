# Root Causes

rag-doctor assigns every diagnosis to one of six root cause codes.

---

## RC-0: `healthy`

**All diagnostic tools passed.** Your RAG pipeline is working correctly for this query.

```
Root Cause : healthy (RC-0)
Severity   : low
Finding    : All tools passed. Pipeline looks healthy.
```

**What to do:** Nothing. Monitor for regressions.

---

## RC-1: `retrieval_miss`

**The correct document was not retrieved.** The vector search failed to find the document that contains the answer.

### Symptoms
- Answer is completely wrong or says "I don't know"
- RetrievalAuditor score below `recall_threshold`
- Often appears suddenly after changing the embedding model

### Common Causes
| Cause | Frequency |
|---|---|
| Embedding model changed (corpus vs query mismatch) | Very common |
| Document not in corpus | Common |
| `top_k` too small | Common |
| Query vocabulary doesn't match document vocabulary | Common (→ RC-5) |
| Embedding model not fine-tuned for domain | Occasional |

### Fixes
```yaml
# Increase top_k
retrieval:
  top_k: 10

# Add hybrid search
retrieval:
  hybrid_search: true
  bm25_weight: 0.3
```

Or in Python:
```python
# Re-embed corpus with consistent model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
# Use same model for both corpus and query embedding
```

---

## RC-2: `context_position_bias`

**The correct document was retrieved but placed in the middle of the context window.** LLMs consistently underperform on middle-positioned context, even when the answer is present.

### The "Lost in the Middle" Effect

Research (Liu et al. 2023) shows LLMs are best at using context that appears at the **beginning** or **end** of the context window. A document at position 2 out of 5 may be almost completely ignored.

### Symptoms
- Retrieval scores look good (audit passes)
- But the answer is still wrong or incomplete
- PositionTester detects best document at position 1, 2, or 3 (not 0)

### Fixes

```python
# Option 1: Enable a reranker to move best doc to position 0
from rag_doctor.connectors.base import PipelineConnector

class MyConnector(PipelineConnector):
    def retrieve(self, query, top_k=5):
        docs = self.vector_store.search(query, top_k)
        docs = self.reranker.rerank(query, docs)  # push best to position 0
        return docs
```

```yaml
# Option 2: Enable reranker in config
retrieval:
  reranker: true
  reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2
```

```python
# Option 3: Use Maximal Marginal Relevance (MMR)
docs = vector_store.max_marginal_relevance_search(query, k=5)
```

---

## RC-3: `chunk_fragmentation`

**Key context is split across chunks.** Chunks are truncated mid-sentence, incoherent, or too small to contain a complete answer.

### Symptoms
- ChunkAnalyzer finds truncated chunks (start lowercase or end without punctuation)
- Low intra-chunk coherence score
- Answer is incomplete or missing key facts
- Answer contains partial information that's technically in the corpus

### Chunk Quality Metrics

| Metric | Healthy | Fragmented |
|---|---|---|
| Truncation rate | < 10% | > 30% |
| Coherence score (TF-IDF) | > 0.25 | < 0.15 |
| Coherence score (sent-trans) | > 0.55 | < 0.35 |
| Chunk size | 80–800 tokens | < 50 or > 1200 tokens |

### Fixes

```python
# Switch from fixed to recursive chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,
    chunk_overlap=32,
    separators=["\n\n", "\n", ". ", " "],
)
chunks = splitter.split_text(document)
```

```yaml
# In rag-doctor.yaml
chunking:
  strategy: recursive      # was: fixed
  chunk_size: 256          # was: 512
  chunk_overlap: 32
```

Use `corpus_texts` with `diagnose()` to trigger `ChunkOptimizer` and get ranked strategy recommendations:

```python
report = Doctor.default().diagnose(
    query="...", answer="...", docs=docs,
    corpus_texts=my_corpus,  # ← enables ChunkOptimizer
)
```

---

## RC-4: `hallucination`

**The LLM generated claims not grounded in the retrieved documents.**

### Symptoms
- HallucinationTracer faithfulness score below `faithfulness_threshold`
- Answer contains specific facts (names, numbers, dates) not present in docs
- Answer is confident but wrong

### Faithfulness Score

`faithfulness_score = grounded_claims / total_claims`

| Score | Interpretation |
|---|---|
| 0.90–1.00 | Fully grounded |
| 0.70–0.89 | Mostly grounded, minor additions |
| 0.50–0.69 | Significant hallucination |
| < 0.50 | Critical hallucination |

### Fixes

```python
# Strengthen the system prompt
system_prompt = """You are a helpful assistant. Answer ONLY from the provided sources.
If the sources do not contain the answer, say "I don't have that information."
Do NOT add any information not explicitly present in the sources."""
```

```python
# Lower LLM temperature
llm = ChatOpenAI(model="gpt-4o", temperature=0.0)  # was temperature=0.7
```

```python
# Add a post-generation faithfulness filter
report = Doctor.default().diagnose(query, answer, docs)
if report.root_cause == "hallucination":
    answer = regenerate_with_stricter_prompt(query, docs)
```

---

## RC-5: `query_mismatch`

**Query vocabulary doesn't match document vocabulary.** Retrieval fails because the user used different terminology than the documents.

### Examples

| User Query | Document Uses |
|---|---|
| "maternity leave policy" | "parental leave" |
| "fired" / "terminated" | "employment separation" |
| "PTO" | "paid time off" |
| "heart attack" | "myocardial infarction" |

### How rag-doctor Detects This

QueryRewriter tries three strategies and checks if any improve retrieval:
1. **HyDE** (Hypothetical Document Embedding): turns the query into a hypothetical answer
2. **Step-back**: broadens the query to a higher abstraction level
3. **Synonym expansion**: replaces domain-specific terms with alternatives

If a rewrite significantly improves the retrieval score, the root cause is `query_mismatch`.

### Fixes

```python
# Option 1: Use query expansion in production
def expand_query(query):
    expansions = {
        "maternity": "parental",
        "fired": "terminated",
        "pto": "paid time off",
    }
    for term, expansion in expansions.items():
        query = query.replace(term, f"{term} OR {expansion}")
    return query
```

```python
# Option 2: Add HyDE in production
def hyde_embed(query, llm):
    hypothetical_doc = llm.generate(f"Write a passage that answers: {query}")
    return embed(hypothetical_doc)  # embed the hypothetical doc, not the query
```

```yaml
# Option 3: Fine-tune your embedding model on domain data
# Use sentence-transformers fine-tuning with your corpus
```

---

## Severity Levels

| Severity | Meaning | Action |
|---|---|---|
| `low` | Minor issue or within tolerance | Monitor |
| `medium` | Noticeable quality degradation | Plan fix in next sprint |
| `high` | Significant failure, users affected | Fix this week |
| `critical` | Severe failure, pipeline unreliable | Fix immediately |
