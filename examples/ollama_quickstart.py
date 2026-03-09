"""
rag-doctor × Ollama quickstart
================================
The simplest possible way to use rag-doctor with a local LLM.

Requirements:
    brew install ollama
    ollama serve          # in a separate terminal
    ollama pull llama3.2

Run:
    python examples/ollama_quickstart.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rag_doctor import Doctor
from rag_doctor.connectors.ollama_connector import OllamaConnector

# ── 1. Create connector (auto-selects best installed model) ───────────────────
connector = OllamaConnector(
    # model="llama3.2",          # uncomment to force a specific model
    # embed_model="nomic-embed-text",  # uncomment for better embeddings
)

# ── 2. Load your documents ────────────────────────────────────────────────────
connector.load_corpus([
    {
        "id": "policy_001",
        "content": (
            "Enterprise Refund Policy: Enterprise customers are entitled to a full refund "
            "within 90 days of purchase. Requests must be submitted via the account portal. "
            "Refunds are processed within 10 business days."
        ),
        "metadata": {"source": "policy_doc", "type": "policy"}
    },
    {
        "id": "policy_002",
        "content": (
            "Standard Refund Policy: Standard tier customers may request refunds within "
            "30 days of purchase. Items must be unused and in original packaging."
        ),
        "metadata": {"source": "policy_doc", "type": "policy"}
    },
    {
        "id": "faq_001",
        "content": "For billing questions, contact support@company.com or call 1-800-SUPPORT.",
        "metadata": {"source": "faq"}
    },
])

# ── 3. Retrieve and generate ──────────────────────────────────────────────────
query = "How long do enterprise customers have to request a refund?"
docs  = connector.retrieve(query, top_k=3)
answer = connector.generate(query, docs)

print(f"Query  : {query}")
print(f"Answer : {answer}")
print()

# ── 4. Diagnose ───────────────────────────────────────────────────────────────
doctor = Doctor.default(connector)
report = doctor.diagnose(
    query    = query,
    answer   = answer,
    docs     = docs,
    expected = "Enterprise customers have 90 days to request a refund.",
)

print(report.to_text())
