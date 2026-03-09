"""Shared test fixtures and corpus data."""
from rag_doctor.connectors.mock import MockConnector

LEGAL_CORPUS = [
    {"id": "contract_001", "content": "Acme Corp Master Services Agreement. Termination clause: Either party may terminate this agreement with 90 days written notice to the other party. Termination must be delivered via certified mail.", "metadata": {"type": "contract", "client": "acme"}},
    {"id": "contract_002", "content": "Standard service agreement. Termination may occur with 30 days notice for month-to-month contracts. Enterprise contracts require longer notice periods.", "metadata": {"type": "contract", "client": "standard"}},
    {"id": "policy_001", "content": "General refund policy applies to all standard tier customers. Enterprise customers have custom SLAs defined in their individual contracts.", "metadata": {"type": "policy"}},
    {"id": "faq_001", "content": "Frequently asked questions about billing and payment terms for all service tiers.", "metadata": {"type": "faq"}},
    {"id": "sla_001", "content": "Service Level Agreement defines uptime guarantees and support response times for enterprise customers.", "metadata": {"type": "sla"}},
]

MEDICAL_CORPUS = [
    {"id": "drug_001", "content": "Acetaminophen dosing guidelines. Standard adult dose: up to 4000mg per day. For patients with liver disease or hepatic impairment, maximum daily dose is reduced to 2000mg. Consult physician before use.", "metadata": {"type": "clinical"}},
    {"id": "drug_002", "content": "General pain management overview. Common OTC analgesics include ibuprofen and acetaminophen.", "metadata": {"type": "general"}},
    {"id": "drug_003", "content": "Ibuprofen dosing: 400-800mg every 4-6 hours. Maximum 3200mg per day.", "metadata": {"type": "clinical"}},
]

TECH_CORPUS = [
    {"id": "api_001", "content": "uploadFile() method: Uploads a file to the server. Parameters: file (File), metadata (dict, optional). Returns: UploadResponse with file_id. Does not support streaming.", "metadata": {"type": "api"}},
    {"id": "api_002", "content": "streaming uploads are handled by the streamUpload() method, not uploadFile(). Use streamUpload() for large files over 100MB.", "metadata": {"type": "api"}},
    {"id": "api_003", "content": "API authentication uses Bearer tokens. Include Authorization header in all requests.", "metadata": {"type": "auth"}},
]

HR_CORPUS_OLD = [
    {"id": "policy_2022", "content": "Parental leave policy (2022): All full-time employees are entitled to 12 weeks of paid parental leave.", "metadata": {"type": "policy", "year": "2022"}},
]

HR_CORPUS_NEW = [
    {"id": "policy_2024", "content": "Parental leave policy (2024): Effective January 2024, all full-time employees are entitled to 16 weeks of paid parental leave.", "metadata": {"type": "policy", "year": "2024"}},
]


def make_legal_connector(**kwargs) -> MockConnector:
    return MockConnector(corpus=LEGAL_CORPUS, **kwargs)


def make_medical_connector(**kwargs) -> MockConnector:
    return MockConnector(corpus=MEDICAL_CORPUS, **kwargs)


def make_tech_connector(**kwargs) -> MockConnector:
    return MockConnector(corpus=TECH_CORPUS, **kwargs)


def make_hr_connector(**kwargs) -> MockConnector:
    return MockConnector(corpus=HR_CORPUS_OLD + HR_CORPUS_NEW, **kwargs)
