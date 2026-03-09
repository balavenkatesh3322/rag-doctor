"""rag-doctor: Agentic RAG pipeline failure diagnosis."""
from .doctor import Doctor
from .config import RagDoctorConfig
from .report import DiagnosisReport

__version__ = "1.0.0"
__all__ = ["Doctor", "RagDoctorConfig", "DiagnosisReport"]
