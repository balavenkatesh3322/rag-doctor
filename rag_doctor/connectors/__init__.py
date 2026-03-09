from .base import Document, PipelineConnector
from .mock import MockConnector

__all__ = ["Document", "PipelineConnector", "MockConnector"]

# Ollama connector available if ollama is running
def get_ollama_connector(**kwargs):
    from .ollama_connector import OllamaConnector
    return OllamaConnector(**kwargs)
