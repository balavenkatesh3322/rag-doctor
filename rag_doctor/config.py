"""Configuration schema for rag-doctor (stdlib only)."""
from __future__ import annotations
from dataclasses import dataclass, field
import yaml


@dataclass
class RetrievalConfig:
    top_k: int = 5
    reranker: bool = False
    similarity_threshold: float = 0.0
    position_danger_zone_start: int = 1


@dataclass
class ChunkingConfig:
    strategy: str = "fixed"
    chunk_size: int = 512
    chunk_overlap: int = 64


@dataclass
class DiagnosisConfig:
    chunk_analyzer: bool = True
    retrieval_auditor: bool = True
    position_tester: bool = True
    hallucination_tracer: bool = True
    chunk_optimizer: bool = True
    query_rewriter: bool = True
    # Thresholds tuned for TF-IDF (the default fallback embedder).
    # TF-IDF paraphrase similarity ceiling is ~0.40, so thresholds are set below that.
    # With sentence-transformers installed these can be raised to 0.65–0.80.
    recall_threshold: float = 0.35        # flag retrieval miss below this
    faithfulness_threshold: float = 0.40  # flag hallucination below this
    coherence_threshold: float = 0.25     # flag chunk incoherence below this
    severity_threshold: str = "medium"


@dataclass
class RagDoctorConfig:
    pipeline_type: str = "custom"
    vector_db: str = "chroma"
    embedding_model: str = "all-MiniLM-L6-v2"
    llm: str = "mock"
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    diagnosis: DiagnosisConfig = field(default_factory=DiagnosisConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "RagDoctorConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        r = data.pop("retrieval", {})
        c = data.pop("chunking", {})
        d = data.pop("diagnosis", {})
        return cls(
            **{k: v for k, v in data.items() if k in cls.__dataclass_fields__},
            retrieval=RetrievalConfig(**r) if r else RetrievalConfig(),
            chunking=ChunkingConfig(**c) if c else ChunkingConfig(),
            diagnosis=DiagnosisConfig(**d) if d else DiagnosisConfig(),
        )

    @classmethod
    def default(cls) -> "RagDoctorConfig":
        return cls()
