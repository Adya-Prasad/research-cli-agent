from collections.abc import Sequence
from typing import Any
import numpy as np

from pathlib import Path

from research_agent.retrieval.models import ResearchDocument
from research_agent.retrieval.port import FloatMatrix, FloatVector

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, device:str = "cpu" ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model: Any = SentenceTransformer(model_name, device=device)

    def embed_documents(self, docs_text: Sequence[str]) -> FloatMatrix:
        vectors = self._model.encode_document(
            list(docs_text),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, query_text: str) -> FloatVector:
        vectors = self._model.encode_query(
            query_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

def make_document(text: str, name: str = "paper.md") -> ResearchDocument:
    print(ResearchDocument.from_text(source_path=Path(name), doc_type="md", text=text))
    return ResearchDocument.from_text(source_path=Path(name), doc_type="md", text=text)

