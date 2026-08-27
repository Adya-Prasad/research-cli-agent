"""
This is the provider boundary. `DenseRetriever` depends on `Embedder`, not on Sentence Transformers.
"""
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from research_agent.retrieval.models import SearchHit

FloatVector = NDArray[np.float32]
FloatMatrix = NDArray[np.float32]

class Embedder(Protocol):
    """
    embed_documents(): corpus/chunk representation
    embed_query():     user-question representation
    """
    def embed_documents(self, text: Sequence[str]) -> FloatMatrix: ...

    def embed_query(self, text: str) -> FloatVector: ...

class Retriever(Protocol):
    def search(self, query: str, top_k: int = 5) -> list[SearchHit]: ...