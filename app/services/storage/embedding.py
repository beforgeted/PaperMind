"""Qwen embeddings as LangChain `Embeddings` implementations.

Two pluggable backends, selected by `settings.embedding_backend`:
- `dashscope`: Aliyun cloud API (Qwen text-embedding-v3, 1024 dims).
- `local`:     local SentenceTransformer (Qwen3-Embedding-0.6B).

DashScope's `text-embedding-v3` enforces a per-call batch limit of 25 inputs;
we batch transparently so callers can pass arbitrarily long lists.

`get_embeddings()` returns a singleton subclass of
`langchain_core.embeddings.Embeddings`; the extra `dim` attribute is used at
startup to align Elasticsearch dense_vector mapping with the model.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.embeddings import Embeddings
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Raised when an embedding backend fails."""


class _BaseEmbeddings(Embeddings):
    dim: int = 0


# ---------------------------------------------------------------------------
# DashScope backend
# ---------------------------------------------------------------------------


class DashScopeEmbeddings(_BaseEmbeddings):
    BATCH_SIZE = 10  # DashScope text-embedding-v3 request limit

    def __init__(self) -> None:
        try:
            import dashscope as _ds
            from dashscope import TextEmbedding
        except ImportError as exc:  # pragma: no cover
            raise EmbeddingError(
                "dashscope is not installed. `pip install dashscope`"
            ) from exc
        if not settings.dashscope_api_key:
            raise EmbeddingError(
                "DASHSCOPE_API_KEY is not set; cannot use the dashscope backend."
            )
        _ds.api_key = settings.dashscope_api_key
        self._client = TextEmbedding
        self._model = settings.embedding_model
        self.dim = settings.es_vector_dims
        logger.info(
            "DashScope embedding ready (model={}, dim={})", self._model, self.dim
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(EmbeddingError),
        reraise=True,
    )
    def _call(self, batch: List[str]) -> List[List[float]]:
        response = self._client.call(model=self._model, input=batch)
        status_code = getattr(response, "status_code", None) or response.get(
            "status_code", 200
        )
        if status_code != 200:
            msg = getattr(response, "message", None) or response.get(
                "message", "unknown"
            )
            raise EmbeddingError(f"DashScope error {status_code}: {msg}")

        output = getattr(response, "output", None) or response.get("output", {})
        embeddings = (
            output.get("embeddings")
            if isinstance(output, dict)
            else getattr(output, "embeddings", None)
        )
        if not embeddings:
            raise EmbeddingError("DashScope returned an empty embedding list")
        embeddings_sorted = sorted(embeddings, key=lambda e: e.get("text_index", 0))
        return [e["embedding"] for e in embeddings_sorted]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        out: List[List[float]] = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            out.extend(self._call(texts[i : i + self.BATCH_SIZE]))
        return out

    def embed_query(self, text: str) -> List[float]:
        return self._call([text])[0]


# ---------------------------------------------------------------------------
# Local SentenceTransformer backend
# ---------------------------------------------------------------------------


class LocalHFEmbeddings(_BaseEmbeddings):
    def __init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise EmbeddingError("sentence-transformers is not installed.") from exc
        logger.info("Loading local embedding model {}…", settings.local_embedding_model)
        self._model = SentenceTransformer(settings.local_embedding_model)
        self.dim = self._model.get_sentence_embedding_dimension()
        if self.dim != settings.es_vector_dims:
            logger.warning(
                "Local model dim {} != configured ES_VECTOR_DIMS {}; ES index "
                "will use the model's actual dim.",
                self.dim,
                settings.es_vector_dims,
            )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=16,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_embeddings: Optional[_BaseEmbeddings] = None


def get_embeddings() -> _BaseEmbeddings:
    global _embeddings
    if _embeddings is None:
        choice = settings.embedding_backend
        if choice == "dashscope":
            _embeddings = DashScopeEmbeddings()
        elif choice == "local":
            _embeddings = LocalHFEmbeddings()
        else:  # pragma: no cover
            raise EmbeddingError(f"Unknown EMBEDDING_BACKEND={choice!r}")
    return _embeddings
