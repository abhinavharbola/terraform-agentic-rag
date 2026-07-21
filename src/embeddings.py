import numpy as np
from google.genai import types

from src.clients import gemini_client
from src.config import settings


def _normalize(vector: list[float]) -> list[float]:
    array = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(array)
    if norm == 0:
        return vector
    return (array / norm).tolist()


def embed_texts(texts: list[str], task_type: str) -> list[list[float]]:
    """task_type: RETRIEVAL_DOCUMENT for ingestion, RETRIEVAL_QUERY for queries,
    SEMANTIC_SIMILARITY for cache lookups."""
    vectors = []
    for text in texts:
        response = gemini_client.models.embed_content(
            model=settings.gemini_embedding_model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=settings.embedding_dim,
            ),
        )
        vectors.append(_normalize(response.embeddings[0].values))
    return vectors


def embed_query(text: str) -> list[float]:
    return embed_texts([text], task_type="RETRIEVAL_QUERY")[0]


def embed_document(text: str) -> list[float]:
    return embed_texts([text], task_type="RETRIEVAL_DOCUMENT")[0]


def embed_for_cache(text: str) -> list[float]:
    return embed_texts([text], task_type="SEMANTIC_SIMILARITY")[0]