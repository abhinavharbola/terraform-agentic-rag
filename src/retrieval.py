from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.clients import qdrant_client
from src.config import settings
from src.embeddings import embed_query


def retrieve(question: str, resource_type: str | None = None, top_k: int | None = None) -> list[dict]:
    vector = embed_query(question)
    query_filter = None
    if resource_type:
        query_filter = Filter(
            must=[FieldCondition(key="resource_type", match=MatchValue(value=resource_type))]
        )

    results = qdrant_client.query_points(
        collection_name=settings.qdrant_docs_collection,
        query=vector,
        query_filter=query_filter,
        limit=top_k or settings.rerank_top_k,
    ).points

    return [
        {
            "text": point.payload["text"],
            "metadata": point.payload.get("metadata", {}),
            "retrieval_score": point.score,
        }
        for point in results
    ]