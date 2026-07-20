from flashrank import Ranker, RerankRequest

from src.config import settings

# constructed once at import time, CPU/ONNX, no GPU dependency.
_ranker = Ranker(model_name=settings.rerank_model)

AUTHORITY_BOOST = 0.02  # tie-breaker only, never overrides the rank order on its own


def rerank_and_gate(question: str, candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []

    passages = [
        {"id": i, "text": candidate["text"], "meta": candidate["metadata"]}
        for i, candidate in enumerate(candidates)
    ]
    request = RerankRequest(query=question, passages=passages)
    reranked = _ranker.rerank(request)

    scored = []
    for result in reranked:
        candidate = candidates[result["id"]]
        score = result["score"]
        if candidate["metadata"].get("source_authority") == "official":
            score += AUTHORITY_BOOST
        scored.append({**candidate, "rerank_score": score})

    scored.sort(key=lambda c: c["rerank_score"], reverse=True)

    survivors = [c for c in scored if c["rerank_score"] >= settings.rerank_score_threshold]
    return survivors