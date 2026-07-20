from unittest.mock import MagicMock, patch

import pytest

from src.cache import (
    exact_cache_get,
    exact_cache_set,
    normalize_exact,
    semantic_cache_get,
    semantic_cache_set,
)


@pytest.fixture(autouse=True)
def clear_exact_cache():
    from src.cache import _exact_cache
    _exact_cache.clear()
    yield
    _exact_cache.clear()


def test_normalize_collapses_whitespace_case_and_punctuation():
    assert normalize_exact("  How Do I Create a Resource?  ") == "how do i create a resource"


def test_normalize_does_not_collapse_different_questions():
    a = normalize_exact("how do I create a resource")
    b = normalize_exact("how do I destroy a resource")
    assert a != b


def test_exact_cache_roundtrip():
    exact_cache_set("How do I create a resource?", "answer text")
    assert exact_cache_get("how do i create a resource") == "answer text"


def test_exact_cache_miss_returns_none():
    assert exact_cache_get("nothing stored for this question") is None


@patch("src.cache.embed_for_cache", return_value=[0.1] * 768)
@patch("src.cache.qdrant_client")
def test_semantic_cache_hit_above_threshold(mock_qdrant, mock_embed):
    mock_point = MagicMock()
    mock_point.payload = {"answer": "cached semantic answer"}
    mock_qdrant.query_points.return_value.points = [mock_point]

    result = semantic_cache_get("how terraform is used")
    assert result == "cached semantic answer"


@patch("src.cache.embed_for_cache", return_value=[0.1] * 768)
@patch("src.cache.qdrant_client")
def test_semantic_cache_miss_below_threshold(mock_qdrant, mock_embed):
    mock_qdrant.query_points.return_value.points = []
    assert semantic_cache_get("an unrelated question") is None


@patch("src.cache.embed_for_cache", return_value=[0.1] * 768)
@patch("src.cache.qdrant_client")
def test_semantic_cache_set_upserts_with_score_threshold_applied_on_read(mock_qdrant, mock_embed):
    semantic_cache_set("canonical question", "an answer")
    assert mock_qdrant.upsert.called