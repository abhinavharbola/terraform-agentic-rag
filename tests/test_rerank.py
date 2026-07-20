from unittest.mock import patch

from src.rerank import rerank_and_gate


def _candidate(text, resource_type=None, authority=None):
    return {"text": text, "metadata": {"resource_type": resource_type, "source_authority": authority}}


@patch("src.rerank.settings")
@patch("src.rerank._ranker")
def test_empty_candidates_returns_empty(mock_ranker, mock_settings):
    assert rerank_and_gate("how do I create a resource?", []) == []


@patch("src.rerank.settings")
@patch("src.rerank._ranker")
def test_below_threshold_candidates_are_dropped_not_reordered_to_bottom(mock_ranker, mock_settings):
    mock_settings.rerank_score_threshold = 0.5
    mock_ranker.rerank.return_value = [
        {"id": 0, "score": 0.9},
        {"id": 1, "score": 0.2},
    ]
    candidates = [_candidate("relevant chunk"), _candidate("noisy chunk")]

    survivors = rerank_and_gate("question", candidates)

    assert len(survivors) == 1
    assert survivors[0]["text"] == "relevant chunk"


@patch("src.rerank.settings")
@patch("src.rerank._ranker")
def test_zero_survivors_when_all_below_threshold(mock_ranker, mock_settings):
    mock_settings.rerank_score_threshold = 0.5
    mock_ranker.rerank.return_value = [{"id": 0, "score": 0.1}]
    survivors = rerank_and_gate("question", [_candidate("weak chunk")])
    assert survivors == []


@patch("src.rerank.settings")
@patch("src.rerank._ranker")
def test_official_authority_boost_flips_a_close_tie(mock_ranker, mock_settings):
    mock_settings.rerank_score_threshold = 0.5
    mock_ranker.rerank.return_value = [
        {"id": 0, "score": 0.60},
        {"id": 1, "score": 0.59},
    ]
    candidates = [
        _candidate("community chunk", authority="community"),
        _candidate("official chunk", authority="official"),
    ]

    survivors = rerank_and_gate("question", candidates)

    # official's +0.02 boost (0.59 -> 0.61) edges out community's 0.60
    assert survivors[0]["text"] == "official chunk"
    assert len(survivors) == 2


@patch("src.rerank.settings")
@patch("src.rerank._ranker")
def test_official_authority_boost_does_not_override_a_clear_gap(mock_ranker, mock_settings):
    mock_settings.rerank_score_threshold = 0.5
    mock_ranker.rerank.return_value = [
        {"id": 0, "score": 0.80},
        {"id": 1, "score": 0.55},
    ]
    candidates = [
        _candidate("community chunk", authority="community"),
        _candidate("official chunk", authority="official"),
    ]

    survivors = rerank_and_gate("question", candidates)

    # official's +0.02 boost (0.55 -> 0.57) is nowhere near community's 0.80
    assert survivors[0]["text"] == "community chunk"