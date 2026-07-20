from unittest.mock import MagicMock, patch

import pytest
from openai import APIError

from src.llm import generate_main, generate_planner


def _mock_completion(content: str) -> MagicMock:
    completion = MagicMock()
    completion.choices[0].message.content = content
    return completion


def _api_error() -> APIError:
    return APIError("boom", request=MagicMock(), body=None)


@patch("src.llm.groq_client")
@patch("src.llm.nim_client")
def test_uses_primary_when_it_succeeds(mock_nim, mock_groq):
    mock_nim.chat.completions.create.return_value = _mock_completion("nim answer")
    result = generate_main([{"role": "user", "content": "hi"}])
    assert result.provider == "nim"
    assert result.content == "nim answer"
    assert mock_groq.chat.completions.create.called is False


@patch("src.llm.groq_client")
@patch("src.llm.nim_client")
def test_falls_back_to_groq_when_primary_fails(mock_nim, mock_groq):
    mock_nim.chat.completions.create.side_effect = _api_error()
    mock_groq.chat.completions.create.return_value = _mock_completion("groq answer")
    result = generate_main([{"role": "user", "content": "hi"}])
    assert result.provider == "groq"
    assert result.content == "groq answer"


@patch("src.llm.groq_client")
@patch("src.llm.nim_client")
def test_raises_clear_error_when_both_providers_fail(mock_nim, mock_groq):
    mock_nim.chat.completions.create.side_effect = _api_error()
    mock_groq.chat.completions.create.side_effect = _api_error()
    with pytest.raises(RuntimeError, match="both nim and groq failed"):
        generate_main([{"role": "user", "content": "hi"}])


@patch("src.llm.groq_client")
@patch("src.llm.nim_client")
def test_planner_never_touches_main_generation_models(mock_nim, mock_groq):
    mock_nim.chat.completions.create.return_value = _mock_completion("planner answer")
    generate_planner([{"role": "user", "content": "rewrite this"}])
    called_model = mock_nim.chat.completions.create.call_args.kwargs["model"]
    assert "8b" in called_model