import logging
from dataclasses import dataclass

from openai import OpenAI, APIError, APITimeoutError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.clients import nim_client, groq_client
from src.config import settings

logger = logging.getLogger(__name__)

RETRYABLE = (APIError, APITimeoutError, RateLimitError)


@dataclass
class CompletionResult:
    content: str
    provider: str
    model: str


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(RETRYABLE),
    reraise=True,
)
def _call(client: OpenAI, model: str, messages: list[dict], temperature: float, max_tokens: int) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def _complete_with_failover(
    messages: list[dict],
    primary_client: OpenAI,
    primary_model: str,
    primary_name: str,
    fallback_client: OpenAI,
    fallback_model: str,
    fallback_name: str,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> CompletionResult:
    try:
        content = _call(primary_client, primary_model, messages, temperature, max_tokens)
        logger.info("served by %s (%s)", primary_name, primary_model)
        return CompletionResult(content=content, provider=primary_name, model=primary_model)
    except RETRYABLE as primary_error:
        logger.warning("%s failed after retry, falling back to %s: %s", primary_name, fallback_name, primary_error)

    try:
        content = _call(fallback_client, fallback_model, messages, temperature, max_tokens)
        logger.info("served by %s (%s)", fallback_name, fallback_model)
        return CompletionResult(content=content, provider=fallback_name, model=fallback_model)
    except RETRYABLE as fallback_error:
        raise RuntimeError(
            f"both {primary_name} and {fallback_name} failed: {fallback_error}"
        ) from fallback_error


def generate_main(messages: list[dict], temperature: float = 0.2, max_tokens: int = 1024) -> CompletionResult:
    return _complete_with_failover(
        messages,
        primary_client=nim_client,
        primary_model=settings.nim_main_model,
        primary_name="nim",
        fallback_client=groq_client,
        fallback_model=settings.groq_main_model,
        fallback_name="groq",
        temperature=temperature,
        max_tokens=max_tokens,
    )


def generate_planner(messages: list[dict], temperature: float = 0.0, max_tokens: int = 256) -> CompletionResult:
    return _complete_with_failover(
        messages,
        primary_client=nim_client,
        primary_model=settings.nim_planner_model,
        primary_name="nim",
        fallback_client=groq_client,
        fallback_model=settings.groq_planner_model,
        fallback_name="groq",
        temperature=temperature,
        max_tokens=max_tokens,
    )