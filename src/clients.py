from openai import OpenAI
from google import genai
from qdrant_client import QdrantClient

from src.config import settings

# module-level singletons: constructed once on first import, reused across
# every Streamlit rerun in the same process (module import cache handles this).
#
# timeout=30 + max_retries=0: without an explicit timeout the SDK default is
# 10 minutes, so a slow/stuck provider hangs the whole turn instead of failing
# fast into the Groq fallback. max_retries=0 disables the SDK's own internal
# retry so only src/llm.py's retry-then-failover logic is in control, instead
# of two retry layers compounding unpredictably.

nim_client = OpenAI(
    api_key=settings.nvidia_nim_api_key,
    base_url=settings.nim_base_url,
    timeout=30.0,
    max_retries=0,
)

groq_client = OpenAI(
    api_key=settings.groq_api_key,
    base_url=settings.groq_base_url,
    timeout=30.0,
    max_retries=0,
)

gemini_client = genai.Client(api_key=settings.gemini_api_key)

qdrant_client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
)