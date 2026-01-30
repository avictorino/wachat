import os

from core.llm.base import LLMClient
from core.llm.groc_llm import GroqLLMClient
from core.llm.openai_llm import OpenAILLMClient


def get_llm_client() -> LLMClient:
    provider = os.getenv("LLM_PROVIDER", "openai")

    if provider == "openai":
        return OpenAILLMClient(
            api_key=os.environ["OPENAI_API_KEY"],
            model=os.getenv("OPENAI_MODEL", "gpt-4.1"),
        )
    if provider == "groq":
        return GroqLLMClient(
            api_key=os.environ["GROQ_API_KEY"],
            model=os.getenv(
                "GROQ_MODEL",
                "llama-3.1-70b-versatile",
            ),
        )

    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
