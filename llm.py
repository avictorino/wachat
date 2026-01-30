"""
LLM client implementations for various providers.

This module provides a unified interface for interacting with different
Large Language Model providers (OpenAI, Groq, etc.).
"""
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Optional

from groq import Groq
from openai import OpenAI


@dataclass(frozen=True)
class LLMMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    raw: Optional[Any] = None


class LLMClient(ABC):
    @abstractmethod
    def chat(
        self,
        messages: Iterable[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        raise NotImplementedError


class DummyLLM(LLMClient):
    def chat(
        self,
        messages: Iterable[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        # Troque pelo provider real
        last = messages[-1]["content"] if messages else ""
        return type(
            "R", (), {"text": f"(stub) Recebi: {last}", "raw": {"stub": True}}
        )()


class OpenAILLMClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4.1",
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def chat(
        self,
        messages: Iterable[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[m.__dict__ for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            text=choice.message.content,
            model=response.model,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
            raw=response,
        )


class GroqLLMClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.1-70b-versatile",
    ):
        self.client = Groq(api_key=api_key)
        self.model = model

    def chat(
        self,
        messages: Iterable[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[m.__dict__ for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        choice = response.choices[0]

        # Groq usage object is OpenAI-compatible, but be defensive
        usage = getattr(response, "usage", None)

        return LLMResponse(
            text=choice.message.content,
            model=self.model,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            raw=response.model_dump() if hasattr(response, "model_dump") else None,
        )


def get_llm_client() -> LLMClient:
    """Factory function to create the appropriate LLM client based on environment config."""
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
