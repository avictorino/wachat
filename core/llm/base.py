from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Optional


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
