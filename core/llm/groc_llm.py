from typing import Iterable, Optional

from groq import Groq

from biblical_friend.llm.base import LLMClient, LLMMessage, LLMResponse


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