from typing import Iterable, Optional

from openai import OpenAI

from biblical_friend.llm.base import LLMClient, LLMMessage, LLMResponse


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