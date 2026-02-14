import os
from typing import Any, Dict, Optional, Protocol


class LLMService(Protocol):
    def basic_call(
        self,
        prompt,
        model,
        temperature=0.7,
        max_tokens=100,
        url_type="generate",
        timeout=60,
        top_p=None,
        repeat_penalty=None,
        num_ctx=None,
        system=None,
    ) -> str:
        ...

    def get_last_prompt_payload(self) -> Optional[Dict[str, Any]]:
        ...


def get_llm_service(provider: Optional[str] = None) -> LLMService:
    selected_provider = (provider or os.environ.get("LLM_PROVIDER", "openai")).lower()

    if selected_provider == "ollama":
        from services.ollama_service import OllamaService

        return OllamaService()

    if selected_provider == "openai":
        from services.openai_service import OpenAIService

        return OpenAIService()

    raise ValueError(
        f"Unsupported LLM_PROVIDER '{selected_provider}'. Use 'openai' or 'ollama'."
    )
