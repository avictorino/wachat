import os
from typing import Any, Dict, Literal, Optional, Union
from urllib.parse import urljoin

import requests


class OllamaService:
    """Provider implementation for local Ollama API."""

    def __init__(self):
        self.base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.api_url_base = f"{self.base_url}/api/"
        self.default_model = os.environ.get("OLLAMA_CHAT_MODEL", "llama3:8b")
        self._last_prompt_payload: Optional[Dict[str, Any]] = None

    def basic_call(
        self,
        prompt: Union[str, list],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 100,
        url_type: str = Literal["chat", "generate"],
        timeout: int = 60,
        top_p: float = None,
        repeat_penalty: float = None,
        num_ctx: int = None,
        system: Optional[str] = None,
    ) -> str:
        endpoint = urljoin(self.api_url_base, url_type)
        selected_model = model or self.default_model

        payload: Dict[str, Any] = {
            "model": selected_model,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if url_type == "chat":
            if isinstance(prompt, list):
                messages = prompt
            else:
                messages = [{"role": "user", "content": str(prompt)}]
            if system:
                messages = [{"role": "system", "content": system}] + messages
            payload["messages"] = messages
        else:
            payload["prompt"] = prompt
            if system:
                payload["system"] = system

        if top_p is not None:
            payload["options"]["top_p"] = top_p
        if repeat_penalty is not None:
            payload["options"]["repeat_penalty"] = repeat_penalty
        if num_ctx is not None:
            payload["options"]["num_ctx"] = num_ctx

        self._last_prompt_payload = {
            "provider": "ollama",
            "endpoint": endpoint,
            "url_type": url_type,
            "timeout": timeout,
            "request_params": {
                "model": selected_model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "repeat_penalty": repeat_penalty,
                "num_ctx": num_ctx,
                "system": system,
            },
            "payload": payload,
        }

        response = requests.post(endpoint, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()

        if url_type == "chat":
            return (data.get("message", {}) or {}).get("content", "").strip()
        return data.get("response", "").strip()

    def get_last_prompt_payload(self) -> Optional[Dict[str, Any]]:
        return self._last_prompt_payload
