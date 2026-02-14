import os
from typing import Any, Dict, Literal, Optional, Union

from openai import OpenAI

GPT5_MODEL = "gpt-5-mini"
GPT5_TEMPERATURE = 1.0
OPENAI_TIMEOUT_SECONDS = 60
DEFAULT_MAX_COMPLETION_TOKENS = 900


class OpenAIService:
    """OpenAI client wrapper fixed to GPT-5."""

    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required.")

        self.client = OpenAI(api_key=api_key)
        self.default_model = GPT5_MODEL
        self._last_prompt_payload: Optional[Dict[str, Any]] = None

    def basic_call(
        self,
        prompt: Union[str, list],
        max_tokens: int = 100,
        url_type: str = Literal["chat", "generate"],
        num_ctx: int = None,
        system: Optional[str] = None,
    ) -> str:
        selected_model = self.default_model
        resolved_temperature = GPT5_TEMPERATURE
        messages = self._build_messages(prompt=prompt, system=system)
        resolved_max_tokens = self._resolve_max_completion_tokens(
            messages=messages,
            requested_max_tokens=max_tokens,
        )
        request_payload = self._build_request_payload(
            model=selected_model,
            messages=messages,
            temperature=resolved_temperature,
            max_completion_tokens=resolved_max_tokens,
            timeout=OPENAI_TIMEOUT_SECONDS,
        )

        self._last_prompt_payload = {
            "provider": "openai",
            "url_type": url_type,
            "request_params": {
                "model": selected_model,
                "temperature": resolved_temperature,
                "max_tokens": resolved_max_tokens,
                "max_completion_tokens": resolved_max_tokens,
                "num_ctx": num_ctx,
                "system": system,
            },
            "payload": request_payload,
        }

        self._log_request_debug(request_payload, attempt_label="initial")
        response = self.client.chat.completions.create(**request_payload)
        response_text = self._extract_text_response(response)
        self._log_response_debug(response, response_text, attempt_label="initial")

        if response_text.strip():
            return response_text

        raise RuntimeError(self._build_empty_content_error_message(response))

    def _build_messages(self, prompt: Union[str, list], system: Optional[str]) -> list:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})

        if isinstance(prompt, list):
            messages.extend(prompt)
        else:
            messages.append({"role": "user", "content": str(prompt)})
        return messages

    def _resolve_max_completion_tokens(
        self, messages: list, requested_max_tokens: int
    ) -> int:
        if requested_max_tokens and requested_max_tokens > 0:
            return requested_max_tokens
        return DEFAULT_MAX_COMPLETION_TOKENS

    def _build_request_payload(
        self,
        model: str,
        messages: list,
        temperature: float,
        max_completion_tokens: int,
        timeout: int,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
            "timeout": timeout,
        }
        return payload

    def _log_request_debug(self, payload: Dict[str, Any], attempt_label: str) -> None:
        if not self._is_dev_logging_enabled():
            return
        debug_payload = {
            "model": payload.get("model"),
            "temperature": payload.get("temperature"),
            "max_completion_tokens": payload.get("max_completion_tokens"),
            "max_tokens": payload.get("max_tokens"),
            "timeout": payload.get("timeout"),
            "message_roles": [m.get("role") for m in payload.get("messages", [])],
        }
        print(f"[OpenAIService:{attempt_label}] request_params={debug_payload}")

    def _log_response_debug(
        self, response: Any, content: str, attempt_label: str
    ) -> None:
        if not self._is_dev_logging_enabled():
            return
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        if hasattr(completion_details, "model_dump"):
            completion_details = completion_details.model_dump()
        finish_reason = None
        choices = getattr(response, "choices", None) or []
        if choices:
            finish_reason = getattr(choices[0], "finish_reason", None)
        content_preview = (content or "")[:200]
        print(
            f"[OpenAIService:{attempt_label}] usage.prompt_tokens={prompt_tokens} "
            f"usage.completion_tokens={completion_tokens} "
            f"usage.completion_tokens_details={completion_details}"
        )
        print(
            f"[OpenAIService:{attempt_label}] finish_reason={finish_reason} "
            f"content_preview={content_preview!r}"
        )

    def _is_dev_logging_enabled(self) -> bool:
        return os.environ.get("DEBUG", "true").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _build_empty_content_error_message(self, response: Any) -> str:
        choices = getattr(response, "choices", None) or []
        finish_reason = None
        if choices:
            finish_reason = getattr(choices[0], "finish_reason", None)
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        if hasattr(completion_details, "model_dump"):
            completion_details = completion_details.model_dump()
        return (
            "OpenAI returned empty assistant content. "
            f"finish_reason={finish_reason}, "
            f"prompt_tokens={prompt_tokens}, "
            f"completion_tokens={completion_tokens}, "
            f"completion_tokens_details={completion_details}"
        )

    def _extract_text_response(self, response: Any) -> str:
        """Normalize text extraction for Chat Completions responses."""
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""

        message = getattr(choices[0], "message", None)
        if not message:
            return ""

        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            chunks = []
            for part in content:
                text = getattr(part, "text", None)
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
                    continue

                if isinstance(part, dict):
                    dict_text = part.get("text")
                    if isinstance(dict_text, str) and dict_text.strip():
                        chunks.append(dict_text.strip())
            return "\n".join(chunks).strip()

        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str):
            return output_text.strip()

        return ""

    def get_last_prompt_payload(self) -> Optional[Dict[str, Any]]:
        return self._last_prompt_payload
