"""
Ollama LLM service for AI-powered features.

This module handles interactions with a local Ollama server for:
- Gender inference from names
- Welcome message generation
- Context-aware fallback responses
"""

import logging
import os
from typing import Any, Dict, List, Optional

import requests

from services.input_sanitizer import sanitize_input
from services.prompt.llm_service_interface import LLMServiceInterface
from services.prompt import PromptComposer
from services.prompt.prompts import (
    GENDER_INFERENCE_PROMPT,
    WELCOME_MESSAGE_PROMPT,
    INTENT_DETECTION_PROMPT,
    THEME_APPROXIMATION_PROMPT,
    VALID_THEMES,
    build_gender_inference_user_prompt,
    build_welcome_message_user_prompt,
    build_intent_detection_user_prompt,
    build_theme_approximation_user_prompt,
)

logger = logging.getLogger(__name__)


class OllamaService(LLMServiceInterface):
    """Service class for interacting with local Ollama LLM API."""

    def __init__(self):
        """Initialize Ollama client with configuration from environment."""
        self.base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.environ.get("OLLAMA_MODEL", "llama3.1")
        self.api_url = f"{self.base_url}/api/chat"

        logger.info(
            f"Initialized OllamaService with base_url={self.base_url}, model={self.model}"
        )

    def _make_chat_request(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """
        Make a chat completion request to Ollama API.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate (optional, no enforcement)
            **kwargs: Additional Ollama options (top_p, repeat_penalty, etc.)

        Returns:
            The assistant's response text

        Raises:
            requests.exceptions.RequestException: On connection or API errors
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                **kwargs,
            },
        }

        # Add max_tokens if provided (Ollama calls it num_predict)
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=60,  # 60 second timeout for local requests
            )
            response.raise_for_status()

            response_data = response.json()
            content = response_data.get("message", {}).get("content", "").strip()

            if not content:
                logger.warning("Ollama returned empty content")
                raise ValueError("Empty response from Ollama")

            return content

        except requests.exceptions.ConnectionError as e:
            logger.error(f"Failed to connect to Ollama at {self.base_url}: {str(e)}")
            raise
        except requests.exceptions.Timeout as e:
            logger.error(f"Ollama request timed out: {str(e)}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {str(e)}")
            raise
        except (KeyError, ValueError) as e:
            logger.error(f"Failed to parse Ollama response: {str(e)}")
            raise

    def infer_gender(self, name: str) -> str:
        """
        Infer gender from a user's name using Ollama LLM.

        This is a soft, probabilistic inference based solely on the name.
        The result is for internal use only and should never be explicitly
        stated to the user.

        Args:
            name: The user's name (first name or full name)

        Returns:
            One of: "male", "female", or "unknown"
        """
        try:
            # Sanitize input before sending to LLM
            sanitized_name = sanitize_input(name)

            user_prompt = build_gender_inference_user_prompt(sanitized_name)

            messages = [
                {"role": "system", "content": GENDER_INFERENCE_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            response_text = self._make_chat_request(
                messages, temperature=0.3, max_tokens=10
            )

            inferred = response_text.strip().lower()

            # Validate response
            if inferred not in ["male", "female", "unknown"]:
                logger.warning(f"Unexpected gender inference result: {inferred}")
                return "unknown"

            logger.info(f"Gender inferred for name '{name}': {inferred}")
            return inferred

        except Exception as e:
            logger.error(f"Error inferring gender: {str(e)}", exc_info=True)
            return "unknown"

    def generate_welcome_message(
        self, name: str, inferred_gender: Optional[str] = None
    ) -> str:
        """
        Generate a personalized welcome message using Ollama LLM.

        The message is warm, human, and inviting without being cliché.
        It adapts subtly based on the user's name and inferred gender,
        and always ends with an open question.

        Args:
            name: The user's name
            inferred_gender: Inferred gender (male/female/unknown or None)

        Returns:
            The generated welcome message in Brazilian Portuguese
        """
        try:
            # Sanitize input before sending to LLM
            sanitized_name = sanitize_input(name)

            user_prompt = build_welcome_message_user_prompt(
                sanitized_name, inferred_gender
            )

            messages = [
                {"role": "system", "content": WELCOME_MESSAGE_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            welcome_message = self._make_chat_request(
                messages, temperature=0.8, max_tokens=300
            )

            logger.info(f"Generated welcome message for '{name}'")
            return welcome_message

        except Exception as e:
            logger.error(f"Error generating welcome message: {str(e)}", exc_info=True)
            # Fallback to a simple message if API fails
            return f"Olá, {name}. Este é um espaço seguro de escuta espiritual. O que te trouxe aqui hoje?"

    def detect_intent(self, user_message: str) -> str:
        """
        Detect and normalize user intent from their message.

        Maps the user's message to one of the predefined intent categories.

        Args:
            user_message: The user's text message

        Returns:
            The detected intent category as a string
        """
        try:
            user_prompt = build_intent_detection_user_prompt(user_message)

            messages = [
                {"role": "system", "content": INTENT_DETECTION_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            response_text = self._make_chat_request(
                messages, temperature=0.3, max_tokens=20
            )

            intent = response_text.strip().lower()

            logger.info(f"Intent detected: {intent}")
            return intent

        except Exception as e:
            logger.error(f"Error detecting intent: {str(e)}", exc_info=True)
            return "outro"

    def approximate_theme(self, user_input: str) -> str:
        """
        Approximate user input to one of the predefined theme categories using LLM.

        This method takes a user's theme input (which may be in natural language,
        synonyms, or variations) and maps it to the closest predefined theme category.

        Args:
            user_input: The user's theme input (e.g., "enfermidade", "pecado")

        Returns:
            The approximated theme category as a string (one of the valid themes)
        """
        try:
            # Sanitize input before sending to LLM
            sanitized_input = sanitize_input(user_input)

            user_prompt = build_theme_approximation_user_prompt(sanitized_input)

            messages = [
                {"role": "system", "content": THEME_APPROXIMATION_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            response_text = self._make_chat_request(
                messages, temperature=0.2, max_tokens=20
            )

            theme = response_text.strip().lower()

            # Validate and normalize the response
            if theme not in VALID_THEMES:
                logger.warning(
                    f"Unexpected theme approximated: {theme}, defaulting to 'outro'"
                )
                theme = "outro"

            logger.info(f"Theme approximated: '{user_input}' -> '{theme}'")
            return theme

        except Exception as e:
            logger.error(f"Error approximating theme: {str(e)}", exc_info=True)
            # Default to "outro" on error
            return "outro"

    def generate_intent_response(
        self,
        user_message: str,
        intent: str,
        name: str,
        inferred_gender: Optional[str] = None,
        theme_id: Optional[str] = None,
    ) -> List[str]:
        """
        Generate an empathetic, spiritually-aware response based on detected intent.

        Args:
            user_message: The user's original message
            intent: The detected intent category
            name: The user's name
            inferred_gender: Inferred gender (male/female/unknown or None)
            theme_id: Optional theme identifier

        Returns:
            List of message strings to send sequentially
        """
        try:
            # Sanitize input before sending to LLM
            sanitized_message = sanitize_input(user_message)

            if not theme_id:
                # Default theme selection based on the detected intent.
                from services.theme_selector import select_theme_from_intent_and_message

                selection = select_theme_from_intent_and_message(
                    intent=intent,
                    message_text=sanitized_message,
                    existing_theme_id=None,
                )
                theme_id = selection.theme_id

            gender_context = ""
            if inferred_gender and inferred_gender != "unknown":
                gender_context = f"\nGênero inferido (use APENAS para ajustar sutilmente o tom, NUNCA mencione explicitamente): {inferred_gender}"

            system_prompt = PromptComposer.compose_system_prompt(
                theme_id=theme_id, mode="intent_response"
            )

            user_prompt = (
                f"Nome da pessoa: {name}"
                f"\nIntenção detectada: {intent}"
                f"\nMensagem dela: {sanitized_message}{gender_context}"
                "\n\nResponda agora."
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response_text = self._make_chat_request(
                messages, temperature=0.85, max_tokens=250
            )

            # Split response into multiple messages if separator is used
            messages = self._split_response_messages(response_text)

            logger.info(
                f"Generated intent-based response with {len(messages)} message(s) for intent: {intent} (theme_id={theme_id})"
            )
            return messages

        except Exception as e:
            logger.error(f"Error generating intent response: {str(e)}", exc_info=True)
            # Fallback to a simple empathetic message with a follow-up question
            return [
                "Obrigado por compartilhar isso comigo. O que mais te incomoda agora?"
            ]

    def generate_fallback_response(
        self,
        user_message: str,
        conversation_context: List[dict],
        name: str,
        inferred_gender: Optional[str] = None,
        theme_id: Optional[str] = None,
    ) -> List[str]:
        """
        Generate a context-aware fallback response when intent is unclear.

        Args:
            user_message: The user's current message
            conversation_context: List of recent messages (dicts with 'role' and 'content')
            name: The user's name
            inferred_gender: Inferred gender (male/female/unknown or None)
            theme_id: Optional theme identifier

        Returns:
            List of message strings to send sequentially
        """
        try:
            # Sanitize input before sending to LLM
            sanitized_message = sanitize_input(user_message)

            # Build conversation context for the LLM
            context_messages = []
            for msg in conversation_context:
                context_messages.append({"role": msg["role"], "content": msg["content"]})

            gender_context = ""
            if inferred_gender and inferred_gender != "unknown":
                gender_context = f"\nGênero inferido (use APENAS para ajustar sutilmente o tom, NUNCA mencione explicitamente): {inferred_gender}"

            system_prompt = PromptComposer.compose_system_prompt(
                theme_id=theme_id, mode="fallback_response"
            )
            system_prompt += (
                "\n"
                "CONTEXTO FIXO\n"
                f"- Nome da pessoa: {name}{gender_context}\n"
                "- Esta é uma continuação natural da conversa.\n"
            )

            # Add the current user message to context
            context_messages.append({"role": "user", "content": sanitized_message})

            # Prepend system message
            all_messages = [{"role": "system", "content": system_prompt}] + context_messages

            response_text = self._make_chat_request(
                all_messages, temperature=0.85, max_tokens=350
            )

            # Split response into multiple messages if separator is used
            messages = self._split_response_messages(response_text)

            logger.info(
                f"Generated fallback response with {len(messages)} message(s) for ambiguous intent"
            )
            return messages

        except Exception as e:
            logger.error(f"Error generating fallback response: {str(e)}", exc_info=True)
            # Fallback to a simple empathetic message with a follow-up question
            return [
                "Obrigado por compartilhar isso comigo. Como você está se sentindo agora?"
            ]

    def _split_response_messages(self, response: str) -> List[str]:
        """
        Split a response into multiple messages if separator is present.

        Args:
            response: The generated response, possibly with ||| separators

        Returns:
            List of message strings
        """
        # Split by triple pipe separator
        messages = [msg.strip() for msg in response.split("|||")]

        # Filter out empty messages
        messages = [msg for msg in messages if msg]

        # Ensure we have at least one message
        if not messages:
            messages = [response]

        # Limit to 3 messages maximum for safety
        messages = messages[:3]

        return messages
