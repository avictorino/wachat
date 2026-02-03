"""
Groq LLM service for AI-powered features.

This module handles interactions with the Groq API for:
- Gender inference from names
- Welcome message generation
- Context-aware fallback responses
"""

import logging
import os
from typing import List, Optional

from groq import Groq

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


class GroqService(LLMServiceInterface):
    """Service class for interacting with Groq LLM API."""

    def __init__(self):
        """Initialize Groq client with API key from environment."""
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            logger.error("GROQ_API_KEY environment variable is required")
            raise ValueError("GROQ_API_KEY environment variable is required")

        self.client = Groq(api_key=api_key)
        self.model = (
            "llama-3.3-70b-versatile"  # Using a capable model for nuanced tasks
        )

    def infer_gender(self, name: str) -> str:
        """
        Infer gender from a user's name using Groq LLM.

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

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": GENDER_INFERENCE_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,  # Low temperature for more deterministic results
                max_tokens=10,
            )

            inferred = response.choices[0].message.content.strip().lower()

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
        Generate a personalized welcome message using Groq LLM.

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

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": WELCOME_MESSAGE_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.8,  # Higher temperature for more creative, varied responses
                max_tokens=300,
            )

            welcome_message = response.choices[0].message.content.strip()

            logger.info(f"Generated welcome message for '{name}'")
            return welcome_message

        except Exception as e:
            logger.error(f"Error generating welcome message: {str(e)}", exc_info=True)
            # Fallback to a simple message if API fails
            return f"Olá, {name}. Este é um espaço seguro de escuta espiritual. O que te trouxe aqui hoje?"

    def detect_intent(self, user_message: str) -> str:
        """
        Detect and normalize user intent from their message.

        Maps the user's message to one of the predefined intent categories:
        1. Problemas financeiros
        2. Distante da religião/espiritualidade
        3. Ato criminoso ou pecado
        4. Doença (própria ou familiar)
        5. Ansiedade
        6. Desabafar
        7. Viu nas redes sociais
        8. Addiction-related conditions (drogas, alcool, sexo, cigarro)
        9. Outro (for unmatched cases)

        Args:
            user_message: The user's text message

        Returns:
            The detected intent category as a string
        """
        try:
            user_prompt = build_intent_detection_user_prompt(user_message)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": INTENT_DETECTION_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,  # Low temperature for more deterministic classification
                max_tokens=20,
            )

            intent = response.choices[0].message.content.strip().lower()

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

        For example:
        - "enfermidade" -> "doenca"
        - "problemas de dinheiro" -> "problemas_financeiros"
        - "pecado" -> "ato_criminoso_pecado"

        Args:
            user_input: The user's theme input (e.g., "enfermidade", "pecado")

        Returns:
            The approximated theme category as a string (one of the valid themes)
        """
        try:
            # Sanitize input before sending to LLM
            sanitized_input = sanitize_input(user_input)

            user_prompt = build_theme_approximation_user_prompt(sanitized_input)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": THEME_APPROXIMATION_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,  # Low temperature for more deterministic classification
                max_tokens=20,
            )

            theme = response.choices[0].message.content.strip().lower()

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

        The response:
        - Acknowledges the user's situation
        - Validates feelings without reinforcing despair (implicitly)
        - Includes subtle spiritual undertones (optional)
        - May or may not end with a question (optional)
        - Is warm, calm, and non-judgmental
        - Avoids preaching, sermons, or explicit religious content
        - Uses micro-responses (1-3 sentences)
        - Can be split into multiple messages for natural flow

        Args:
            user_message: The user's original message
            intent: The detected intent category
            name: The user's name
            inferred_gender: Inferred gender (male/female/unknown or None)

        Returns:
            List of message strings to send sequentially
        """
        try:
            # Sanitize input before sending to LLM
            sanitized_message = sanitize_input(user_message)

            if not theme_id:
                # Default theme selection based on the detected intent.
                # The caller (e.g., views) may override/persist themes explicitly.
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

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": INTENT_DETECTION_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.85,  # Higher temperature for more natural, varied responses
                max_tokens=250,  # Reduced from 400 to enforce brevity
            )

            generated_response = response.choices[0].message.content.strip()

            # Split response into multiple messages if separator is used
            messages = self._split_response_messages(generated_response)

            logger.info(
                f"Generated intent-based response with {len(messages)} message(s) for intent: {intent} (theme_id={theme_id})"
            )
            return messages

        except Exception as e:
            logger.error(f"Error generating intent response: {str(e)}", exc_info=True)
            # Fallback to a simple empathetic message with a follow-up question
            return ["Obrigado por compartilhar isso comigo. O que mais te incomoda agora?"]

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

        This method is used when the user's message doesn't clearly match
        any predefined intent category. It maintains conversational continuity
        by using recent conversation history and a script-driven approach.

        The response follows a flexible 4-part structure when the user's motivation is clear:
        1. Brief acknowledgment (no clichés, one short sentence)
        2. Gentle initiatives (2-3 max, invitations not commands)
        3. Light reflection (shared observation, not advice)
        4. Open invitation (optional, not a direct question)

        The response:
        - Acknowledges the user's message respectfully
        - Reflects the intention behind the message
        - Avoids labeling the user's state
        - Avoids religious authority language
        - Uses soft, pastoral, human tone
        - May or may not include a question (not forced)
        - Returns 1-3 short messages to feel natural

        Args:
            user_message: The user's current message
            conversation_context: List of recent messages (dicts with 'role' and 'content')
            name: The user's name
            inferred_gender: Inferred gender (male/female/unknown or None)

        Returns:
            List of message strings to send sequentially
        """
        try:
            # Sanitize input before sending to LLM
            sanitized_message = sanitize_input(user_message)

            # Build conversation context for the LLM
            context_messages = []
            for msg in conversation_context:
                context_messages.append(
                    {"role": msg["role"], "content": msg["content"]}
                )

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

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system_prompt}]
                + context_messages,
                temperature=0.85,  # Higher than 0.8 for intent responses, for more natural conversation
                max_tokens=350,  # Reduced from 500 to enforce brevity
            )

            generated_response = response.choices[0].message.content.strip()

            # Split response into multiple messages if separator is used
            messages = self._split_response_messages(generated_response)

            logger.info(
                f"Generated fallback response with {len(messages)} message(s) for ambiguous intent"
            )
            return messages

        except Exception as e:
            logger.error(f"Error generating fallback response: {str(e)}", exc_info=True)
            # Fallback to a simple empathetic message with a follow-up question
            return ["Obrigado por compartilhar isso comigo. Como você está se sentindo agora?"]

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
