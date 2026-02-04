"""
Abstract base class interface for LLM services.

This interface defines the contract that all LLM service implementations
(e.g., OllamaService) must follow for consistency.
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class LLMServiceInterface(ABC):
    """Abstract interface for LLM service implementations."""

    @abstractmethod
    def infer_gender(self, name: str) -> str:
        """
        Infer gender from a user's name using LLM.

        Args:
            name: The user's name (first name or full name)

        Returns:
            One of: "male", "female", or "unknown"
        """
        pass

    @abstractmethod
    def generate_welcome_message(
        self, name: str, inferred_gender: Optional[str] = None
    ) -> str:
        """
        Generate a personalized welcome message using LLM.

        Args:
            name: The user's name
            inferred_gender: Inferred gender (male/female/unknown or None)

        Returns:
            The generated welcome message in Brazilian Portuguese
        """
        pass

    @abstractmethod
    def detect_intent(self, user_message: str) -> str:
        """
        Detect and normalize user intent from their message.

        Args:
            user_message: The user's text message

        Returns:
            The detected intent category as a string
        """
        pass

    @abstractmethod
    def approximate_theme(self, user_input: str) -> str:
        """
        Approximate user input to one of the predefined theme categories using LLM.

        Args:
            user_input: The user's theme input (e.g., "enfermidade", "pecado")

        Returns:
            The approximated theme category as a string (one of the valid themes)
        """
        pass

    @abstractmethod
    def generate_intent_response(
        self,
        user_message: str,
        intent: str,
        name: str,
        inferred_gender: Optional[str] = None,
        theme_id: Optional[str] = None,
        conversation_context: Optional[List[dict]] = None,
    ) -> List[str]:
        """
        Generate an empathetic, spiritually-aware response based on detected intent.

        Args:
            user_message: The user's original message
            intent: The detected intent category
            name: The user's name
            inferred_gender: Inferred gender (male/female/unknown or None)
            theme_id: Optional theme identifier
            conversation_context: Optional list of recent messages (dicts with 'role' and 'content')

        Returns:
            List of message strings to send sequentially
        """
        pass

    @abstractmethod
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
        pass
