"""
Factory for creating LLM service instances based on environment configuration.

This module provides a factory function that returns the appropriate LLM service
implementation (GroqService or OllamaService) based on the LLM_PROVIDER environment variable.
"""

import logging
import os

from services.llm_service_interface import LLMServiceInterface

logger = logging.getLogger(__name__)


def get_llm_service() -> LLMServiceInterface:
    """
    Get the configured LLM service instance.

    Reads the LLM_PROVIDER environment variable to determine which service to instantiate.
    Defaults to "groq" if not specified.

    Supported providers:
    - "groq": GroqService (requires GROQ_API_KEY)
    - "ollama": OllamaService (local Ollama server)

    Returns:
        An instance of the configured LLM service

    Raises:
        ValueError: If provider is "groq" but GROQ_API_KEY is not set
        ValueError: If provider is not recognized
    """
    provider = os.environ.get("LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        from services.groq_service import GroqService

        # Validate that GROQ_API_KEY is set
        if not os.environ.get("GROQ_API_KEY"):
            error_msg = (
                "LLM_PROVIDER is set to 'groq' but GROQ_API_KEY environment variable is not set. "
                "Please set GROQ_API_KEY or change LLM_PROVIDER to 'ollama'."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info("Using GroqService as LLM provider")
        return GroqService()

    elif provider == "ollama":
        from services.ollama_service import OllamaService

        logger.info("Using OllamaService as LLM provider")
        return OllamaService()

    else:
        error_msg = (
            f"Unknown LLM_PROVIDER: '{provider}'. "
            f"Supported providers are: 'groq', 'ollama'"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
