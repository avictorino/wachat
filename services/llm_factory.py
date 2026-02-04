"""
Factory for creating LLM service instances based on environment configuration.

This module provides a factory function that returns the appropriate LLM service
implementation (OllamaService) based on the LLM_PROVIDER environment variable.
"""

import logging
import os

from services.llm_service_interface import LLMServiceInterface

logger = logging.getLogger(__name__)


def get_llm_service() -> LLMServiceInterface:
    """
    Get the configured LLM service instance.

    Reads the LLM_PROVIDER environment variable to determine which service to instantiate.
    Defaults to "ollama" if not specified.

    Supported providers:
    - "ollama": OllamaService (local Ollama server)

    Returns:
        An instance of the configured LLM service

    Raises:
        ValueError: If provider is not recognized
    """
    provider = os.environ.get("LLM_PROVIDER", "ollama").lower()

    if provider == "ollama":
        from services.ollama_service import OllamaService

        logger.info("Using OllamaService as LLM provider")
        return OllamaService()

    else:
        error_msg = (
            f"Unknown LLM_PROVIDER: '{provider}'. "
            f"Supported providers are: 'ollama'"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
