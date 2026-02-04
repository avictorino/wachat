"""
Tests for LLM service factory.

This module tests the factory function that creates LLM service instances
based on the LLM_PROVIDER environment variable.
"""

from unittest.mock import patch

from django.test import TestCase

from services.llm_factory import get_llm_service
from services.ollama_service import OllamaService


class LLMFactoryTest(TestCase):
    """Tests for the LLM service factory."""

    @patch.dict("os.environ", {"LLM_PROVIDER": "ollama"})
    def test_factory_returns_ollama_service_when_provider_is_ollama(self):
        """Test that factory returns OllamaService when LLM_PROVIDER=ollama."""
        service = get_llm_service()
        self.assertIsInstance(service, OllamaService)

    @patch.dict("os.environ", {}, clear=True)
    def test_factory_defaults_to_ollama_when_provider_not_set(self):
        """Test that factory defaults to OllamaService when LLM_PROVIDER is not set."""
        service = get_llm_service()
        self.assertIsInstance(service, OllamaService)

    @patch.dict("os.environ", {"LLM_PROVIDER": "unknown"})
    def test_factory_raises_error_for_unknown_provider(self):
        """Test that factory raises ValueError for unknown provider."""
        with self.assertRaises(ValueError) as context:
            get_llm_service()
        
        self.assertIn("Unknown LLM_PROVIDER", str(context.exception))
        self.assertIn("unknown", str(context.exception))

    @patch.dict("os.environ", {"LLM_PROVIDER": "Ollama"})
    def test_factory_is_case_insensitive_for_ollama(self):
        """Test that factory handles ollama provider name case-insensitively."""
        service = get_llm_service()
        self.assertIsInstance(service, OllamaService)
