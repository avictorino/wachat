"""
Integration tests for GroqService with input sanitization.

These tests verify that the sanitization layer is properly applied
before all LLM calls in the GroqService.
"""

from unittest.mock import Mock, patch

from django.test import TestCase


class GroqServiceSanitizationTest(TestCase):
    """Tests to verify GroqService applies input sanitization."""

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    @patch("services.groq_service.Groq")
    @patch("services.groq_service.sanitize_input")
    def test_infer_gender_sanitizes_input(self, mock_sanitize, mock_groq_client):
        """Test that infer_gender sanitizes the name before sending to LLM."""
        from services.groq_service import GroqService

        # Setup mocks
        mock_sanitize.return_value = "[tema sensível] Silva"

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "unknown"

        mock_groq_instance = Mock()
        mock_groq_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_groq_instance

        # Create service and call method
        service = GroqService()
        service.infer_gender("Sexo Silva")

        # Verify sanitization was called
        mock_sanitize.assert_called_once_with("Sexo Silva")

        # Verify the sanitized name was used in the prompt
        call_args = mock_groq_instance.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        user_message = next(m for m in messages if m["role"] == "user")
        self.assertIn("[tema sensível] Silva", user_message["content"])
        self.assertNotIn("Sexo", user_message["content"])

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    @patch("services.groq_service.Groq")
    @patch("services.groq_service.sanitize_input")
    def test_generate_welcome_message_sanitizes_input(
        self, mock_sanitize, mock_groq_client
    ):
        """Test that generate_welcome_message sanitizes the name before sending to LLM."""
        from services.groq_service import GroqService

        # Setup mocks
        mock_sanitize.return_value = "[tema sensível] Santos"

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Olá! Bem-vindo ao nosso espaço."

        mock_groq_instance = Mock()
        mock_groq_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_groq_instance

        # Create service and call method
        service = GroqService()
        service.generate_welcome_message("Morte Santos", inferred_gender="male")

        # Verify sanitization was called
        mock_sanitize.assert_called_once_with("Morte Santos")

        # Verify the sanitized name was used in the prompt
        call_args = mock_groq_instance.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        user_message = next(m for m in messages if m["role"] == "user")
        self.assertIn("[tema sensível] Santos", user_message["content"])
        self.assertNotIn("Morte", user_message["content"])

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    @patch("services.groq_service.Groq")
    @patch("services.groq_service.sanitize_input")
    def test_clean_names_pass_through_unchanged(self, mock_sanitize, mock_groq_client):
        """Test that clean names without harmful content pass through unchanged."""
        from services.groq_service import GroqService

        # Setup mocks - sanitize returns the same clean name
        mock_sanitize.return_value = "João Silva"

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "male"

        mock_groq_instance = Mock()
        mock_groq_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_groq_instance

        # Create service and call method
        service = GroqService()
        service.infer_gender("João Silva")

        # Verify sanitization was called
        mock_sanitize.assert_called_once_with("João Silva")

        # Verify the clean name was preserved in the prompt
        call_args = mock_groq_instance.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        user_message = next(m for m in messages if m["role"] == "user")
        self.assertIn("João Silva", user_message["content"])
