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

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    @patch("services.groq_service.Groq")
    @patch("services.groq_service.sanitize_input")
    def test_approximate_theme_sanitizes_input(self, mock_sanitize, mock_groq_client):
        """Test that approximate_theme sanitizes the input before sending to LLM."""
        from services.groq_service import GroqService

        # Setup mocks
        mock_sanitize.return_value = "[tema sensível]"

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "outro"

        mock_groq_instance = Mock()
        mock_groq_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_groq_instance

        # Create service and call method
        service = GroqService()
        result = service.approximate_theme("morte")

        # Verify sanitization was called
        mock_sanitize.assert_called_once_with("morte")

        # Verify the sanitized input was used in the prompt
        call_args = mock_groq_instance.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        user_message = next(m for m in messages if m["role"] == "user")
        self.assertIn("[tema sensível]", user_message["content"])
        self.assertNotIn("morte", user_message["content"])

        # Verify result is "outro" (couldn't map sanitized harmful content)
        self.assertEqual(result, "outro")

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    @patch("services.groq_service.Groq")
    @patch("services.groq_service.sanitize_input")
    def test_approximate_theme_maps_enfermidade_to_doenca(
        self, mock_sanitize, mock_groq_client
    ):
        """Test that approximate_theme maps 'enfermidade' to 'doenca'."""
        from services.groq_service import GroqService

        # Setup mocks - clean input passes through
        mock_sanitize.return_value = "enfermidade"

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "doenca"

        mock_groq_instance = Mock()
        mock_groq_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_groq_instance

        # Create service and call method
        service = GroqService()
        result = service.approximate_theme("enfermidade")

        # Verify sanitization was called
        mock_sanitize.assert_called_once_with("enfermidade")

        # Verify the input was used in the prompt
        call_args = mock_groq_instance.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        user_message = next(m for m in messages if m["role"] == "user")
        self.assertIn("enfermidade", user_message["content"])

        # Verify result is "doenca"
        self.assertEqual(result, "doenca")

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    @patch("services.groq_service.Groq")
    @patch("services.groq_service.sanitize_input")
    def test_approximate_theme_maps_pecado_to_ato_criminoso_pecado(
        self, mock_sanitize, mock_groq_client
    ):
        """Test that approximate_theme maps 'pecado' to 'ato_criminoso_pecado'."""
        from services.groq_service import GroqService

        # Setup mocks
        mock_sanitize.return_value = "pecado"

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "ato_criminoso_pecado"

        mock_groq_instance = Mock()
        mock_groq_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_groq_instance

        # Create service and call method
        service = GroqService()
        result = service.approximate_theme("pecado")

        # Verify result is "ato_criminoso_pecado"
        self.assertEqual(result, "ato_criminoso_pecado")

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    @patch("services.groq_service.Groq")
    @patch("services.groq_service.sanitize_input")
    def test_approximate_theme_defaults_to_outro_on_invalid_response(
        self, mock_sanitize, mock_groq_client
    ):
        """Test that approximate_theme defaults to 'outro' when LLM returns invalid response."""
        from services.groq_service import GroqService

        # Setup mocks
        mock_sanitize.return_value = "tema_desconhecido"

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "tema_invalido"  # Not in valid list

        mock_groq_instance = Mock()
        mock_groq_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_groq_instance

        # Create service and call method
        service = GroqService()
        result = service.approximate_theme("tema_desconhecido")

        # Verify result defaults to "outro"
        self.assertEqual(result, "outro")

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    @patch("services.groq_service.Groq")
    @patch("services.groq_service.sanitize_input")
    def test_approximate_theme_handles_api_error(
        self, mock_sanitize, mock_groq_client
    ):
        """Test that approximate_theme handles API errors gracefully."""
        from services.groq_service import GroqService

        # Setup mocks
        mock_sanitize.return_value = "teste"

        mock_groq_instance = Mock()
        mock_groq_instance.chat.completions.create.side_effect = Exception("API Error")
        mock_groq_client.return_value = mock_groq_instance

        # Create service and call method
        service = GroqService()
        result = service.approximate_theme("teste")

        # Verify result defaults to "outro" on error
        self.assertEqual(result, "outro")
