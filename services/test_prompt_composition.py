"""
Tests for the prompt composition system (Theme + Mode).

These tests verify that:
- Themes are layered when appropriate (e.g., addiction theme for addiction-related intents)
- Mode-specific instructions are included
- The base behavioral prompt is NOT included in the composed prompt
  (it's defined in the Modelfile at the project root)
"""

from unittest.mock import Mock, patch

from django.test import TestCase


class PromptCompositionTest(TestCase):
    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    @patch("services.groq_service.Groq")
    @patch("services.groq_service.sanitize_input")
    def test_fallback_prompt_includes_mode_only(
        self, mock_sanitize, mock_groq_client
    ):
        from services.groq_service import GroqService

        mock_sanitize.return_value = "test"

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Resposta"

        mock_groq_instance = Mock()
        mock_groq_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_groq_instance

        service = GroqService()
        service.generate_fallback_response(
            user_message="test",
            conversation_context=[],
            name="João",
            inferred_gender="male",
        )

        call_args = mock_groq_instance.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        system_message = next(m for m in messages if m["role"] == "system")
        content = system_message["content"]

        # Check for mode-specific instructions
        self.assertIn("TAREFA", content)
        self.assertIn("Continue a conversa", content)

        # Verify the base behavioral prompt is NOT included here
        # (it's defined in the Modelfile, not in application code)
        self.assertNotIn("IDENTIDADE CENTRAL", content)
        self.assertNotIn("PRINCÍPIOS DE CONVERSAÇÃO", content)

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    @patch("services.groq_service.Groq")
    @patch("services.groq_service.sanitize_input")
    def test_intent_prompt_layers_addiction_theme(
        self, mock_sanitize, mock_groq_client
    ):
        from services.groq_service import GroqService

        mock_sanitize.return_value = "Estou viciado e com vergonha"

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Resposta"

        mock_groq_instance = Mock()
        mock_groq_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_groq_instance

        service = GroqService()
        service.generate_intent_response(
            user_message="Estou viciado e com vergonha",
            intent="drogas",
            name="Maria",
            inferred_gender="female",
        )

        call_args = mock_groq_instance.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        system_message = next(m for m in messages if m["role"] == "system")
        content = system_message["content"]

        # Check that addiction theme is present
        self.assertIn("TEMA: DROGAS / ÁLCOOL / CIGARRO / VÍCIOS", content)
        self.assertIn("condição real e séria", content.lower())

        # Check for theme-specific soft language requirements
        self.assertIn("O que costuma acontecer", content)
        self.assertIn("Em quais momentos", content)
