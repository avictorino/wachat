"""
Tests for the prompt composition system (Base + Theme).

These tests verify that:
- The shared base prompt is always present
- The addiction theme is layered when intent indicates addiction
- The new prompts allow optional scripture (no hard ban on verses)
"""

from unittest.mock import Mock, patch

from django.test import TestCase


class PromptCompositionTest(TestCase):
    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    @patch("services.groq_service.Groq")
    @patch("services.groq_service.sanitize_input")
    def test_fallback_prompt_includes_base_and_mode(
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

        self.assertIn("OBJETIVO", content)
        self.assertIn("TOM E RITMO", content)
        self.assertIn("ESCRITURAS (OPCIONAL)", content)
        self.assertIn("TAREFA", content)
        self.assertIn("Continue a conversa", content)

        # The old monolithic prompt hard-banned scripture; the new base prompt should not.
        self.assertNotIn("NUNCA cite versículos", content)

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

        self.assertIn("OBJETIVO", content)
        self.assertIn("TEMA: DROGAS / ÁLCOOL / CIGARRO / VÍCIOS", content)
        self.assertIn("condição real e séria", content.lower())
