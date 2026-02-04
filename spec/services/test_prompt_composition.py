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

        # Check that base prompt structure is present
        self.assertIn("IDENTIDADE E TOM", content)
        self.assertIn("MEMÓRIA E CONTEXTO", content)
        self.assertIn("BASE ESPIRITUAL DA CONVERSA", content)
        self.assertIn("TAREFA", content)
        self.assertIn("Continue a conversa", content)

        # Verify requirements are present
        self.assertIn("Nunca mais de uma pergunta por mensagem", content)
        self.assertIn("NUNCA faça mais de uma pergunta", content)

        # The prompt should not hard-ban scripture
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

        # Check that base prompt structure is present
        self.assertIn("IDENTIDADE E TOM", content)
        self.assertIn("MEMÓRIA E CONTEXTO", content)
        
        # Check that addiction theme is present
        self.assertIn("Tema principal: uso de drogas / dependência química", content)
        self.assertIn("condição real", content.lower())
        
        # Check for guidance in theme
        self.assertIn("Tratar recaídas como parte do processo", content)
