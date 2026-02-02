"""
Tests for spiritual comfort and repetition prevention in GroqService.

These tests verify that the system prompt changes ensure:
1. Repetitive emotional acknowledgments are prevented
2. Spiritual comfort is provided when user expresses suffering and asks for it
3. Conversational progression happens after validation
"""

from unittest.mock import Mock, patch

from django.test import TestCase


class GroqServiceSpiritualComfortTest(TestCase):
    """Tests to verify spiritual comfort and anti-repetition features."""

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    @patch("services.groq_service.Groq")
    @patch("services.groq_service.sanitize_input")
    def test_system_prompt_contains_repetition_prevention_rules(
        self, mock_sanitize, mock_groq_client
    ):
        """Test that system prompt includes rules to prevent repetitive validations."""
        from services.groq_service import GroqService

        # Setup mocks
        mock_sanitize.return_value = "test message"

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = (
            "Há uma força que não vem só de nós."
        )

        mock_groq_instance = Mock()
        mock_groq_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_groq_instance

        # Create service and call fallback response
        service = GroqService()
        conversation_context = [
            {"role": "assistant", "content": "Olá! Bem-vindo."},
            {"role": "user", "content": "Meu pai está doente"},
            {"role": "assistant", "content": "Isso é pesado."},
        ]

        service.generate_fallback_response(
            user_message="test message",
            conversation_context=conversation_context,
            name="João",
            inferred_gender="male",
        )

        # Verify the system prompt was called
        call_args = mock_groq_instance.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        system_message = next(m for m in messages if m["role"] == "system")

        # Check that system prompt contains repetition prevention rules
        self.assertIn("PROIBIÇÃO DE REPETIÇÃO", system_message["content"])
        self.assertIn(
            "Repetição sem progressão é INACEITÁVEL", system_message["content"]
        )
        self.assertIn(
            "NUNCA repetida se foi usada nas últimas 1-2 mensagens",
            system_message["content"],
        )

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    @patch("services.groq_service.Groq")
    @patch("services.groq_service.sanitize_input")
    def test_system_prompt_contains_spiritual_comfort_requirements(
        self, mock_sanitize, mock_groq_client
    ):
        """Test that system prompt includes mandatory spiritual comfort for suffering scenarios."""
        from services.groq_service import GroqService

        # Setup mocks
        mock_sanitize.return_value = "preciso de força"

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[
            0
        ].message.content = "Há uma força que não vem só de nós. Às vezes vem do cuidado que nos cerca."

        mock_groq_instance = Mock()
        mock_groq_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_groq_instance

        # Create service and call fallback response with suffering context
        service = GroqService()
        conversation_context = [
            {"role": "assistant", "content": "O que te trouxe aqui?"},
            {"role": "user", "content": "Meu pai está muito doente"},
            {"role": "assistant", "content": "Isso pesa."},
        ]

        service.generate_fallback_response(
            user_message="preciso de força",
            conversation_context=conversation_context,
            name="Maria",
            inferred_gender="female",
        )

        # Verify the system prompt includes spiritual comfort rules
        call_args = mock_groq_instance.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        system_message = next(m for m in messages if m["role"] == "system")

        # Check that system prompt contains spiritual comfort requirements
        self.assertIn("SOFRIMENTO + PEDIDO DE CONFORTO", system_message["content"])
        self.assertIn("Presença espiritual gentil", system_message["content"])
        self.assertIn(
            "Você NÃO pode responder a sofrimento + pedido de conforto APENAS com validação emocional",
            system_message["content"],
        )
        self.assertIn(
            "presença espiritual é obrigatória", system_message["content"]
        )

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    @patch("services.groq_service.Groq")
    @patch("services.groq_service.sanitize_input")
    def test_system_prompt_includes_conversation_progression_rules(
        self, mock_sanitize, mock_groq_client
    ):
        """Test that system prompt includes rules for conversational progression."""
        from services.groq_service import GroqService

        # Setup mocks
        mock_sanitize.return_value = "sim, é difícil"

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Há um cuidado que te sustenta."

        mock_groq_instance = Mock()
        mock_groq_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_groq_instance

        # Create service and call fallback response
        service = GroqService()
        conversation_context = [
            {"role": "user", "content": "Estou com medo"},
            {"role": "assistant", "content": "Dá para sentir o tamanho disso."},
        ]

        service.generate_fallback_response(
            user_message="sim, é difícil",
            conversation_context=conversation_context,
            name="Pedro",
            inferred_gender="male",
        )

        # Verify the system prompt includes progression rules
        call_args = mock_groq_instance.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        system_message = next(m for m in messages if m["role"] == "system")

        # Check that system prompt contains progression requirements
        self.assertIn(
            "Você DEVE introduzir uma nova função conversacional",
            system_message["content"],
        )
        self.assertIn(
            "Se você já fez validação/reflexão nas últimas 1-2 mensagens, escolha uma das outras opções",
            system_message["content"],
        )
        self.assertIn(
            "Progresso conversacional é mais importante que validação repetida",
            system_message["content"],
        )

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    @patch("services.groq_service.Groq")
    @patch("services.groq_service.sanitize_input")
    def test_system_prompt_includes_concrete_examples(
        self, mock_sanitize, mock_groq_client
    ):
        """Test that system prompt includes concrete examples of spiritual comfort."""
        from services.groq_service import GroqService

        # Setup mocks
        mock_sanitize.return_value = "test"

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"

        mock_groq_instance = Mock()
        mock_groq_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_groq_instance

        # Create service and call fallback response
        service = GroqService()
        conversation_context = []

        service.generate_fallback_response(
            user_message="test",
            conversation_context=conversation_context,
            name="Test",
            inferred_gender=None,
        )

        # Verify the system prompt includes concrete examples
        call_args = mock_groq_instance.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        system_message = next(m for m in messages if m["role"] == "system")

        # Check that system prompt contains the spiritual comfort example
        self.assertIn("EXEMPLO DE SOFRIMENTO + PEDIDO DE CONFORTO", system_message["content"])
        self.assertIn("Há uma força que não vem só de nós", system_message["content"])
        self.assertIn("cuidado que nos cerca", system_message["content"])
