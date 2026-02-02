"""
Tests for spiritual comfort and repetition prevention in GroqService.

These tests verify that the system prompt changes ensure:
1. Repetitive emotional acknowledgments are prevented
2. Spiritual comfort is provided when user expresses suffering and asks for it
3. Conversational progression happens after validation
4. Direct questions are answered directly (identity, religion)
5. Grave situations (hunger, risk) trigger objective questions
6. Single message responses (no ||| separator)
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
        self.assertIn("NUNCA REPITA FRASES OU ESTRUTURAS EMOCIONAIS", system_message["content"])
        self.assertIn(
            "Repetição sem progressão é INACEITÁVEL", system_message["content"]
        )
        self.assertIn(
            "Se você já usou validação emocional similar, você está PROIBIDO de repetir",
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
        self.assertIn("presença espiritual gentil", system_message["content"])
        self.assertIn(
            "Você NÃO pode responder a sofrimento + pedido de conforto APENAS com validação emocional",
            system_message["content"],
        )
        self.assertIn(
            "Referência sutil a esperança, cuidado que sustenta", system_message["content"]
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
            "Se você já fez validação/reflexão recentemente, escolha outra opção",
            system_message["content"],
        )
        self.assertIn(
            "Priorize progressão sobre repetição",
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
        self.assertIn("EXEMPLO 3 - Sofrimento + pedido de conforto", system_message["content"])
        self.assertIn("Há uma força que não vem só de nós", system_message["content"])
        self.assertIn("cuidado que nos cerca", system_message["content"])

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    @patch("services.groq_service.Groq")
    @patch("services.groq_service.sanitize_input")
    def test_system_prompt_enforces_single_message_response(
        self, mock_sanitize, mock_groq_client
    ):
        """Test that system prompt enforces one response = one message rule."""
        from services.groq_service import GroqService

        # Setup mocks
        mock_sanitize.return_value = "me dê sugestões de como melhorar meu dia"

        mock_response = Mock()
        mock_response.choices = [Mock()]
        # Single message response without ||| separator
        mock_response.choices[0].message.content = (
            "Isso mostra que você está buscando um caminho. "
            "Talvez começar com pequenos gestos de auto-cuidado, alguns minutos pela manhã, "
            "ou pausas para respirar com mais calma. Posso te acompanhar nisso."
        )

        mock_groq_instance = Mock()
        mock_groq_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_groq_instance

        # Create service and call fallback response
        service = GroqService()
        conversation_context = [
            {"role": "assistant", "content": "Olá! O que te trouxe aqui?"}
        ]

        result = service.generate_fallback_response(
            user_message="me dê sugestões de como melhorar meu dia",
            conversation_context=conversation_context,
            name="João",
            inferred_gender="male",
        )

        # Verify the system prompt enforces single message
        call_args = mock_groq_instance.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        system_message = next(m for m in messages if m["role"] == "system")

        # Check that system prompt contains single message rule
        self.assertIn("UMA RESPOSTA = UMA MENSAGEM", system_message["content"])
        self.assertIn("Nunca quebre uma resposta em várias mensagens curtas", system_message["content"])
        self.assertIn("NÃO use \"|||\" para separar mensagens", system_message["content"])
        self.assertIn("Uma resposta = uma mensagem (NUNCA use \"|||\")", system_message["content"])
        
        # Verify result is a single message (list with one element)
        self.assertEqual(len(result), 1, "Should return a single message")

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    @patch("services.groq_service.Groq")
    @patch("services.groq_service.sanitize_input")
    def test_system_prompt_includes_identity_response_rule(
        self, mock_sanitize, mock_groq_client
    ):
        """Test that system prompt includes direct response for identity questions."""
        from services.groq_service import GroqService

        # Setup mocks
        mock_sanitize.return_value = "Quem é você?"

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = (
            "Sou um assistente criado para ouvir, orientar e ajudar dentro do que for possível por aqui."
        )

        mock_groq_instance = Mock()
        mock_groq_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_groq_instance

        # Create service and call fallback response
        service = GroqService()
        conversation_context = []

        service.generate_fallback_response(
            user_message="Quem é você?",
            conversation_context=conversation_context,
            name="João",
            inferred_gender="male",
        )

        # Verify the system prompt includes identity response rule
        call_args = mock_groq_instance.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        system_message = next(m for m in messages if m["role"] == "system")

        # Check that system prompt contains identity response
        self.assertIn("RESPONDA PERGUNTAS DIRETAS DE FORMA DIRETA", system_message["content"])
        self.assertIn("Se o usuário perguntar quem você é, diga claramente", system_message["content"])
        self.assertIn("Sou um assistente criado para ouvir, orientar e ajudar", system_message["content"])

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    @patch("services.groq_service.Groq")
    @patch("services.groq_service.sanitize_input")
    def test_system_prompt_includes_hunger_situation_rule(
        self, mock_sanitize, mock_groq_client
    ):
        """Test that system prompt includes special handling for hunger/grave situations."""
        from services.groq_service import GroqService

        # Setup mocks
        mock_sanitize.return_value = "Estou com fome e não tenho o que dar para meus filhos"

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = (
            "Entendo a gravidade disso. Você está sem comida agora ou é uma situação recorrente? "
            "Quantas pessoas dependem de você nesse momento?"
        )

        mock_groq_instance = Mock()
        mock_groq_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_groq_instance

        # Create service and call fallback response
        service = GroqService()
        conversation_context = []

        service.generate_fallback_response(
            user_message="Estou com fome e não tenho o que dar para meus filhos",
            conversation_context=conversation_context,
            name="Maria",
            inferred_gender="female",
        )

        # Verify the system prompt includes hunger situation rules
        call_args = mock_groq_instance.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        system_message = next(m for m in messages if m["role"] == "system")

        # Check that system prompt contains grave situation handling
        self.assertIn("DIRETRIZ CRÍTICA PARA SITUAÇÕES GRAVES", system_message["content"])
        self.assertIn("fome ou falta de comida", system_message["content"])
        self.assertIn("Perguntar algo objetivo imediatamente", system_message["content"])
        self.assertIn("Você está sem comida agora ou é uma situação recorrente?", system_message["content"])
        self.assertIn("EVITAR perguntas filosóficas ou abertas demais", system_message["content"])

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    @patch("services.groq_service.Groq")
    @patch("services.groq_service.sanitize_input")
    def test_system_prompt_includes_prohibited_actions(
        self, mock_sanitize, mock_groq_client
    ):
        """Test that system prompt includes clear list of prohibited actions."""
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

        # Verify the system prompt includes prohibited actions
        call_args = mock_groq_instance.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        system_message = next(m for m in messages if m["role"] == "system")

        # Check that system prompt contains prohibited actions section
        self.assertIn("O QUE É PROIBIDO", system_message["content"])
        self.assertIn("Repetir a mesma frase emocional", system_message["content"])
        self.assertIn("Ignorar perguntas diretas", system_message["content"])
        self.assertIn("Enrolar quando o usuário pede ajuda concreta", system_message["content"])
        self.assertIn("Quebrar resposta em múltiplas mensagens", system_message["content"])
