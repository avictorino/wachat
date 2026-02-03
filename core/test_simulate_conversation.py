"""
Tests for the simulate_conversation management command.
"""

from io import StringIO
from unittest.mock import MagicMock, patch

from django.core.management import call_command
from django.test import TestCase

from core.models import Message, Profile


class SimulateConversationCommandTest(TestCase):
    """Tests for the simulate_conversation management command."""

    @patch.dict("os.environ", {
        "TELEGRAM_BOT_TOKEN": "test-token",
        "TELEGRAM_WEBHOOK_SECRET": "test-secret",
        "GROQ_API_KEY": "test-key"
    })
    @patch("services.human_simulator.Groq")
    @patch("services.telegram_service.requests.post")
    @patch("core.views.get_llm_service")
    @patch("core.views.TelegramService")
    def test_command_executes_with_mock_mode(
        self, mock_telegram_service, mock_groq_service, mock_requests, mock_groq_client
    ):
        """Test that the command executes successfully in mock mode."""
        # Setup mocks for GroqService
        mock_groq_instance = MagicMock()
        mock_groq_service.return_value = mock_groq_instance
        mock_groq_instance.infer_gender.return_value = "male"
        mock_groq_instance.generate_welcome_message.return_value = (
            "Olá! Bem-vindo ao espaço de escuta."
        )
        mock_groq_instance.detect_intent.return_value = "ansiedade"
        mock_groq_instance.generate_intent_response.return_value = [
            "Entendo sua preocupação."
        ]
        mock_groq_instance.generate_fallback_response.return_value = [
            "Estou aqui para ouvir."
        ]

        # Setup mocks for TelegramService
        mock_telegram_instance = MagicMock()
        mock_telegram_service.return_value = mock_telegram_instance
        mock_telegram_instance.send_message.return_value = True
        mock_telegram_instance.send_messages.return_value = True

        # Setup mock for requests.post (Telegram API)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True}
        mock_requests.return_value = mock_response

        # Setup mock for Groq client (HumanSimulator)
        mock_groq_client_instance = MagicMock()
        mock_groq_client.return_value = mock_groq_client_instance
        mock_chat_response = MagicMock()
        mock_chat_response.choices = [
            MagicMock(message=MagicMock(content="Oi, tô com dúvidas"))
        ]
        mock_groq_client_instance.chat.completions.create.return_value = (
            mock_chat_response
        )

        out = StringIO()

        # Run the command with minimal turns
        call_command(
            "simulate_conversation",
            "--turns=1",
            "--mock-telegram",
            "--name=Test User",
            "--delay=0.1",
            stdout=out,
        )

        output = out.getvalue()

        # Verify command executed
        self.assertIn("Starting Conversation Simulation", output)
        self.assertIn("Test User", output)

        # Verify a profile was created
        profiles = Profile.objects.all()
        self.assertGreaterEqual(profiles.count(), 1)

        # Verify messages were created (at least welcome message)
        messages = Message.objects.all()
        self.assertGreaterEqual(messages.count(), 1)

    def test_command_arguments_available(self):
        """Test that the command accepts expected arguments."""
        from django.core.management import get_commands, load_command_class

        # Verify command is registered
        commands = get_commands()
        self.assertIn("simulate_conversation", commands)

        # Load the command and check it has the right arguments
        app_name = commands["simulate_conversation"]
        command = load_command_class(app_name, "simulate_conversation")

        # Verify command has a handle method
        self.assertTrue(hasattr(command, "handle"))

        # Verify command has add_arguments method
        self.assertTrue(hasattr(command, "add_arguments"))
