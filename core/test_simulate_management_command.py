"""
Tests for the simulate management command.
"""

from io import StringIO
from unittest.mock import MagicMock, patch

from django.core.management import call_command
from django.test import TestCase

from core.models import Message, Profile


class SimulateManagementCommandTest(TestCase):
    """Tests for the simulate management command."""

    @patch("services.simulation_service.Groq")
    @patch.dict(
        "os.environ",
        {
            "GROQ_API_KEY": "test-groq-key",
        },
    )
    def test_command_executes_successfully(self, mock_groq_client):
        """Test that the command executes successfully with mocked API."""
        # Setup mock for Groq client
        mock_groq_client_instance = MagicMock()
        mock_groq_client.return_value = mock_groq_client_instance

        # Mock generate_simulated_conversation behavior
        mock_chat_response = MagicMock()
        mock_chat_response.choices = [
            MagicMock(message=MagicMock(content="Mensagem simulada"))
        ]
        mock_groq_client_instance.chat.completions.create.return_value = (
            mock_chat_response
        )

        out = StringIO()

        # Run the command with minimal messages
        call_command(
            "simulate",
            "--num-messages=6",
            "--quiet",
            stdout=out,
        )

        output = out.getvalue()

        # Verify command executed
        self.assertIn("Conversa Simulada", output)

        # Verify a profile was created
        profiles = Profile.objects.all()
        self.assertGreaterEqual(profiles.count(), 1)

        # Verify messages were created (should have 6 messages)
        messages = Message.objects.all()
        self.assertGreaterEqual(messages.count(), 6)

    @patch.dict(
        "os.environ",
        {
            "GROQ_API_KEY": "",  # Empty API key - should fail gracefully
        },
    )
    def test_command_without_groq_api_key(self):
        """Test that command handles missing GROQ_API_KEY gracefully."""
        out = StringIO()

        # Run the command
        call_command("simulate", stdout=out)

        output = out.getvalue()

        # Verify error message is displayed
        self.assertIn("GROQ_API_KEY", output)
        self.assertIn("não configurado", output)

    def test_command_arguments_available(self):
        """Test that the command accepts expected arguments."""
        from django.core.management import get_commands, load_command_class

        # Verify command is registered
        commands = get_commands()
        self.assertIn("simulate", commands)

        # Load the command and check it has the right arguments
        app_name = commands["simulate"]
        command = load_command_class(app_name, "simulate")

        # Verify command has a handle method
        self.assertTrue(hasattr(command, "handle"))

        # Verify command has add_arguments method
        self.assertTrue(hasattr(command, "add_arguments"))

    @patch("services.simulation_service.Groq")
    @patch.dict(
        "os.environ",
        {
            "GROQ_API_KEY": "test-groq-key",
        },
    )
    def test_command_validates_num_messages_range(self, mock_groq_client):
        """Test that command validates num_messages is within bounds."""
        # Setup mock
        mock_groq_client_instance = MagicMock()
        mock_groq_client.return_value = mock_groq_client_instance
        mock_chat_response = MagicMock()
        mock_chat_response.choices = [
            MagicMock(message=MagicMock(content="Test"))
        ]
        mock_groq_client_instance.chat.completions.create.return_value = (
            mock_chat_response
        )

        out = StringIO()

        # Run the command with out-of-range value
        call_command("simulate", "--num-messages=15", stdout=out)

        output = out.getvalue()

        # Verify warning is shown
        self.assertIn("must be between 6 and 10", output)

    @patch("services.simulation_service.Groq")
    @patch.dict(
        "os.environ",
        {
            "GROQ_API_KEY": "test-groq-key",
        },
    )
    def test_command_adjusts_odd_num_messages(self, mock_groq_client):
        """Test that command adjusts odd num_messages to even."""
        # Setup mock
        mock_groq_client_instance = MagicMock()
        mock_groq_client.return_value = mock_groq_client_instance
        mock_chat_response = MagicMock()
        mock_chat_response.choices = [
            MagicMock(message=MagicMock(content="Test"))
        ]
        mock_groq_client_instance.chat.completions.create.return_value = (
            mock_chat_response
        )

        out = StringIO()

        # Run the command with odd number
        call_command("simulate", "--num-messages=7", stdout=out)

        output = out.getvalue()

        # Verify adjustment is shown
        self.assertIn("Adjusted to 8 messages", output)
