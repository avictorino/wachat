"""
Tests for the /simulate Telegram command.
"""

from unittest.mock import MagicMock, patch

from django.test import TestCase

from core.models import Message, Profile


class SimulateCommandTest(TestCase):
    """Tests for the /simulate command."""

    @patch("core.views.SimulationService")
    @patch("core.views.TelegramService")
    @patch.dict(
        "os.environ",
        {
            "TELEGRAM_WEBHOOK_SECRET": "test-secret",
            "GROQ_API_KEY": "test-groq-key",
        },
    )
    def test_simulate_command_creates_profile_and_conversation(
        self, mock_telegram_service, mock_simulation_service
    ):
        """Test that /simulate command creates a profile and generates a conversation."""
        # Setup mocks
        mock_telegram_instance = MagicMock()
        mock_telegram_service.return_value = mock_telegram_instance
        mock_telegram_instance.send_message.return_value = True

        mock_simulation_instance = MagicMock()
        mock_simulation_service.return_value = mock_simulation_instance

        # Create a fake profile
        fake_profile = Profile.objects.create(
            name="Simulation_123456789", inferred_gender="unknown", detected_intent="simulation"
        )
        mock_simulation_instance.create_simulation_profile.return_value = fake_profile

        # Create fake conversation
        fake_conversation = [
            {"role": "ROLE_A", "content": "Estou me sentindo perdido..."},
            {"role": "ROLE_B", "content": "Entendo sua busca. Estou aqui para ouvir."},
            {"role": "ROLE_A", "content": "Não sei se tenho mais fé."},
            {"role": "ROLE_B", "content": "A fé pode ser um caminho de perguntas."},
        ]
        mock_simulation_instance.generate_simulated_conversation.return_value = (
            fake_conversation
        )

        # Create fake analysis
        fake_analysis = "Esta conversa refletiu uma busca espiritual genuína."
        mock_simulation_instance.analyze_conversation_emotions.return_value = fake_analysis

        # Send /simulate command
        payload = {
            "update_id": 1,
            "message": {
                "message_id": 1,
                "from": {"id": 12345, "first_name": "Test"},
                "chat": {"id": 12345, "type": "private"},
                "text": "/simulate",
            },
        }

        response = self.client.post(
            "/webhooks/telegram/",
            data=payload,
            content_type="application/json",
            HTTP_X_TELEGRAM_BOT_API_SECRET_TOKEN="test-secret",
        )

        # Verify response
        self.assertEqual(response.status_code, 200)

        # Verify SimulationService was initialized
        mock_simulation_service.assert_called_once_with("test-groq-key")

        # Verify simulation methods were called
        mock_simulation_instance.create_simulation_profile.assert_called_once()
        mock_simulation_instance.generate_simulated_conversation.assert_called_once_with(
            fake_profile, 8
        )
        mock_simulation_instance.analyze_conversation_emotions.assert_called_once_with(
            fake_conversation
        )

        # Verify messages were sent to Telegram
        # Should send: init message + 4 conversation messages + 1 analysis message = 6 total
        self.assertEqual(mock_telegram_instance.send_message.call_count, 6)

    @patch("core.views.TelegramService")
    @patch.dict(
        "os.environ",
        {
            "TELEGRAM_WEBHOOK_SECRET": "test-secret",
            "GROQ_API_KEY": "",  # Empty API key - should fail gracefully
        },
    )
    def test_simulate_command_without_groq_api_key(self, mock_telegram_service):
        """Test that /simulate command handles missing GROQ_API_KEY gracefully."""
        # Setup mock
        mock_telegram_instance = MagicMock()
        mock_telegram_service.return_value = mock_telegram_instance
        mock_telegram_instance.send_message.return_value = True

        # Send /simulate command
        payload = {
            "update_id": 1,
            "message": {
                "message_id": 1,
                "from": {"id": 12345, "first_name": "Test"},
                "chat": {"id": 12345, "type": "private"},
                "text": "/simulate",
            },
        }

        response = self.client.post(
            "/webhooks/telegram/",
            data=payload,
            content_type="application/json",
            HTTP_X_TELEGRAM_BOT_API_SECRET_TOKEN="test-secret",
        )

        # Verify response
        self.assertEqual(response.status_code, 200)

        # Verify error message was sent (should be first call)
        self.assertGreaterEqual(mock_telegram_instance.send_message.call_count, 1)
        first_call_args = mock_telegram_instance.send_message.call_args_list[0]
        self.assertIn("not available", first_call_args[0][1])

    @patch.dict(
        "os.environ",
        {
            "TELEGRAM_WEBHOOK_SECRET": "test-secret",
            "GROQ_API_KEY": "test-groq-key",
        },
    )
    def test_simulate_command_accepted_by_webhook(self):
        """Test that /simulate command is recognized by the webhook."""
        payload = {
            "update_id": 1,
            "message": {
                "message_id": 1,
                "from": {"id": 12345, "first_name": "Test"},
                "chat": {"id": 12345, "type": "private"},
                "text": "/simulate",
            },
        }

        with patch("core.views.TelegramService"), patch("core.views.SimulationService"):
            response = self.client.post(
                "/webhooks/telegram/",
                data=payload,
                content_type="application/json",
                HTTP_X_TELEGRAM_BOT_API_SECRET_TOKEN="test-secret",
            )

            # Verify response is successful
            self.assertEqual(response.status_code, 200)

    @patch("core.views.SimulationService")
    @patch("core.views.TelegramService")
    @patch.dict(
        "os.environ",
        {
            "TELEGRAM_WEBHOOK_SECRET": "test-secret",
            "GROQ_API_KEY": "test-groq-key",
        },
    )
    def test_simulate_command_with_valid_parameter(
        self, mock_telegram_service, mock_simulation_service
    ):
        """Test that /simulate command accepts valid num_messages parameter."""
        # Setup mocks
        mock_telegram_instance = MagicMock()
        mock_telegram_service.return_value = mock_telegram_instance
        mock_telegram_instance.send_message.return_value = True

        mock_simulation_instance = MagicMock()
        mock_simulation_service.return_value = mock_simulation_instance

        # Create a fake profile
        fake_profile = Profile.objects.create(
            name="Simulation_123456789", inferred_gender="unknown", detected_intent="simulation"
        )
        mock_simulation_instance.create_simulation_profile.return_value = fake_profile

        # Create fake conversation
        fake_conversation = [
            {"role": "ROLE_A", "content": "Message 1"},
            {"role": "ROLE_B", "content": "Message 2"},
            {"role": "ROLE_A", "content": "Message 3"},
            {"role": "ROLE_B", "content": "Message 4"},
            {"role": "ROLE_A", "content": "Message 5"},
            {"role": "ROLE_B", "content": "Message 6"},
        ]
        mock_simulation_instance.generate_simulated_conversation.return_value = (
            fake_conversation
        )

        # Create fake analysis
        fake_analysis = "Analysis of 6-message conversation."
        mock_simulation_instance.analyze_conversation_emotions.return_value = fake_analysis

        # Send /simulate 6 command
        payload = {
            "update_id": 1,
            "message": {
                "message_id": 1,
                "from": {"id": 12345, "first_name": "Test"},
                "chat": {"id": 12345, "type": "private"},
                "text": "/simulate 6",
            },
        }

        response = self.client.post(
            "/webhooks/telegram/",
            data=payload,
            content_type="application/json",
            HTTP_X_TELEGRAM_BOT_API_SECRET_TOKEN="test-secret",
        )

        # Verify response
        self.assertEqual(response.status_code, 200)

        # Verify generate_simulated_conversation was called with 6
        mock_simulation_instance.generate_simulated_conversation.assert_called_once_with(
            fake_profile, 6
        )

    @patch("core.views.TelegramService")
    @patch.dict(
        "os.environ",
        {
            "TELEGRAM_WEBHOOK_SECRET": "test-secret",
            "GROQ_API_KEY": "test-groq-key",
        },
    )
    def test_simulate_command_with_invalid_parameter_too_low(self, mock_telegram_service):
        """Test that /simulate command rejects num_messages < 6."""
        # Setup mock
        mock_telegram_instance = MagicMock()
        mock_telegram_service.return_value = mock_telegram_instance
        mock_telegram_instance.send_message.return_value = True

        # Send /simulate 4 command (invalid, too low)
        payload = {
            "update_id": 1,
            "message": {
                "message_id": 1,
                "from": {"id": 12345, "first_name": "Test"},
                "chat": {"id": 12345, "type": "private"},
                "text": "/simulate 4",
            },
        }

        response = self.client.post(
            "/webhooks/telegram/",
            data=payload,
            content_type="application/json",
            HTTP_X_TELEGRAM_BOT_API_SECRET_TOKEN="test-secret",
        )

        # Verify response
        self.assertEqual(response.status_code, 200)

        # Verify error message was sent
        self.assertGreaterEqual(mock_telegram_instance.send_message.call_count, 1)
        first_call_args = mock_telegram_instance.send_message.call_args_list[0]
        self.assertIn("inválido", first_call_args[0][1])
        self.assertIn("6 e 10", first_call_args[0][1])

    @patch("core.views.TelegramService")
    @patch.dict(
        "os.environ",
        {
            "TELEGRAM_WEBHOOK_SECRET": "test-secret",
            "GROQ_API_KEY": "test-groq-key",
        },
    )
    def test_simulate_command_with_invalid_parameter_too_high(self, mock_telegram_service):
        """Test that /simulate command rejects num_messages > 10."""
        # Setup mock
        mock_telegram_instance = MagicMock()
        mock_telegram_service.return_value = mock_telegram_instance
        mock_telegram_instance.send_message.return_value = True

        # Send /simulate 12 command (invalid, too high)
        payload = {
            "update_id": 1,
            "message": {
                "message_id": 1,
                "from": {"id": 12345, "first_name": "Test"},
                "chat": {"id": 12345, "type": "private"},
                "text": "/simulate 12",
            },
        }

        response = self.client.post(
            "/webhooks/telegram/",
            data=payload,
            content_type="application/json",
            HTTP_X_TELEGRAM_BOT_API_SECRET_TOKEN="test-secret",
        )

        # Verify response
        self.assertEqual(response.status_code, 200)

        # Verify error message was sent
        self.assertGreaterEqual(mock_telegram_instance.send_message.call_count, 1)
        first_call_args = mock_telegram_instance.send_message.call_args_list[0]
        self.assertIn("inválido", first_call_args[0][1])
        self.assertIn("6 e 10", first_call_args[0][1])

    @patch("core.views.SimulationService")
    @patch("core.views.TelegramService")
    @patch.dict(
        "os.environ",
        {
            "TELEGRAM_WEBHOOK_SECRET": "test-secret",
            "GROQ_API_KEY": "test-groq-key",
        },
    )
    def test_simulate_command_with_invalid_parameter_non_numeric(
        self, mock_telegram_service, mock_simulation_service
    ):
        """Test that /simulate command handles non-numeric parameters gracefully."""
        # Setup mocks
        mock_telegram_instance = MagicMock()
        mock_telegram_service.return_value = mock_telegram_instance
        mock_telegram_instance.send_message.return_value = True

        mock_simulation_instance = MagicMock()
        mock_simulation_service.return_value = mock_simulation_instance

        # Create a fake profile
        fake_profile = Profile.objects.create(
            name="Simulation_123456789", inferred_gender="unknown", detected_intent="simulation"
        )
        mock_simulation_instance.create_simulation_profile.return_value = fake_profile

        # Create fake conversation
        fake_conversation = [
            {"role": "ROLE_A", "content": "Message 1"},
            {"role": "ROLE_B", "content": "Message 2"},
        ]
        mock_simulation_instance.generate_simulated_conversation.return_value = (
            fake_conversation
        )

        # Create fake analysis
        fake_analysis = "Analysis"
        mock_simulation_instance.analyze_conversation_emotions.return_value = fake_analysis

        # Send /simulate abc command (non-numeric, should use default)
        payload = {
            "update_id": 1,
            "message": {
                "message_id": 1,
                "from": {"id": 12345, "first_name": "Test"},
                "chat": {"id": 12345, "type": "private"},
                "text": "/simulate abc",
            },
        }

        response = self.client.post(
            "/webhooks/telegram/",
            data=payload,
            content_type="application/json",
            HTTP_X_TELEGRAM_BOT_API_SECRET_TOKEN="test-secret",
        )

        # Verify response
        self.assertEqual(response.status_code, 200)

        # Verify generate_simulated_conversation was called with default value (8)
        mock_simulation_instance.generate_simulated_conversation.assert_called_once_with(
            fake_profile, 8
        )

