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
    @patch("core.views.get_llm_service")
    @patch.dict(
        "os.environ",
        {
            "TELEGRAM_WEBHOOK_SECRET": "test-secret",
            "LLM_PROVIDER": "ollama",
        },
    )
    def test_simulate_command_creates_profile_and_conversation(
        self, mock_llm_service, mock_telegram_service, mock_simulation_service
    ):
        """Test that /simulate command creates a profile and generates a conversation."""
        # Setup mocks
        mock_telegram_instance = MagicMock()
        mock_telegram_service.return_value = mock_telegram_instance
        mock_telegram_instance.send_message.return_value = True

        mock_simulation_instance = MagicMock()
        mock_simulation_service.return_value = mock_simulation_instance

        mock_llm_instance = MagicMock()
        mock_llm_service.return_value = mock_llm_instance

        # Create a fake profile with default theme "desabafar"
        fake_profile = Profile.objects.create(
            name="Simulation_123456789", inferred_gender="unknown"
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

        # Send /simulate command (without theme, should use default "desabafar")
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
        mock_simulation_service.assert_called_once_with()

        # Verify LLM approximation was NOT called (no theme provided)
        mock_llm_instance.approximate_theme.assert_not_called()

        # Verify simulation methods were called with theme "desabafar"
        mock_simulation_instance.create_simulation_profile.assert_called_once_with("desabafar")
        mock_simulation_instance.generate_simulated_conversation.assert_called_once_with(
            fake_profile, 8, "desabafar"
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
        },
    )
    def test_simulate_command_without_llm_provider(self, mock_telegram_service):
        """Test that /simulate command handles missing LLM_PROVIDER gracefully."""
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
            "LLM_PROVIDER": "ollama",
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
    @patch("core.views.get_llm_service")
    @patch.dict(
        "os.environ",
        {
            "TELEGRAM_WEBHOOK_SECRET": "test-secret",
            "LLM_PROVIDER": "ollama",
        },
    )
    def test_simulate_command_with_theme_parameter(
        self, mock_llm_service, mock_telegram_service, mock_simulation_service
    ):
        """Test that /simulate command accepts theme parameter and uses LLM approximation."""
        # Setup mocks
        mock_telegram_instance = MagicMock()
        mock_telegram_service.return_value = mock_telegram_instance
        mock_telegram_instance.send_message.return_value = True

        mock_simulation_instance = MagicMock()
        mock_simulation_service.return_value = mock_simulation_instance

        mock_llm_instance = MagicMock()
        mock_llm_service.return_value = mock_llm_instance
        # Mock LLM approximation - "doenca" stays as "doenca"
        mock_llm_instance.approximate_theme.return_value = "doenca"

        # Create a fake profile with "doenca" theme
        fake_profile = Profile.objects.create(
            name="Simulation_123456789", inferred_gender="unknown"
        )
        mock_simulation_instance.create_simulation_profile.return_value = fake_profile

        # Create fake conversation
        fake_conversation = [
            {"role": "ROLE_A", "content": "Não tô me sentindo bem ultimamente..."},
            {"role": "ROLE_B", "content": "Quer me contar mais sobre isso?"},
            {"role": "ROLE_A", "content": "É difícil explicar..."},
            {"role": "ROLE_B", "content": "Estou aqui, sem pressa."},
            {"role": "ROLE_A", "content": "Talvez seja só cansaço."},
            {"role": "ROLE_B", "content": "Como você se sente com esse cansaço?"},
        ]
        mock_simulation_instance.generate_simulated_conversation.return_value = (
            fake_conversation
        )

        # Create fake analysis
        fake_analysis = "A conversa abordou preocupações de saúde de forma cautelosa."
        mock_simulation_instance.analyze_conversation_emotions.return_value = fake_analysis

        # Send /simulate doenca command
        payload = {
            "update_id": 1,
            "message": {
                "message_id": 1,
                "from": {"id": 12345, "first_name": "Test"},
                "chat": {"id": 12345, "type": "private"},
                "text": "/simulate doenca",
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

        # Verify LLM approximation was called
        mock_llm_instance.approximate_theme.assert_called_once_with("doenca")

        # Verify create_simulation_profile was called with "doenca" theme
        mock_simulation_instance.create_simulation_profile.assert_called_once_with("doenca")
        
        # Verify generate_simulated_conversation was called with theme "doenca"
        mock_simulation_instance.generate_simulated_conversation.assert_called_once_with(
            fake_profile, 8, "doenca"
        )

    @patch("core.views.get_llm_service")
    @patch("core.views.TelegramService")
    @patch.dict(
        "os.environ",
        {
            "TELEGRAM_WEBHOOK_SECRET": "test-secret",
            "LLM_PROVIDER": "ollama",
        },
    )
    def test_simulate_command_with_invalid_theme(self, mock_telegram_service, mock_llm_service):
        """Test that /simulate command handles themes that LLM can't map."""
        # Setup mock
        mock_telegram_instance = MagicMock()
        mock_telegram_service.return_value = mock_telegram_instance
        mock_telegram_instance.send_message.return_value = True

        mock_llm_instance = MagicMock()
        mock_llm_service.return_value = mock_llm_instance
        # Mock LLM approximation returning "outro" (couldn't map)
        mock_llm_instance.approximate_theme.return_value = "outro"

        # Send /simulate invalid_theme command
        payload = {
            "update_id": 1,
            "message": {
                "message_id": 1,
                "from": {"id": 12345, "first_name": "Test"},
                "chat": {"id": 12345, "type": "private"},
                "text": "/simulate invalid_theme",
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

        # Verify LLM approximation was called
        mock_llm_instance.approximate_theme.assert_called_once_with("invalid_theme")

        # Verify error message was sent
        self.assertGreaterEqual(mock_telegram_instance.send_message.call_count, 1)
        first_call_args = mock_telegram_instance.send_message.call_args_list[0]
        self.assertIn("identificar", first_call_args[0][1])  # "Não consegui identificar"
        self.assertIn("doenca", first_call_args[0][1])  # Should list valid theme examples

    @patch("core.views.SimulationService")
    @patch("core.views.TelegramService")
    @patch("core.views.get_llm_service")
    @patch.dict(
        "os.environ",
        {
            "TELEGRAM_WEBHOOK_SECRET": "test-secret",
            "LLM_PROVIDER": "ollama",
        },
    )
    def test_simulate_command_with_theme_alias(
        self, mock_llm_service, mock_telegram_service, mock_simulation_service
    ):
        """Test that /simulate command uses LLM to map theme aliases (e.g., pecado -> ato_criminoso_pecado)."""
        # Setup mocks
        mock_telegram_instance = MagicMock()
        mock_telegram_service.return_value = mock_telegram_instance
        mock_telegram_instance.send_message.return_value = True

        mock_simulation_instance = MagicMock()
        mock_simulation_service.return_value = mock_simulation_instance

        mock_llm_instance = MagicMock()
        mock_llm_service.return_value = mock_llm_instance
        # Mock LLM approximation - "pecado" maps to "ato_criminoso_pecado"
        mock_llm_instance.approximate_theme.return_value = "ato_criminoso_pecado"

        # Create a fake profile
        fake_profile = Profile.objects.create(
            name="Simulation_123456789", inferred_gender="unknown"
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

        # Send /simulate pecado command (should be approximated by LLM to ato_criminoso_pecado)
        payload = {
            "update_id": 1,
            "message": {
                "message_id": 1,
                "from": {"id": 12345, "first_name": "Test"},
                "chat": {"id": 12345, "type": "private"},
                "text": "/simulate pecado",
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

        # Verify LLM approximation was called
        mock_llm_instance.approximate_theme.assert_called_once_with("pecado")

        # Verify create_simulation_profile was called with "ato_criminoso_pecado" (mapped from "pecado")
        mock_simulation_instance.create_simulation_profile.assert_called_once_with("ato_criminoso_pecado")
        
        # Verify generate_simulated_conversation was called with mapped theme
        mock_simulation_instance.generate_simulated_conversation.assert_called_once_with(
            fake_profile, 8, "ato_criminoso_pecado"
        )

    @patch("core.views.SimulationService")
    @patch("core.views.TelegramService")
    @patch("core.views.get_llm_service")
    @patch.dict(
        "os.environ",
        {
            "TELEGRAM_WEBHOOK_SECRET": "test-secret",
            "LLM_PROVIDER": "ollama",
        },
    )
    def test_simulate_command_with_enfermidade_approximation(
        self, mock_llm_service, mock_telegram_service, mock_simulation_service
    ):
        """Test that /simulate command uses LLM to approximate 'enfermidade' to 'doenca'."""
        # Setup mocks
        mock_telegram_instance = MagicMock()
        mock_telegram_service.return_value = mock_telegram_instance
        mock_telegram_instance.send_message.return_value = True

        mock_simulation_instance = MagicMock()
        mock_simulation_service.return_value = mock_simulation_instance

        mock_llm_instance = MagicMock()
        mock_llm_service.return_value = mock_llm_instance
        # Mock LLM approximation - "enfermidade" maps to "doenca"
        mock_llm_instance.approximate_theme.return_value = "doenca"

        # Create a fake profile
        fake_profile = Profile.objects.create(
            name="Simulation_123456789", inferred_gender="unknown"
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

        # Send /simulate enfermidade command (should be approximated by LLM to doenca)
        payload = {
            "update_id": 1,
            "message": {
                "message_id": 1,
                "from": {"id": 12345, "first_name": "Test"},
                "chat": {"id": 12345, "type": "private"},
                "text": "/simulate enfermidade",
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

        # Verify LLM approximation was called with "enfermidade"
        mock_llm_instance.approximate_theme.assert_called_once_with("enfermidade")

        # Verify create_simulation_profile was called with "doenca" (approximated from "enfermidade")
        mock_simulation_instance.create_simulation_profile.assert_called_once_with("doenca")
        
        # Verify generate_simulated_conversation was called with mapped theme
        mock_simulation_instance.generate_simulated_conversation.assert_called_once_with(
            fake_profile, 8, "doenca"
        )

