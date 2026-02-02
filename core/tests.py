"""
Tests for the core app, including models and views.
"""

from unittest.mock import Mock, patch

from django.test import TestCase

from core.models import Message, Profile


class ProfileModelTest(TestCase):
    """Tests for the Profile model."""

    def test_create_profile(self):
        """Test creating a new profile."""
        profile = Profile.objects.create(
            telegram_user_id="12345",
            name="João Silva",
            phone_number="+5511999999999",
            inferred_gender="male",
        )

        self.assertEqual(profile.telegram_user_id, "12345")
        self.assertEqual(profile.name, "João Silva")
        self.assertEqual(profile.phone_number, "+5511999999999")
        self.assertEqual(profile.inferred_gender, "male")
        self.assertIsNotNone(profile.created_at)
        self.assertIsNotNone(profile.updated_at)


    def test_profile_str_representation(self):
        """Test the string representation of Profile."""
        profile = Profile.objects.create(telegram_user_id="12345", name="João Silva")
        self.assertEqual(str(profile), "João Silva (12345)")


class MessageModelTest(TestCase):
    """Tests for the Message model."""

    def setUp(self):
        """Create a test profile for message tests."""
        self.profile = Profile.objects.create(
            telegram_user_id="12345", name="João Silva"
        )

    def test_create_message(self):
        """Test creating a new message."""
        message = Message.objects.create(
            profile=self.profile, role="assistant", content="Olá! Bem-vindo."
        )

        self.assertEqual(message.profile, self.profile)
        self.assertEqual(message.role, "assistant")
        self.assertEqual(message.content, "Olá! Bem-vindo.")
        self.assertEqual(message.channel, "telegram")  # Default channel
        self.assertIsNotNone(message.created_at)

    def test_message_str_representation(self):
        """Test the string representation of Message."""
        message = Message.objects.create(
            profile=self.profile,
            role="assistant",
            content="This is a long message that should be truncated in the string representation",
        )
        str_repr = str(message)
        self.assertTrue(str_repr.startswith("assistant:"))
        self.assertTrue("..." in str_repr)


class TelegramWebhookViewTest(TestCase):
    """Tests for the Telegram webhook view."""

    def setUp(self):
        """Set up test fixtures."""
        self.webhook_url = "/webhooks/telegram/"
        self.webhook_secret = "test-secret"

    @patch.dict("os.environ", {"TELEGRAM_WEBHOOK_SECRET": "test-secret"})
    def test_webhook_requires_secret(self):
        """Test that webhook validates secret token."""
        response = self.client.post(
            self.webhook_url, data="{}", content_type="application/json"
        )
        # Should fail without secret header
        self.assertEqual(response.status_code, 403)

    @patch.dict("os.environ", {"TELEGRAM_WEBHOOK_SECRET": "test-secret"})
    def test_webhook_accepts_valid_secret(self):
        """Test that webhook accepts requests with valid secret."""
        response = self.client.post(
            self.webhook_url,
            data='{"update_id": 1, "message": {}}',
            content_type="application/json",
            HTTP_X_TELEGRAM_BOT_API_SECRET_TOKEN="test-secret",
        )
        self.assertEqual(response.status_code, 200)

    @patch("core.views.TelegramService")
    @patch("core.views.GroqService")
    @patch.dict(
        "os.environ",
        {
            "TELEGRAM_WEBHOOK_SECRET": "test-secret",
            "TELEGRAM_BOT_TOKEN": "test-token",
            "GROQ_API_KEY": "test-key",
        },
    )
    def test_start_command_creates_profile(self, mock_groq, mock_telegram):
        """Test that /start command creates a profile and sends welcome message."""
        # Mock Groq service
        mock_groq_instance = Mock()
        mock_groq_instance.infer_gender.return_value = "male"
        mock_groq_instance.generate_welcome_message.return_value = (
            "Olá João! Bem-vindo."
        )
        mock_groq.return_value = mock_groq_instance

        # Mock Telegram service
        mock_telegram_instance = Mock()
        mock_telegram_instance.send_message.return_value = True
        mock_telegram.return_value = mock_telegram_instance

        # Send /start command
        payload = {
            "update_id": 1,
            "message": {
                "message_id": 1,
                "from": {"id": 12345, "first_name": "João", "last_name": "Silva"},
                "chat": {"id": 12345, "type": "private"},
                "text": "/start",
            },
        }

        response = self.client.post(
            self.webhook_url,
            data=payload,
            content_type="application/json",
            HTTP_X_TELEGRAM_BOT_API_SECRET_TOKEN="test-secret",
        )

        self.assertEqual(response.status_code, 200)

        # Verify profile was created
        profile = Profile.objects.get(telegram_user_id="12345")
        self.assertEqual(profile.name, "João Silva")
        self.assertEqual(profile.inferred_gender, "male")

        # Verify message was persisted
        messages = Message.objects.filter(profile=profile)
        self.assertEqual(messages.count(), 1)
        self.assertEqual(messages.first().role, "assistant")
        self.assertEqual(messages.first().content, "Olá João! Bem-vindo.")

        self.assertEqual(messages.first().channel, "telegram")  # Verify channel is set


        # Verify services were called
        mock_groq_instance.infer_gender.assert_called_once_with("João Silva")
        mock_groq_instance.generate_welcome_message.assert_called_once()
        mock_telegram_instance.send_message.assert_called_once()

    @patch("core.views.TelegramService")
    @patch("core.views.GroqService")
    @patch.dict(
        "os.environ",
        {
            "TELEGRAM_WEBHOOK_SECRET": "test-secret",
            "TELEGRAM_BOT_TOKEN": "test-token",
            "GROQ_API_KEY": "test-key",
        },
    )
    def test_regular_message_detects_intent(self, mock_groq, mock_telegram):
        """Test that regular messages detect intent and generate response."""
        # Create a profile first (simulating a user who already did /start)
        profile = Profile.objects.create(
            telegram_user_id="12345", name="João Silva", inferred_gender="male"
        )

        # Add a welcome message to simulate previous conversation
        Message.objects.create(
            profile=profile, role="assistant", content="Olá João! Bem-vindo."
        )

        # Mock Groq service
        mock_groq_instance = Mock()
        mock_groq_instance.detect_intent.return_value = "ansiedade"
        mock_groq_instance.generate_intent_response.return_value = "Entendo que você está se sentindo ansioso. Quer me contar um pouco mais sobre isso?"
        mock_groq.return_value = mock_groq_instance

        # Mock Telegram service
        mock_telegram_instance = Mock()
        mock_telegram_instance.send_message.return_value = True
        mock_telegram.return_value = mock_telegram_instance

        # Send a regular message
        payload = {
            "update_id": 2,
            "message": {
                "message_id": 2,
                "from": {"id": 12345, "first_name": "João", "last_name": "Silva"},
                "chat": {"id": 12345, "type": "private"},
                "text": "Me sinto muito ansioso ultimamente",
            },
        }

        response = self.client.post(
            self.webhook_url,
            data=payload,
            content_type="application/json",
            HTTP_X_TELEGRAM_BOT_API_SECRET_TOKEN="test-secret",
        )

        self.assertEqual(response.status_code, 200)

        # Verify profile was updated with detected intent
        profile.refresh_from_db()
        self.assertEqual(profile.detected_intent, "ansiedade")

        # Verify messages were persisted (user message + assistant response)
        messages = Message.objects.filter(profile=profile).order_by("created_at")
        self.assertEqual(messages.count(), 3)  # welcome + user + assistant

        # Check user message
        user_message = messages[1]
        self.assertEqual(user_message.role, "user")
        self.assertEqual(user_message.content, "Me sinto muito ansioso ultimamente")

        # Check assistant message
        assistant_message = messages[2]
        self.assertEqual(assistant_message.role, "assistant")
        self.assertIn("ansioso", assistant_message.content.lower())

        # Verify services were called
        mock_groq_instance.detect_intent.assert_called_once_with(
            "Me sinto muito ansioso ultimamente"
        )
        mock_groq_instance.generate_intent_response.assert_called_once()
        mock_telegram_instance.send_message.assert_called_once()

    @patch("core.views.TelegramService")
    @patch("core.views.GroqService")
    @patch.dict(
        "os.environ",
        {
            "TELEGRAM_WEBHOOK_SECRET": "test-secret",
            "TELEGRAM_BOT_TOKEN": "test-token",
            "GROQ_API_KEY": "test-key",
        },
    )
    def test_regular_message_without_profile_creates_one(
        self, mock_groq, mock_telegram
    ):
        """Test that regular message from unknown user creates profile."""
        # Mock Groq service
        mock_groq_instance = Mock()
        mock_groq_instance.detect_intent.return_value = "desabafar"
        mock_groq_instance.generate_intent_response.return_value = (
            "Estou aqui para ouvir. O que você gostaria de compartilhar?"
        )
        mock_groq.return_value = mock_groq_instance

        # Mock Telegram service
        mock_telegram_instance = Mock()
        mock_telegram_instance.send_message.return_value = True
        mock_telegram.return_value = mock_telegram_instance

        # Send message from unknown user
        payload = {
            "update_id": 1,
            "message": {
                "message_id": 1,
                "from": {"id": 99999, "first_name": "Maria"},
                "chat": {"id": 99999, "type": "private"},
                "text": "Preciso desabafar",
            },
        }

        response = self.client.post(
            self.webhook_url,
            data=payload,
            content_type="application/json",
            HTTP_X_TELEGRAM_BOT_API_SECRET_TOKEN="test-secret",
        )

        self.assertEqual(response.status_code, 200)

        # Verify profile was created
        profile = Profile.objects.get(telegram_user_id="99999")
        self.assertEqual(profile.name, "Maria")
        self.assertEqual(profile.detected_intent, "desabafar")

        # Verify messages were created
        messages = Message.objects.filter(profile=profile)
        self.assertEqual(messages.count(), 2)  # user message + assistant response

    @patch("core.views.TelegramService")
    @patch("core.views.GroqService")
    @patch.dict(
        "os.environ",
        {
            "TELEGRAM_WEBHOOK_SECRET": "test-secret",
            "TELEGRAM_BOT_TOKEN": "test-token",
            "GROQ_API_KEY": "test-key",
        },
    )
    def test_subsequent_messages_use_stored_intent(self, mock_groq, mock_telegram):
        """Test that subsequent messages use previously detected intent."""
        # Create a profile with existing intent
        profile = Profile.objects.create(
            telegram_user_id="12345", name="João Silva", detected_intent="ansiedade"
        )

        # Add some conversation history
        Message.objects.create(profile=profile, role="assistant", content="Welcome")
        Message.objects.create(profile=profile, role="user", content="First message")
        Message.objects.create(
            profile=profile, role="assistant", content="First response"
        )

        # Mock Groq service
        mock_groq_instance = Mock()
        mock_groq_instance.generate_intent_response.return_value = (
            "Continue me contando sobre isso."
        )
        mock_groq.return_value = mock_groq_instance

        # Mock Telegram service
        mock_telegram_instance = Mock()
        mock_telegram_instance.send_message.return_value = True
        mock_telegram.return_value = mock_telegram_instance

        # Send another message
        payload = {
            "update_id": 3,
            "message": {
                "message_id": 3,
                "from": {"id": 12345, "first_name": "João"},
                "chat": {"id": 12345, "type": "private"},
                "text": "É difícil dormir à noite",
            },
        }

        response = self.client.post(
            self.webhook_url,
            data=payload,
            content_type="application/json",
            HTTP_X_TELEGRAM_BOT_API_SECRET_TOKEN="test-secret",
        )

        self.assertEqual(response.status_code, 200)

        # Verify intent detection was NOT called (should use stored intent)
        mock_groq_instance.detect_intent.assert_not_called()

        # Verify response generation used stored intent
        call_args = mock_groq_instance.generate_intent_response.call_args
        self.assertEqual(call_args[1]["intent"], "ansiedade")


class GroqServiceTest(TestCase):
    """Tests for the Groq service methods."""

    @patch("services.groq_service.Groq")
    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    def test_detect_intent_financial_problems(self, mock_groq_client):
        """Test intent detection for financial problems."""
        from services.groq_service import GroqService

        # Mock the Groq client response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "problemas_financeiros"

        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_client_instance

        # Test the service
        service = GroqService()
        intent = service.detect_intent("Estou com muitas dívidas e sem emprego")

        self.assertEqual(intent, "problemas_financeiros")
        mock_client_instance.chat.completions.create.assert_called_once()

    @patch("services.groq_service.Groq")
    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    def test_detect_intent_anxiety(self, mock_groq_client):
        """Test intent detection for anxiety."""
        from services.groq_service import GroqService

        # Mock the Groq client response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "ansiedade"

        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_client_instance

        # Test the service
        service = GroqService()
        intent = service.detect_intent("Me sinto muito ansioso e preocupado")

        self.assertEqual(intent, "ansiedade")

    @patch("services.groq_service.Groq")
    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    def test_detect_intent_invalid_returns_outro(self, mock_groq_client):
        """Test that invalid intent returns 'outro'."""
        from services.groq_service import GroqService

        # Mock the Groq client to return invalid intent
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "invalid_intent_xyz"

        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_client_instance

        # Test the service
        service = GroqService()
        intent = service.detect_intent("Some random message")

        self.assertEqual(intent, "outro")

    @patch("services.groq_service.Groq")
    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    def test_detect_intent_error_returns_outro(self, mock_groq_client):
        """Test that API errors return 'outro'."""
        from services.groq_service import GroqService

        # Mock the Groq client to raise an exception
        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.side_effect = Exception(
            "API Error"
        )
        mock_groq_client.return_value = mock_client_instance

        # Test the service
        service = GroqService()
        intent = service.detect_intent("Some message")

        self.assertEqual(intent, "outro")

    @patch("services.groq_service.Groq")
    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    def test_generate_intent_response(self, mock_groq_client):
        """Test generating response based on intent."""
        from services.groq_service import GroqService

        # Mock the Groq client response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = (
            "Entendo o peso que você está carregando. Estou aqui contigo nessa caminhada. Quer me contar um pouco mais sobre o que está sentindo?"
        )

        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_client_instance

        # Test the service
        service = GroqService()
        response = service.generate_intent_response(
            user_message="Me sinto muito ansioso",
            intent="ansiedade",
            name="João",
            inferred_gender="male",
        )

        self.assertIn("Entendo", response)
        self.assertIn("?", response)  # Should end with a question
        mock_client_instance.chat.completions.create.assert_called_once()

    @patch("services.groq_service.Groq")
    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    def test_generate_intent_response_error_returns_fallback(self, mock_groq_client):
        """Test that API errors return fallback message."""
        from services.groq_service import GroqService

        # Mock the Groq client to raise an exception
        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.side_effect = Exception(
            "API Error"
        )
        mock_groq_client.return_value = mock_client_instance

        # Test the service
        service = GroqService()
        response = service.generate_intent_response(
            user_message="Me sinto muito ansioso", intent="ansiedade", name="João"
        )

        # Should return fallback message
        self.assertIn("Obrigado", response)
        self.assertIn("ouvir", response)
