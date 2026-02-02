"""
Tests for the core app, including models and views.
"""

from unittest.mock import MagicMock, Mock, patch

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

    def test_profile_nullable_telegram_id(self):
        """Test that telegram_user_id can be null."""
        profile = Profile.objects.create(
            telegram_user_id=None, name="User Without Telegram"
        )
        self.assertIsNone(profile.telegram_user_id)
        self.assertEqual(profile.name, "User Without Telegram")

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

    def test_message_with_explicit_channel(self):
        """Test creating a message with explicit channel."""
        message = Message.objects.create(
            profile=self.profile,
            role="user",
            content="Test from WhatsApp",
            channel="whatsapp",
        )

        self.assertEqual(message.channel, "whatsapp")
        self.assertEqual(message.content, "Test from WhatsApp")

    def test_message_role_choices(self):
        """Test that message role is limited to valid choices."""
        message = Message.objects.create(
            profile=self.profile, role="user", content="Test message"
        )
        self.assertIn(message.role, ["system", "assistant", "user"])

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
