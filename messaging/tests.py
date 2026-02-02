from unittest.mock import MagicMock, patch

from django.test import TestCase

from messaging.types import OutgoingMessage
from messaging.providers import (
    TelegramAdapter,
    ProviderDetector,
    NormalizedMessage,
)
from service.telegram import TelegramProvider


class TelegramAdapterTest(TestCase):
    def setUp(self):
        self.adapter = TelegramAdapter()

    def test_can_handle_telegram_webhook(self):
        """Test that Telegram webhooks are correctly identified"""
        # Valid Telegram webhook
        telegram_payload = {
            "update_id": 123456789,
            "message": {
                "message_id": 123,
                "from": {"id": 987654321, "first_name": "John"},
                "chat": {"id": 123456789, "type": "private"},
                "date": 1234567890,
                "text": "Hello",
            },
        }
        self.assertTrue(self.adapter.can_handle({}, telegram_payload))

        # Invalid payload - no update_id
        self.assertFalse(self.adapter.can_handle({}, {}))

        # Invalid payload - no message
        invalid_payload = {"update_id": 123}
        self.assertFalse(self.adapter.can_handle({}, invalid_payload))

    def test_normalize_text_message(self):
        """Test normalizing a text message"""
        payload = {
            "update_id": 123456789,
            "message": {
                "message_id": 123,
                "from": {"id": 987654321, "first_name": "John"},
                "chat": {"id": 123456789, "type": "private"},
                "date": 1234567890,
                "text": "Hello World",
            },
        }

        result = self.adapter.normalize({}, payload)

        self.assertIsInstance(result, NormalizedMessage)
        self.assertEqual(result.sender_id, "987654321")
        self.assertEqual(result.recipient_id, "123456789")
        self.assertEqual(result.message_body, "Hello World")
        self.assertEqual(result.message_type, "text")
        self.assertEqual(result.provider, "telegram")
        self.assertFalse(result.reply_as_audio)

    def test_normalize_voice_message(self):
        """Test normalizing a voice message"""
        payload = {
            "update_id": 123456789,
            "message": {
                "message_id": 123,
                "from": {"id": 987654321, "first_name": "John"},
                "chat": {"id": 123456789, "type": "private"},
                "date": 1234567890,
                "voice": {"file_id": "voice123", "duration": 10},
            },
        }

        result = self.adapter.normalize({}, payload)

        self.assertIsInstance(result, NormalizedMessage)
        self.assertEqual(result.message_body, "[Audio message received]")
        self.assertEqual(result.message_type, "audio")
        self.assertEqual(result.media_url, "voice123")
        self.assertTrue(result.reply_as_audio)

    def test_normalize_photo_message(self):
        """Test normalizing a photo message"""
        payload = {
            "update_id": 123456789,
            "message": {
                "message_id": 123,
                "from": {"id": 987654321, "first_name": "John"},
                "chat": {"id": 123456789, "type": "private"},
                "date": 1234567890,
                "photo": [
                    {"file_id": "photo123_small", "width": 320},
                    {"file_id": "photo123_large", "width": 1280},
                ],
                "caption": "Beautiful sunset",
            },
        }

        result = self.adapter.normalize({}, payload)

        self.assertEqual(result.message_type, "image")
        self.assertEqual(result.message_body, "Beautiful sunset")
        self.assertEqual(result.media_url, "photo123_large")

    def test_normalize_edited_message(self):
        """Test normalizing an edited message"""
        payload = {
            "update_id": 123456789,
            "edited_message": {
                "message_id": 123,
                "from": {"id": 987654321, "first_name": "John"},
                "chat": {"id": 123456789, "type": "private"},
                "date": 1234567890,
                "edit_date": 1234567900,
                "text": "Edited text",
            },
        }

        result = self.adapter.normalize({}, payload)

        self.assertIsInstance(result, NormalizedMessage)
        self.assertEqual(result.message_body, "Edited text")


class ProviderDetectorTest(TestCase):
    def test_detect_telegram(self):
        """Test that ProviderDetector correctly identifies Telegram"""
        detector = ProviderDetector()

        telegram_payload = {
            "update_id": 123456789,
            "message": {
                "message_id": 123,
                "from": {"id": 987654321, "first_name": "John"},
                "chat": {"id": 123456789, "type": "private"},
                "date": 1234567890,
                "text": "Hello",
            },
        }

        provider, message = detector.detect_and_normalize({}, telegram_payload)

        self.assertEqual(provider, "telegram")
        self.assertIsInstance(message, NormalizedMessage)
        self.assertEqual(message.sender_id, "987654321")


class TelegramProviderTest(TestCase):
    @patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "test_token"})
    def test_from_settings(self):
        """Test creating provider from environment variables"""
        provider = TelegramProvider.from_settings()

        self.assertIsInstance(provider, TelegramProvider)
        self.assertEqual(provider.token, "test_token")
        self.assertEqual(
            provider.api_base_url,
            "https://api.telegram.org/bottest_token",
        )

    @patch("service.telegram.requests.post")
    def test_send_text_message(self, mock_post):
        """Test sending a text message"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        provider = TelegramProvider(token="test_token")
        message = OutgoingMessage(
            channel="telegram",
            to="123456789",
            from_="987654321",
            text="Hello World",
            reply_as_audio=False,
        )

        provider.send(message)

        mock_post.assert_called_once()
        call_args = mock_post.call_args

        # Verify URL
        self.assertEqual(
            call_args[0][0],
            "https://api.telegram.org/bottest_token/sendMessage",
        )

        # Verify payload
        payload = call_args[1]["json"]
        self.assertEqual(payload["chat_id"], "123456789")
        self.assertEqual(payload["text"], "Hello World")
        self.assertEqual(payload["parse_mode"], "Markdown")

    def test_from_settings_missing_token(self):
        """Test that from_settings raises error if token is missing"""
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ValueError):
                TelegramProvider.from_settings()
