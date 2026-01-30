from unittest.mock import MagicMock, patch

from django.test import Client, TestCase


class WhatsAppWebhookViewTest(TestCase):
    def setUp(self):
        self.client = Client()

    @patch("core.views.dispatch")
    @patch("core.views.GroqWhisperSTT")
    def test_post_whatsapp_webhook_with_text(self, mock_whisper, mock_dispatch):
        """Test webhook endpoint with text message"""
        # Arrange
        data = {
            "From": "whatsapp:+5521967337683",
            "To": "whatsapp:+5511999999999",
            "Body": "Hello, this is a test message",
            "NumMedia": "0",
        }

        # Act
        response = self.client.post("/api/webhooks/whatsapp/", data)

        # Assert
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        mock_dispatch.assert_called_once()

    @patch("core.views.dispatch")
    @patch("core.views.GroqWhisperSTT")
    def test_post_whatsapp_webhook_with_audio(self, mock_whisper, mock_dispatch):
        """Test webhook endpoint with audio message"""
        # Arrange
        mock_whisper_instance = MagicMock()
        mock_whisper_instance.transcribe_media_url.return_value = "Transcribed text"
        mock_whisper.return_value = mock_whisper_instance

        data = {
            "From": "whatsapp:+5521967337683",
            "To": "whatsapp:+5511999999999",
            "Body": "",
            "NumMedia": "1",
            "MediaContentType0": "audio/ogg",
            "MediaUrl0": "https://example.com/audio.ogg",
        }

        # Act
        response = self.client.post("/api/webhooks/whatsapp/", data)

        # Assert
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        mock_dispatch.assert_called_once()
        mock_whisper_instance.transcribe_media_url.assert_called_once_with(
            "https://example.com/audio.ogg"
        )

    @patch("core.views.dispatch")
    @patch("core.views.GroqWhisperSTT")
    def test_post_whatsapp_webhook_empty_message(self, mock_whisper, mock_dispatch):
        """Test webhook endpoint with empty message"""
        # Arrange
        data = {
            "From": "whatsapp:+5521967337683",
            "To": "whatsapp:+5511999999999",
            "Body": "",
            "NumMedia": "0",
        }

        # Act
        response = self.client.post("/api/webhooks/whatsapp/", data)

        # Assert
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        mock_dispatch.assert_called_once()

