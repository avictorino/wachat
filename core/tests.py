from unittest.mock import MagicMock, patch

from django.test import Client, TestCase


class WhatsAppWebhookViewTest(TestCase):
    def setUp(self):
        self.client = Client()

    @patch("core.views.process_message_task")
    @patch("core.views.GroqWhisperSTT")
    def test_post_whatsapp_webhook_with_text(self, mock_whisper, mock_process_task):
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
        mock_process_task.assert_called_once()

        # Verify the IncomingMessage content
        call_args = mock_process_task.call_args
        incoming_msg = call_args[0][0]  # First argument
        self.assertEqual(incoming_msg.from_, "whatsapp:+5521967337683")
        self.assertEqual(incoming_msg.to, "whatsapp:+5511999999999")
        self.assertEqual(incoming_msg.text, "Hello, this is a test message")
        self.assertEqual(incoming_msg.channel, "whatsapp")
        self.assertFalse(incoming_msg.reply_as_audio)

    @patch("core.views.process_message_task")
    @patch("core.views.GroqWhisperSTT")
    def test_post_whatsapp_webhook_with_audio(self, mock_whisper, mock_process_task):
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
        mock_process_task.assert_called_once()
        mock_whisper_instance.transcribe_media_url.assert_called_once_with(
            "https://example.com/audio.ogg"
        )

        # Verify the IncomingMessage content
        call_args = mock_process_task.call_args
        incoming_msg = call_args[0][0]  # First argument
        self.assertEqual(incoming_msg.from_, "whatsapp:+5521967337683")
        self.assertEqual(incoming_msg.to, "whatsapp:+5511999999999")
        self.assertEqual(incoming_msg.text, "Transcribed text")
        self.assertTrue(incoming_msg.reply_as_audio)

    @patch("core.views.process_message_task")
    @patch("core.views.GroqWhisperSTT")
    def test_post_whatsapp_webhook_empty_message(self, mock_whisper, mock_process_task):
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
        mock_process_task.assert_called_once()

        # Verify the IncomingMessage content
        call_args = mock_process_task.call_args
        incoming_msg = call_args[0][0]  # First argument
        self.assertEqual(incoming_msg.text, "")
        self.assertFalse(incoming_msg.reply_as_audio)

    @patch("core.views.process_message_task")
    @patch("core.views.GroqWhisperSTT")
    def test_post_whatsapp_webhook_invalid_num_media(self, mock_whisper, mock_process_task):
        """Test webhook endpoint handles invalid NumMedia gracefully"""
        # Arrange
        data = {
            "From": "whatsapp:+5521967337683",
            "To": "whatsapp:+5511999999999",
            "Body": "Test message",
            "NumMedia": "invalid",  # Invalid value
        }

        # Act
        response = self.client.post("/api/webhooks/whatsapp/", data)

        # Assert - Should handle gracefully and not crash
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        mock_process_task.assert_called_once()

    @patch("core.views.process_message_task")
    @patch("core.views.GroqWhisperSTT")
    def test_post_whatsapp_webhook_handles_processing_exception(
        self, mock_whisper, mock_process_task
    ):
        """Test webhook endpoint handles exceptions in process_message_task gracefully"""
        # Arrange
        mock_process_task.side_effect = Exception("Simulated error")
        data = {
            "From": "whatsapp:+5521967337683",
            "To": "whatsapp:+5511999999999",
            "Body": "Test message",
            "NumMedia": "0",
        }

        # Act
        response = self.client.post("/api/webhooks/whatsapp/", data)

        # Assert - Should still return 200 even with exception
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        mock_process_task.assert_called_once()
