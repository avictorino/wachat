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

        # Verify the IncomingMessage content
        call_args = mock_dispatch.call_args
        incoming_msg = call_args[0][1]  # Second argument (first is task)
        self.assertEqual(incoming_msg.from_, "whatsapp:+5521967337683")
        self.assertEqual(incoming_msg.to, "whatsapp:+5511999999999")
        self.assertEqual(incoming_msg.text, "Hello, this is a test message")
        self.assertEqual(incoming_msg.channel, "whatsapp")
        self.assertFalse(incoming_msg.reply_as_audio)

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

        # Verify the IncomingMessage content
        call_args = mock_dispatch.call_args
        incoming_msg = call_args[0][1]  # Second argument (first is task)
        self.assertEqual(incoming_msg.from_, "whatsapp:+5521967337683")
        self.assertEqual(incoming_msg.to, "whatsapp:+5511999999999")
        self.assertEqual(incoming_msg.text, "Transcribed text")
        self.assertTrue(incoming_msg.reply_as_audio)

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

        # Verify the IncomingMessage content
        call_args = mock_dispatch.call_args
        incoming_msg = call_args[0][1]  # Second argument (first is task)
        self.assertEqual(incoming_msg.text, "")
        self.assertFalse(incoming_msg.reply_as_audio)

    @patch("core.views.dispatch")
    @patch("core.views.GroqWhisperSTT")
    def test_post_whatsapp_webhook_invalid_num_media(self, mock_whisper, mock_dispatch):
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
        mock_dispatch.assert_called_once()


class FacebookWhatsAppWebhookViewTest(TestCase):
    def setUp(self):
        self.client = Client()

    @patch.dict("os.environ", {"FACEBOOK_WEBHOOK_VERIFICATION": "test_token"})
    def test_get_webhook_verification_success(self):
        """Test successful webhook verification"""
        # Arrange
        params = {
            "hub.mode": "subscribe",
            "hub.verify_token": "test_token",
            "hub.challenge": "challenge_string",
        }

        # Act
        response = self.client.get("/api/webhooks/whatsapp-facebook/", params)

        # Assert
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content.decode(), "challenge_string")

    @patch.dict("os.environ", {"FACEBOOK_WEBHOOK_VERIFICATION": "test_token"})
    def test_get_webhook_verification_failure(self):
        """Test failed webhook verification with wrong token"""
        # Arrange
        params = {
            "hub.mode": "subscribe",
            "hub.verify_token": "wrong_token",
            "hub.challenge": "challenge_string",
        }

        # Act
        response = self.client.get("/api/webhooks/whatsapp-facebook/", params)

        # Assert
        self.assertEqual(response.status_code, 403)

    @patch("core.views.dispatch")
    def test_post_facebook_webhook_text_message(self, mock_dispatch):
        """Test Facebook webhook with text message"""
        # Arrange
        data = {
            "object": "whatsapp_business_account",
            "entry": [
                {
                    "id": "123456789",
                    "changes": [
                        {
                            "value": {
                                "messaging_product": "whatsapp",
                                "metadata": {
                                    "display_phone_number": "5511999999999",
                                    "phone_number_id": "378663622006192",
                                },
                                "contacts": [
                                    {
                                        "profile": {"name": "John Doe"},
                                        "wa_id": "5521967337683",
                                    }
                                ],
                                "messages": [
                                    {
                                        "from": "5521967337683",
                                        "id": "wamid.123",
                                        "timestamp": "1234567890",
                                        "type": "text",
                                        "text": {"body": "Hello from Facebook"},
                                    }
                                ],
                            },
                            "field": "messages",
                        }
                    ],
                }
            ],
        }

        # Act
        response = self.client.post(
            "/api/webhooks/whatsapp-facebook/",
            data=data,
            content_type="application/json",
        )

        # Assert
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        mock_dispatch.assert_called_once()

        # Verify the IncomingMessage content
        call_args = mock_dispatch.call_args
        incoming_msg = call_args[0][1]  # Second argument (first is task)
        self.assertEqual(incoming_msg.from_, "5521967337683")
        self.assertEqual(incoming_msg.to, "5511999999999")
        self.assertEqual(incoming_msg.text, "Hello from Facebook")
        self.assertEqual(incoming_msg.channel, "whatsapp_facebook")
        self.assertFalse(incoming_msg.reply_as_audio)

    @patch("core.views.dispatch")
    def test_post_facebook_webhook_audio_message(self, mock_dispatch):
        """Test Facebook webhook with audio message"""
        # Arrange
        data = {
            "object": "whatsapp_business_account",
            "entry": [
                {
                    "id": "123456789",
                    "changes": [
                        {
                            "value": {
                                "messaging_product": "whatsapp",
                                "metadata": {
                                    "display_phone_number": "5511999999999",
                                    "phone_number_id": "378663622006192",
                                },
                                "messages": [
                                    {
                                        "from": "5521967337683",
                                        "id": "wamid.123",
                                        "timestamp": "1234567890",
                                        "type": "audio",
                                        "audio": {"id": "audio_id_123"},
                                    }
                                ],
                            },
                            "field": "messages",
                        }
                    ],
                }
            ],
        }

        # Act
        response = self.client.post(
            "/api/webhooks/whatsapp-facebook/",
            data=data,
            content_type="application/json",
        )

        # Assert
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        mock_dispatch.assert_called_once()

        # Verify the IncomingMessage content
        call_args = mock_dispatch.call_args
        incoming_msg = call_args[0][1]
        self.assertEqual(incoming_msg.channel, "whatsapp_facebook")
        self.assertTrue(incoming_msg.reply_as_audio)
        self.assertEqual(incoming_msg.text, "[Audio message received]")

    @patch("core.views.dispatch")
    def test_post_facebook_webhook_invalid_json(self, mock_dispatch):
        """Test Facebook webhook with invalid JSON"""
        # Act
        response = self.client.post(
            "/api/webhooks/whatsapp-facebook/",
            data="invalid json",
            content_type="application/json",
        )

        # Assert
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"status": "error", "message": "Invalid JSON"})
        mock_dispatch.assert_not_called()

    @patch("core.views.dispatch")
    def test_post_facebook_webhook_non_message_change(self, mock_dispatch):
        """Test Facebook webhook with non-message change type"""
        # Arrange
        data = {
            "object": "whatsapp_business_account",
            "entry": [
                {
                    "id": "123456789",
                    "changes": [
                        {
                            "value": {"some": "data"},
                            "field": "status",  # Not a "messages" field
                        }
                    ],
                }
            ],
        }

        # Act
        response = self.client.post(
            "/api/webhooks/whatsapp-facebook/",
            data=data,
            content_type="application/json",
        )

        # Assert
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        # Should not dispatch since it's not a message
        mock_dispatch.assert_not_called()
