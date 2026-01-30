from unittest.mock import MagicMock, patch

from django.test import TestCase

from messaging.providers.factory import get_provider
from messaging.providers.whatsapp_facebook import FacebookWhatsAppProvider
from messaging.types import OutgoingMessage


class FacebookWhatsAppProviderTest(TestCase):
    @patch.dict(
        "os.environ",
        {"FACEBOOK_TOKEN": "test_token", "FACEBOOK_PHONE_NUMBER_ID": "123456789"},
    )
    def test_from_settings(self):
        """Test creating provider from environment variables"""
        # Act
        provider = FacebookWhatsAppProvider.from_settings()

        # Assert
        self.assertIsInstance(provider, FacebookWhatsAppProvider)
        self.assertEqual(provider.token, "test_token")
        self.assertEqual(provider.phone_number_id, "123456789")
        self.assertEqual(
            provider.api_url,
            "https://graph.facebook.com/v22.0/123456789/messages",
        )

    @patch("messaging.providers.whatsapp_facebook.requests.post")
    def test_send_text_message(self, mock_post):
        """Test sending a simple text message"""
        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        provider = FacebookWhatsAppProvider(
            token="test_token", phone_number_id="123456789"
        )
        message = OutgoingMessage(
            channel="whatsapp_facebook",
            to="5521967337683",
            from_="5511999999999",
            text="Hello World",
            reply_as_audio=False,
        )

        # Act
        provider.send(message)

        # Assert
        mock_post.assert_called_once()
        call_args = mock_post.call_args

        # Verify URL
        self.assertEqual(
            call_args[0][0],
            "https://graph.facebook.com/v22.0/123456789/messages",
        )

        # Verify headers
        headers = call_args[1]["headers"]
        self.assertEqual(headers["Authorization"], "Bearer test_token")
        self.assertEqual(headers["Content-Type"], "application/json")

        # Verify payload
        payload = call_args[1]["json"]
        self.assertEqual(payload["messaging_product"], "whatsapp")
        self.assertEqual(payload["to"], "5521967337683")
        self.assertEqual(payload["type"], "text")
        self.assertEqual(payload["text"]["body"], "Hello World")

    @patch("messaging.providers.whatsapp_facebook.requests.post")
    def test_send_text_message_removes_whatsapp_prefix(self, mock_post):
        """Test that whatsapp: prefix is removed from phone number"""
        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        provider = FacebookWhatsAppProvider(
            token="test_token", phone_number_id="123456789"
        )
        message = OutgoingMessage(
            channel="whatsapp_facebook",
            to="whatsapp:5521967337683",  # With prefix
            from_="5511999999999",
            text="Hello World",
            reply_as_audio=False,
        )

        # Act
        provider.send(message)

        # Assert
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["to"], "5521967337683")  # Prefix removed

    @patch("messaging.providers.whatsapp_facebook.TextToSpeechService")
    @patch("messaging.providers.whatsapp_facebook.requests.post")
    def test_send_audio_message(self, mock_post, mock_tts_service):
        """Test sending an audio message"""
        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        mock_tts_instance = MagicMock()
        mock_tts_instance.speak_and_store.return_value = "https://example.com/audio.mp3"
        mock_tts_service.return_value = mock_tts_instance

        provider = FacebookWhatsAppProvider(
            token="test_token", phone_number_id="123456789"
        )
        message = OutgoingMessage(
            channel="whatsapp_facebook",
            to="5521967337683",
            from_="5511999999999",
            text="Hello World",
            reply_as_audio=True,
        )

        # Act
        provider.send(message)

        # Assert
        mock_tts_instance.speak_and_store.assert_called_once_with(
            text="Hello World", conversation_mode=message.conversation_mode
        )
        mock_post.assert_called_once()

        # Verify payload structure for audio
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["type"], "audio")
        self.assertEqual(payload["audio"]["link"], "https://example.com/audio.mp3")

    @patch("messaging.providers.whatsapp_facebook.TextToSpeechService")
    @patch("messaging.providers.whatsapp_facebook.requests.post")
    def test_send_audio_message_fallback_to_text(self, mock_post, mock_tts_service):
        """Test that audio message falls back to text if TTS fails"""
        # Arrange
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        mock_tts_instance = MagicMock()
        mock_tts_instance.speak_and_store.side_effect = Exception("TTS failed")
        mock_tts_service.return_value = mock_tts_instance

        provider = FacebookWhatsAppProvider(
            token="test_token", phone_number_id="123456789"
        )
        message = OutgoingMessage(
            channel="whatsapp_facebook",
            to="5521967337683",
            from_="5511999999999",
            text="Hello World",
            reply_as_audio=True,
        )

        # Act
        provider.send(message)

        # Assert - should fallback to text message
        mock_post.assert_called_once()
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["type"], "text")
        self.assertEqual(payload["text"]["body"], "Hello World")

    @patch("messaging.providers.whatsapp_facebook.requests.post")
    def test_send_message_api_error(self, mock_post):
        """Test handling of API errors"""
        # Arrange
        import requests

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()

        mock_post.return_value = mock_response

        provider = FacebookWhatsAppProvider(
            token="test_token", phone_number_id="123456789"
        )
        message = OutgoingMessage(
            channel="whatsapp_facebook",
            to="5521967337683",
            from_="5511999999999",
            text="Hello World",
            reply_as_audio=False,
        )

        # Act & Assert
        with self.assertRaises(requests.exceptions.HTTPError):
            provider.send(message)


class FactoryTest(TestCase):
    @patch.dict(
        "os.environ",
        {"FACEBOOK_TOKEN": "test_token", "FACEBOOK_PHONE_NUMBER_ID": "123456789"},
    )
    def test_get_provider_whatsapp_facebook(self):
        """Test factory returns FacebookWhatsAppProvider for whatsapp_facebook channel"""
        # Act
        provider = get_provider("whatsapp_facebook")

        # Assert
        self.assertIsInstance(provider, FacebookWhatsAppProvider)

    def test_get_provider_unsupported_channel(self):
        """Test factory raises error for unsupported channel"""
        # Act & Assert
        with self.assertRaises(ValueError) as context:
            get_provider("unsupported_channel")

        self.assertIn("Unsupported channel", str(context.exception))
