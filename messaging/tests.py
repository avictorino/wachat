from unittest.mock import MagicMock, patch

from django.test import TestCase

from messaging.types import OutgoingMessage
from messaging.providers import (
    WhatsAppAdapter,
    FacebookMessengerAdapter,
    TwilioAdapter,
    ProviderDetector,
    NormalizedMessage,
)
from service.whatsapp import FacebookWhatsAppProvider


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

    @patch("service.whatsapp.requests.post")
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

    @patch("service.whatsapp.requests.post")
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

    @patch("service.whatsapp.TextToSpeechService")
    @patch("service.whatsapp.requests.post")
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

    @patch("service.whatsapp.TextToSpeechService")
    @patch("service.whatsapp.requests.post")
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

    @patch("service.whatsapp.requests.post")
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


class WhatsAppAdapterTest(TestCase):
    """Test WhatsApp provider adapter"""

    def test_can_handle_whatsapp_webhook(self):
        """Test that WhatsApp adapter can identify WhatsApp webhooks"""
        adapter = WhatsAppAdapter()
        
        body = {
            "object": "whatsapp_business_account",
            "entry": []
        }
        headers = {}
        
        self.assertTrue(adapter.can_handle(headers, body))

    def test_can_handle_messaging_product_whatsapp(self):
        """Test identification by messaging_product field"""
        adapter = WhatsAppAdapter()
        
        body = {
            "entry": [{
                "changes": [{
                    "value": {
                        "messaging_product": "whatsapp"
                    }
                }]
            }]
        }
        headers = {}
        
        self.assertTrue(adapter.can_handle(headers, body))

    def test_cannot_handle_other_providers(self):
        """Test that WhatsApp adapter rejects other providers"""
        adapter = WhatsAppAdapter()
        
        body = {"object": "page"}
        headers = {}
        
        self.assertFalse(adapter.can_handle(headers, body))

    def test_normalize_text_message(self):
        """Test normalizing a WhatsApp text message"""
        adapter = WhatsAppAdapter()
        
        body = {
            "object": "whatsapp_business_account",
            "entry": [{
                "changes": [{
                    "field": "messages",
                    "value": {
                        "messaging_product": "whatsapp",
                        "metadata": {
                            "phone_number_id": "123456789",
                            "display_phone_number": "+5511999999999"
                        },
                        "messages": [{
                            "from": "+5521967337683",
                            "type": "text",
                            "timestamp": "1234567890",
                            "text": {
                                "body": "Hello World"
                            }
                        }]
                    }
                }]
            }]
        }
        headers = {}
        
        result = adapter.normalize(headers, body)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.sender_id, "+5521967337683")
        self.assertEqual(result.recipient_id, "+5511999999999")
        self.assertEqual(result.message_body, "Hello World")
        self.assertEqual(result.message_type, "text")
        self.assertEqual(result.timestamp, "1234567890")
        self.assertEqual(result.provider, "whatsapp")
        self.assertIsNone(result.media_url)
        self.assertFalse(result.reply_as_audio)

    def test_normalize_audio_message(self):
        """Test normalizing a WhatsApp audio message"""
        adapter = WhatsAppAdapter()
        
        body = {
            "object": "whatsapp_business_account",
            "entry": [{
                "changes": [{
                    "field": "messages",
                    "value": {
                        "messaging_product": "whatsapp",
                        "metadata": {
                            "phone_number_id": "123456789"
                        },
                        "messages": [{
                            "from": "+5521967337683",
                            "type": "audio",
                            "timestamp": "1234567890",
                            "audio": {
                                "id": "audio123"
                            }
                        }]
                    }
                }]
            }]
        }
        headers = {}
        
        result = adapter.normalize(headers, body)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.message_body, "[Audio message received]")
        self.assertEqual(result.message_type, "audio")
        self.assertEqual(result.media_url, "audio123")
        self.assertTrue(result.reply_as_audio)

    def test_normalize_image_message(self):
        """Test normalizing a WhatsApp image message"""
        adapter = WhatsAppAdapter()
        
        body = {
            "object": "whatsapp_business_account",
            "entry": [{
                "changes": [{
                    "field": "messages",
                    "value": {
                        "messaging_product": "whatsapp",
                        "metadata": {
                            "phone_number_id": "123456789"
                        },
                        "messages": [{
                            "from": "+5521967337683",
                            "type": "image",
                            "timestamp": "1234567890"
                        }]
                    }
                }]
            }]
        }
        headers = {}
        
        result = adapter.normalize(headers, body)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.message_body, "[Image message received]")
        self.assertEqual(result.message_type, "image")


class FacebookMessengerAdapterTest(TestCase):
    """Test Facebook Messenger provider adapter"""

    def test_can_handle_messenger_webhook(self):
        """Test that Messenger adapter can identify Messenger webhooks"""
        adapter = FacebookMessengerAdapter()
        
        body = {
            "object": "page",
            "entry": [{
                "messaging": []
            }]
        }
        headers = {}
        
        self.assertTrue(adapter.can_handle(headers, body))

    def test_cannot_handle_whatsapp(self):
        """Test that Messenger adapter rejects WhatsApp"""
        adapter = FacebookMessengerAdapter()
        
        body = {"object": "whatsapp_business_account"}
        headers = {}
        
        self.assertFalse(adapter.can_handle(headers, body))

    def test_normalize_text_message(self):
        """Test normalizing a Messenger text message"""
        adapter = FacebookMessengerAdapter()
        
        body = {
            "object": "page",
            "entry": [{
                "messaging": [{
                    "sender": {"id": "1234567890"},
                    "recipient": {"id": "0987654321"},
                    "timestamp": 1234567890,
                    "message": {
                        "mid": "msg123",
                        "text": "Hello from Messenger"
                    }
                }]
            }]
        }
        headers = {}
        
        result = adapter.normalize(headers, body)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.sender_id, "1234567890")
        self.assertEqual(result.recipient_id, "0987654321")
        self.assertEqual(result.message_body, "Hello from Messenger")
        self.assertEqual(result.message_type, "text")
        self.assertEqual(result.timestamp, "1234567890")
        self.assertEqual(result.provider, "facebook")
        self.assertIsNone(result.media_url)
        self.assertFalse(result.reply_as_audio)

    def test_normalize_image_attachment(self):
        """Test normalizing a Messenger image attachment"""
        adapter = FacebookMessengerAdapter()
        
        body = {
            "object": "page",
            "entry": [{
                "messaging": [{
                    "sender": {"id": "1234567890"},
                    "recipient": {"id": "0987654321"},
                    "timestamp": 1234567890,
                    "message": {
                        "mid": "msg123",
                        "attachments": [{
                            "type": "image",
                            "payload": {
                                "url": "https://example.com/image.jpg"
                            }
                        }]
                    }
                }]
            }]
        }
        headers = {}
        
        result = adapter.normalize(headers, body)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.message_body, "[Image message received]")
        self.assertEqual(result.message_type, "image")
        self.assertEqual(result.media_url, "https://example.com/image.jpg")

    def test_normalize_audio_attachment(self):
        """Test normalizing a Messenger audio attachment"""
        adapter = FacebookMessengerAdapter()
        
        body = {
            "object": "page",
            "entry": [{
                "messaging": [{
                    "sender": {"id": "1234567890"},
                    "recipient": {"id": "0987654321"},
                    "timestamp": 1234567890,
                    "message": {
                        "mid": "msg123",
                        "attachments": [{
                            "type": "audio",
                            "payload": {
                                "url": "https://example.com/audio.mp3"
                            }
                        }]
                    }
                }]
            }]
        }
        headers = {}
        
        result = adapter.normalize(headers, body)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.message_body, "[Audio message received]")
        self.assertEqual(result.message_type, "audio")
        self.assertEqual(result.media_url, "https://example.com/audio.mp3")


class TwilioAdapterTest(TestCase):
    """Test Twilio provider adapter"""

    def test_can_handle_twilio_user_agent(self):
        """Test that Twilio adapter can identify Twilio webhooks by User-Agent"""
        adapter = TwilioAdapter()
        
        headers = {"User-Agent": "TwilioProxy/1.1"}
        body = {}
        
        self.assertTrue(adapter.can_handle(headers, body))

    def test_can_handle_twilio_message_sid(self):
        """Test that Twilio adapter can identify Twilio webhooks by MessageSid"""
        adapter = TwilioAdapter()
        
        headers = {}
        body = {"MessageSid": "SM1234567890"}
        
        self.assertTrue(adapter.can_handle(headers, body))

    def test_can_handle_twilio_account_sid(self):
        """Test that Twilio adapter can identify Twilio webhooks by AccountSid"""
        adapter = TwilioAdapter()
        
        headers = {}
        body = {"AccountSid": "AC1234567890"}
        
        self.assertTrue(adapter.can_handle(headers, body))

    def test_cannot_handle_other_providers(self):
        """Test that Twilio adapter rejects other providers"""
        adapter = TwilioAdapter()
        
        headers = {}
        body = {"object": "page"}
        
        self.assertFalse(adapter.can_handle(headers, body))

    def test_normalize_sms_message(self):
        """Test normalizing a Twilio SMS message"""
        adapter = TwilioAdapter()
        
        body = {
            "MessageSid": "SM1234567890",
            "From": "+5521967337683",
            "To": "+5511999999999",
            "Body": "Hello from SMS",
            "NumMedia": "0"
        }
        headers = {}
        
        result = adapter.normalize(headers, body)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.sender_id, "+5521967337683")
        self.assertEqual(result.recipient_id, "+5511999999999")
        self.assertEqual(result.message_body, "Hello from SMS")
        self.assertEqual(result.message_type, "text")
        self.assertEqual(result.provider, "twilio")
        self.assertIsNone(result.media_url)
        self.assertFalse(result.reply_as_audio)

    def test_normalize_whatsapp_message(self):
        """Test normalizing a Twilio WhatsApp message"""
        adapter = TwilioAdapter()
        
        body = {
            "MessageSid": "SM1234567890",
            "From": "whatsapp:+5521967337683",
            "To": "whatsapp:+5511999999999",
            "Body": "Hello from WhatsApp",
            "NumMedia": "0"
        }
        headers = {}
        
        result = adapter.normalize(headers, body)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.sender_id, "whatsapp:+5521967337683")
        self.assertEqual(result.recipient_id, "whatsapp:+5511999999999")
        self.assertEqual(result.message_body, "Hello from WhatsApp")
        self.assertEqual(result.message_type, "text")
        self.assertEqual(result.provider, "twilio_whatsapp")

    def test_normalize_media_message(self):
        """Test normalizing a Twilio message with media"""
        adapter = TwilioAdapter()
        
        body = {
            "MessageSid": "SM1234567890",
            "From": "+5521967337683",
            "To": "+5511999999999",
            "Body": "Check this image",
            "NumMedia": "1",
            "MediaUrl0": "https://example.com/image.jpg",
            "MediaContentType0": "image/jpeg"
        }
        headers = {}
        
        result = adapter.normalize(headers, body)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.message_body, "Check this image")
        self.assertEqual(result.message_type, "image")
        self.assertEqual(result.media_url, "https://example.com/image.jpg")

    def test_normalize_audio_media(self):
        """Test normalizing a Twilio audio message"""
        adapter = TwilioAdapter()
        
        body = {
            "MessageSid": "SM1234567890",
            "From": "+5521967337683",
            "To": "+5511999999999",
            "Body": "",
            "NumMedia": "1",
            "MediaUrl0": "https://example.com/audio.mp3",
            "MediaContentType0": "audio/mpeg"
        }
        headers = {}
        
        result = adapter.normalize(headers, body)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.message_body, "[Audio message received]")
        self.assertEqual(result.message_type, "audio")
        self.assertEqual(result.media_url, "https://example.com/audio.mp3")


class ProviderDetectorTest(TestCase):
    """Test provider detection and routing"""

    def test_detect_whatsapp(self):
        """Test detecting WhatsApp provider"""
        detector = ProviderDetector()
        
        body = {
            "object": "whatsapp_business_account",
            "entry": [{
                "changes": [{
                    "field": "messages",
                    "value": {
                        "messaging_product": "whatsapp",
                        "metadata": {"phone_number_id": "123456789"},
                        "messages": [{
                            "from": "+5521967337683",
                            "type": "text",
                            "timestamp": "1234567890",
                            "text": {"body": "Hello"}
                        }]
                    }
                }]
            }]
        }
        headers = {}
        
        provider, message = detector.detect_and_normalize(headers, body)
        
        self.assertEqual(provider, "whatsapp")
        self.assertIsNotNone(message)
        self.assertEqual(message.message_body, "Hello")

    def test_detect_facebook_messenger(self):
        """Test detecting Facebook Messenger provider"""
        detector = ProviderDetector()
        
        body = {
            "object": "page",
            "entry": [{
                "messaging": [{
                    "sender": {"id": "1234567890"},
                    "recipient": {"id": "0987654321"},
                    "timestamp": 1234567890,
                    "message": {
                        "mid": "msg123",
                        "text": "Hello from Messenger"
                    }
                }]
            }]
        }
        headers = {}
        
        provider, message = detector.detect_and_normalize(headers, body)
        
        self.assertEqual(provider, "facebook")
        self.assertIsNotNone(message)
        self.assertEqual(message.message_body, "Hello from Messenger")

    def test_detect_twilio(self):
        """Test detecting Twilio provider"""
        detector = ProviderDetector()
        
        body = {
            "MessageSid": "SM1234567890",
            "From": "+5521967337683",
            "To": "+5511999999999",
            "Body": "Hello from Twilio",
            "NumMedia": "0"
        }
        headers = {"User-Agent": "TwilioProxy/1.1"}
        
        provider, message = detector.detect_and_normalize(headers, body)
        
        self.assertEqual(provider, "twilio")
        self.assertIsNotNone(message)
        self.assertEqual(message.message_body, "Hello from Twilio")

    def test_detect_twilio_whatsapp(self):
        """Test detecting Twilio WhatsApp provider"""
        detector = ProviderDetector()
        
        body = {
            "MessageSid": "SM1234567890",
            "From": "whatsapp:+5521967337683",
            "To": "whatsapp:+5511999999999",
            "Body": "Hello from Twilio WhatsApp",
            "NumMedia": "0"
        }
        headers = {}
        
        provider, message = detector.detect_and_normalize(headers, body)
        
        self.assertEqual(provider, "twilio_whatsapp")
        self.assertIsNotNone(message)

    def test_unknown_provider(self):
        """Test handling unknown provider"""
        detector = ProviderDetector()
        
        body = {"unknown": "data"}
        headers = {}
        
        provider, message = detector.detect_and_normalize(headers, body)
        
        self.assertIsNone(provider)
        self.assertIsNone(message)
