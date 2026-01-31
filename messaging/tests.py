from unittest.mock import MagicMock, patch

from django.test import TestCase

from messaging.types import OutgoingMessage
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


class GetFriendOrInitPersonTest(TestCase):
    """Test cases for get_friend_or_init_person function"""

    def setUp(self):
        """Set up test fixtures"""
        from django.contrib.auth.models import User

        # Clean up any existing test users
        User.objects.filter(username__startswith="test_").delete()

    def tearDown(self):
        """Clean up test data"""
        from django.contrib.auth.models import User

        User.objects.filter(username__startswith="test_").delete()

    @patch("service.whatsapp.infer_gender_from_name")
    def test_get_friend_with_unknown_gender(self, mock_infer_gender):
        """Test that unknown gender doesn't cause crash"""
        from messaging.types import IncomingMessage
        from service.whatsapp import get_friend_or_init_person

        # Arrange - mock gender inference to return 'unknown'
        mock_infer_gender.return_value = "unknown"

        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "contacts": [{"profile": {"name": "Test User"}}],
                                "messages": [{"from": "test_123"}],
                            }
                        }
                    ]
                }
            ]
        }

        message = IncomingMessage(
            channel="whatsapp_facebook",
            to="5511999999999",
            from_="test_123",
            text="Hello",
            raw_payload=payload,
        )

        # Act - should not raise IndexError
        friend = get_friend_or_init_person(message)

        # Assert
        self.assertIsNotNone(friend)
        self.assertIsNotNone(friend.name)
        self.assertIsNotNone(friend.gender)

    @patch("service.whatsapp.infer_gender_from_name")
    def test_get_friend_with_male_gender(self, mock_infer_gender):
        """Test that male gender works correctly"""
        from messaging.types import IncomingMessage
        from service.whatsapp import get_friend_or_init_person
        from core.constants import Gender

        # Arrange
        mock_infer_gender.return_value = "male"

        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "contacts": [{"profile": {"name": "John Doe"}}],
                                "messages": [{"from": "test_male_456"}],
                            }
                        }
                    ]
                }
            ]
        }

        message = IncomingMessage(
            channel="whatsapp_facebook",
            to="5511999999999",
            from_="test_male_456",
            text="Hello",
            raw_payload=payload,
        )

        # Act
        friend = get_friend_or_init_person(message)

        # Assert
        self.assertIsNotNone(friend)
        self.assertEqual(friend.gender, Gender.MALE)

    @patch("service.whatsapp.infer_gender_from_name")
    def test_get_friend_with_female_gender(self, mock_infer_gender):
        """Test that female gender works correctly"""
        from messaging.types import IncomingMessage
        from service.whatsapp import get_friend_or_init_person
        from core.constants import Gender

        # Arrange
        mock_infer_gender.return_value = "female"

        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "contacts": [{"profile": {"name": "Jane Doe"}}],
                                "messages": [{"from": "test_female_789"}],
                            }
                        }
                    ]
                }
            ]
        }

        message = IncomingMessage(
            channel="whatsapp_facebook",
            to="5511999999999",
            from_="test_female_789",
            text="Hello",
            raw_payload=payload,
        )

        # Act
        friend = get_friend_or_init_person(message)

        # Assert
        self.assertIsNotNone(friend)
        self.assertEqual(friend.gender, Gender.FEMALE)

    def test_get_friend_with_empty_payload(self):
        """Test that empty payload doesn't cause crash"""
        from messaging.types import IncomingMessage
        from service.whatsapp import get_friend_or_init_person

        # Arrange
        message = IncomingMessage(
            channel="whatsapp_facebook",
            to="5511999999999",
            from_="test_empty_001",
            text="Hello",
            raw_payload={},
        )

        # Act - should not raise exception
        friend = get_friend_or_init_person(message)

        # Assert
        self.assertIsNotNone(friend)

    def test_get_friend_with_none_payload(self):
        """Test that None payload doesn't cause crash"""
        from messaging.types import IncomingMessage
        from service.whatsapp import get_friend_or_init_person

        # Arrange
        message = IncomingMessage(
            channel="whatsapp_facebook",
            to="5511999999999",
            from_="test_none_002",
            text="Hello",
            raw_payload=None,
        )

        # Act - should not raise exception
        friend = get_friend_or_init_person(message)

        # Assert
        self.assertIsNotNone(friend)

    @patch("service.whatsapp.infer_gender_from_name")
    def test_get_friend_with_malformed_payload(self, mock_infer_gender):
        """Test that malformed payload is handled gracefully"""
        from messaging.types import IncomingMessage
        from service.whatsapp import get_friend_or_init_person

        # Arrange
        mock_infer_gender.return_value = "unknown"

        # Malformed payload missing expected structure
        payload = {"unexpected": "structure"}

        message = IncomingMessage(
            channel="whatsapp_facebook",
            to="5511999999999",
            from_="test_malformed_003",
            text="Hello",
            raw_payload=payload,
        )

        # Act - should not raise exception
        friend = get_friend_or_init_person(message)

        # Assert
        self.assertIsNotNone(friend)

    def test_get_existing_friend(self):
        """Test that existing user returns existing friend"""
        from django.contrib.auth.models import User
        from core.models import VirtualFriend
        from messaging.types import IncomingMessage
        from service.whatsapp import get_friend_or_init_person
        from core.constants import Gender

        # Arrange - create existing user and friend
        user = User.objects.create(username="test_existing_004", is_active=True)
        existing_friend = VirtualFriend.objects.create(
            owner=user, name="Pedro", gender=Gender.MALE
        )

        message = IncomingMessage(
            channel="whatsapp_facebook",
            to="5511999999999",
            from_="test_existing_004",
            text="Hello",
            raw_payload=None,
        )

        # Act
        friend = get_friend_or_init_person(message)

        # Assert
        self.assertEqual(friend.id, existing_friend.id)
        self.assertEqual(friend.name, "Pedro")
