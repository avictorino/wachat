from unittest.mock import patch, MagicMock
from django.contrib.auth.models import User
from django.test import Client, TestCase
from django.core.cache import cache
import json

from core.models import VirtualFriend, UserSpiritualProfile, Conversation
from service.data_deletion import (
    normalize_phone_number,
    mask_phone_number,
    delete_user_data,
)


class MultiProviderWebhookViewTest(TestCase):
    """Test the unified multi-provider webhook endpoint"""

    def setUp(self):
        self.client = Client()

    @patch("core.views.process_message_task")
    def test_whatsapp_webhook_post(self, mock_process_message):
        """Test that WhatsApp webhooks are processed correctly"""
        payload = {
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

        response = self.client.post(
            "/api/webhooks/whatsapp-facebook/",
            data=json.dumps(payload),
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        
        # Verify message processing was called
        mock_process_message.assert_called_once()
        called_msg = mock_process_message.call_args[0][0]
        self.assertEqual(called_msg.channel, "whatsapp_facebook")
        self.assertEqual(called_msg.from_, "+5521967337683")
        self.assertEqual(called_msg.text, "Hello World")

    @patch("core.views.process_message_task")
    def test_facebook_messenger_webhook_post(self, mock_process_message):
        """Test that Facebook Messenger webhooks are processed correctly"""
        payload = {
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

        response = self.client.post(
            "/api/webhooks/whatsapp-facebook/",
            data=json.dumps(payload),
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        
        # Verify message processing was called
        mock_process_message.assert_called_once()
        called_msg = mock_process_message.call_args[0][0]
        self.assertEqual(called_msg.channel, "facebook")
        self.assertEqual(called_msg.from_, "1234567890")
        self.assertEqual(called_msg.text, "Hello from Messenger")

    @patch("core.views.process_message_task")
    def test_twilio_webhook_post(self, mock_process_message):
        """Test that Twilio webhooks are processed correctly"""
        payload = {
            "MessageSid": "SM1234567890",
            "From": "+5521967337683",
            "To": "+5511999999999",
            "Body": "Hello from Twilio",
            "NumMedia": "0"
        }

        response = self.client.post(
            "/api/webhooks/whatsapp-facebook/",
            data=json.dumps(payload),
            content_type="application/json",
            HTTP_USER_AGENT="TwilioProxy/1.1"
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        
        # Verify message processing was called
        mock_process_message.assert_called_once()
        called_msg = mock_process_message.call_args[0][0]
        self.assertEqual(called_msg.channel, "twilio")
        self.assertEqual(called_msg.from_, "+5521967337683")
        self.assertEqual(called_msg.text, "Hello from Twilio")

    @patch("core.views.process_message_task")
    def test_telegram_webhook_post(self, mock_process_message):
        """Test that Telegram webhooks are processed correctly"""
        payload = {
            "update_id": 123456789,
            "message": {
                "message_id": 123,
                "from": {
                    "id": 987654321,
                    "first_name": "John"
                },
                "chat": {
                    "id": 123456789,
                    "type": "private"
                },
                "date": 1234567890,
                "text": "Hello from Telegram"
            }
        }

        response = self.client.post(
            "/api/webhooks/whatsapp-facebook/",
            data=json.dumps(payload),
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        
        # Verify message processing was called
        mock_process_message.assert_called_once()
        called_msg = mock_process_message.call_args[0][0]
        self.assertEqual(called_msg.channel, "telegram")
        self.assertEqual(called_msg.from_, "987654321")
        self.assertEqual(called_msg.text, "Hello from Telegram")

    @patch("core.views.process_message_task")
    def test_slack_webhook_post(self, mock_process_message):
        """Test that Slack webhooks are processed correctly"""
        payload = {
            "type": "event_callback",
            "event": {
                "type": "message",
                "user": "U123456789",
                "text": "Hello from Slack",
                "channel": "C987654321",
                "ts": "1234567890.123456"
            }
        }

        response = self.client.post(
            "/api/webhooks/whatsapp-facebook/",
            data=json.dumps(payload),
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        
        # Verify message processing was called
        mock_process_message.assert_called_once()
        called_msg = mock_process_message.call_args[0][0]
        self.assertEqual(called_msg.channel, "slack")
        self.assertEqual(called_msg.from_, "U123456789")
        self.assertEqual(called_msg.text, "Hello from Slack")

    def test_webhook_invalid_json(self):
        """Test that invalid JSON returns 400"""
        response = self.client.post(
            "/api/webhooks/whatsapp-facebook/",
            data="invalid json",
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["status"], "error")

    @patch("core.views.process_message_task")
    def test_webhook_unknown_provider(self, mock_process_message):
        """Test that unknown providers return ok but don't process"""
        payload = {"unknown": "data"}

        response = self.client.post(
            "/api/webhooks/whatsapp-facebook/",
            data=json.dumps(payload),
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        
        # Verify message processing was NOT called
        mock_process_message.assert_not_called()


class WhatsAppWebhookViewTest(TestCase):
    def setUp(self):
        self.client = Client()

    @patch.dict("os.environ", {"FACEBOOK_WEBHOOK_VERIFICATION": "test_token"})
    def test_get_request_verification_success(self):
        """Test that GET requests with correct token verify successfully"""
        response = self.client.get(
            "/api/webhooks/whatsapp-facebook/",
            {
                "hub.mode": "subscribe",
                "hub.verify_token": "test_token",
                "hub.challenge": "challenge_string"
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content.decode(), "challenge_string")

    @patch.dict("os.environ", {"FACEBOOK_WEBHOOK_VERIFICATION": "test_token"})
    def test_get_request_verification_failure(self):
        """Test that GET requests with incorrect token fail verification"""
        response = self.client.get(
            "/api/webhooks/whatsapp-facebook/",
            {
                "hub.mode": "subscribe",
                "hub.verify_token": "wrong_token",
                "hub.challenge": "challenge_string"
            }
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.content.decode(), "Forbidden")


class DataDeletionServiceTest(TestCase):
    """Test data deletion service functions"""

    def test_normalize_phone_number(self):
        """Test phone number normalization"""
        test_cases = [
            ("+55 11 99999-9999", "+5511999999999"),
            ("+55 (11) 99999-9999", "+5511999999999"),
            ("+55-11-99999-9999", "+5511999999999"),
            ("+5511999999999", "+5511999999999"),
            ("5511999999999", "5511999999999"),
        ]
        for input_phone, expected in test_cases:
            result = normalize_phone_number(input_phone)
            self.assertEqual(result, expected)

    def test_mask_phone_number(self):
        """Test phone number masking for logs"""
        self.assertEqual(mask_phone_number("+5511999999999"), "+55...99")
        self.assertEqual(mask_phone_number("+1234"), "***")
        self.assertEqual(mask_phone_number("123"), "***")


class DataDeletionViewTest(TestCase):
    """Test data deletion views"""

    def setUp(self):
        self.client = Client()
        cache.clear()

        # Create test user
        self.test_phone = "+5511999999999"
        self.user = User.objects.create_user(
            username=self.test_phone, email="test@example.com"
        )

        # Create related data
        self.profile = UserSpiritualProfile.objects.create(user=self.user)
        self.friend = VirtualFriend.objects.create(owner=self.user, name="Test Friend")
        self.conversation = Conversation.objects.create(friend=self.friend)

    def test_get_data_deletion_page(self):
        """Test GET request to data deletion page"""
        response = self.client.get("/data-deletion/")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Solicitação de Exclusão de Dados")
        self.assertContains(response, "Número de Telefone")

    def test_post_valid_phone_number(self):
        """Test POST request with valid phone number"""
        response = self.client.post("/data-deletion/", {"phone": self.test_phone})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Solicitação Recebida")

        # Verify user was deleted
        self.assertFalse(User.objects.filter(username=self.test_phone).exists())

    def test_post_phone_with_formatting(self):
        """Test POST request with formatted phone number"""
        response = self.client.post("/data-deletion/", {"phone": "+55 11 99999-9999"})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Solicitação Recebida")

        # Verify user was deleted
        self.assertFalse(User.objects.filter(username=self.test_phone).exists())

    def test_post_nonexistent_phone(self):
        """Test POST request with non-existent phone number"""
        response = self.client.post("/data-deletion/", {"phone": "+5511888888888"})

        # Should still return success (don't reveal if user exists)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Solicitação Recebida")

    def test_post_empty_phone(self):
        """Test POST request with empty phone number"""
        response = self.client.post("/data-deletion/", {"phone": ""})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "número de telefone válido")

    def test_cascade_deletion(self):
        """Test that cascade deletion works for related data"""
        # Verify related data exists
        self.assertTrue(UserSpiritualProfile.objects.filter(user=self.user).exists())
        self.assertTrue(VirtualFriend.objects.filter(owner=self.user).exists())
        self.assertTrue(Conversation.objects.filter(friend=self.friend).exists())

        # Delete user
        success, error = delete_user_data(self.test_phone)

        self.assertTrue(success)
        self.assertIsNone(error)

        # Verify all related data was deleted
        self.assertFalse(User.objects.filter(username=self.test_phone).exists())
        self.assertFalse(
            UserSpiritualProfile.objects.filter(user__username=self.test_phone).exists()
        )
        self.assertFalse(
            VirtualFriend.objects.filter(owner__username=self.test_phone).exists()
        )
        # Verify conversations were also deleted via cascade
        self.assertFalse(Conversation.objects.filter(id=self.conversation.id).exists())

    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        # Make 5 requests (max allowed per hour)
        for i in range(5):
            response = self.client.post(
                "/data-deletion/", {"phone": f"+551199999999{i}"}
            )
            self.assertEqual(response.status_code, 200)

        # 6th request should be rate limited
        response = self.client.post("/data-deletion/", {"phone": "+5511999999995"})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Limite Excedido")

    def test_csrf_protection(self):
        """Test that CSRF protection is enabled"""
        # Create a client that doesn't follow redirects
        client = Client(enforce_csrf_checks=True)

        # POST without CSRF token should fail
        response = client.post("/data-deletion/", {"phone": "+5511999999999"})

        self.assertEqual(response.status_code, 403)


class TelegramWebhookViewTest(TestCase):
    """Test the dedicated Telegram webhook endpoint"""

    def setUp(self):
        self.client = Client()
        self.telegram_payload = {
            "update_id": 123456789,
            "message": {
                "message_id": 123,
                "from": {
                    "id": 987654321,
                    "first_name": "John",
                    "last_name": "Doe"
                },
                "chat": {
                    "id": 123456789,
                    "type": "private"
                },
                "date": 1234567890,
                "text": "Hello from Telegram"
            }
        }

    @patch.dict("os.environ", {"TELEGRAM_WEBHOOK_SECRET": "test_secret_token"})
    @patch("core.views.process_message_task")
    def test_telegram_webhook_with_valid_secret(self, mock_process_message):
        """Test that Telegram webhook with valid secret is processed"""
        response = self.client.post(
            "/webhooks/telegram/",
            data=json.dumps(self.telegram_payload),
            content_type="application/json",
            HTTP_X_TELEGRAM_BOT_API_SECRET_TOKEN="test_secret_token"
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        
        # Verify message processing was called
        mock_process_message.assert_called_once()
        called_msg = mock_process_message.call_args[0][0]
        self.assertEqual(called_msg.channel, "telegram")
        self.assertEqual(called_msg.from_, "987654321")
        self.assertEqual(called_msg.text, "Hello from Telegram")

    @patch.dict("os.environ", {"TELEGRAM_WEBHOOK_SECRET": "test_secret_token"})
    def test_telegram_webhook_with_invalid_secret(self):
        """Test that Telegram webhook with invalid secret returns 403"""
        response = self.client.post(
            "/webhooks/telegram/",
            data=json.dumps(self.telegram_payload),
            content_type="application/json",
            HTTP_X_TELEGRAM_BOT_API_SECRET_TOKEN="wrong_secret"
        )

        self.assertEqual(response.status_code, 403)
        data = response.json()
        self.assertEqual(data["status"], "error")
        self.assertEqual(data["message"], "Forbidden")

    @patch.dict("os.environ", {"TELEGRAM_WEBHOOK_SECRET": "test_secret_token"})
    def test_telegram_webhook_without_secret_header(self):
        """Test that Telegram webhook without secret header returns 403"""
        response = self.client.post(
            "/webhooks/telegram/",
            data=json.dumps(self.telegram_payload),
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 403)
        data = response.json()
        self.assertEqual(data["status"], "error")

    @patch.dict("os.environ", {"TELEGRAM_WEBHOOK_SECRET": "test_secret_token"})
    def test_telegram_webhook_invalid_json(self):
        """Test that invalid JSON returns 400"""
        response = self.client.post(
            "/webhooks/telegram/",
            data="invalid json",
            content_type="application/json",
            HTTP_X_TELEGRAM_BOT_API_SECRET_TOKEN="test_secret_token"
        )

        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data["status"], "error")

    @patch.dict("os.environ", {"TELEGRAM_WEBHOOK_SECRET": "test_secret_token"})
    @patch("core.views.process_message_task")
    def test_telegram_webhook_start_command(self, mock_process_message):
        """Test that /start command is processed correctly"""
        payload = {
            "update_id": 123456789,
            "message": {
                "message_id": 123,
                "from": {
                    "id": 987654321,
                    "first_name": "John"
                },
                "chat": {
                    "id": 123456789,
                    "type": "private"
                },
                "date": 1234567890,
                "text": "/start"
            }
        }

        response = self.client.post(
            "/webhooks/telegram/",
            data=json.dumps(payload),
            content_type="application/json",
            HTTP_X_TELEGRAM_BOT_API_SECRET_TOKEN="test_secret_token"
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        
        # Verify message processing was called
        mock_process_message.assert_called_once()
        called_msg = mock_process_message.call_args[0][0]
        self.assertEqual(called_msg.channel, "telegram")
        self.assertEqual(called_msg.text, "/start")

    @patch.dict("os.environ", {"TELEGRAM_WEBHOOK_SECRET": "test_secret_token"})
    def test_telegram_webhook_non_message_update(self):
        """Test that non-message updates return ok but don't process"""
        payload = {
            "update_id": 123456789,
            "callback_query": {
                "id": "123",
                "from": {"id": 987654321},
                "data": "button_click"
            }
        }

        response = self.client.post(
            "/webhooks/telegram/",
            data=json.dumps(payload),
            content_type="application/json",
            HTTP_X_TELEGRAM_BOT_API_SECRET_TOKEN="test_secret_token"
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
