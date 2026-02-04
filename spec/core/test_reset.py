"""
Tests for the Reset User Data functionality.
"""

from datetime import timedelta
from unittest.mock import Mock, patch

from django.test import TestCase
from django.utils import timezone

from core.models import Message, Profile
from services.reset_user_data import ResetUserDataUseCase


class ResetUserDataUseCaseTest(TestCase):
    """Tests for the ResetUserDataUseCase."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a test profile with some data
        self.profile = Profile.objects.create(
            telegram_user_id="123456",
            name="Test User",
            phone_number="+1234567890",
            inferred_gender="male",
        )

        # Create some messages
        Message.objects.create(
            profile=self.profile,
            role="assistant",
            content="Welcome message",
            channel="telegram",
        )
        Message.objects.create(
            profile=self.profile,
            role="user",
            content="Hello",
            channel="telegram",
        )
        Message.objects.create(
            profile=self.profile,
            role="assistant",
            content="Response",
            channel="telegram",
        )

    def test_execute_deletes_all_user_data(self):
        """Test that execute deletes profile and all related messages."""
        # Verify data exists
        self.assertEqual(Profile.objects.count(), 1)
        self.assertEqual(Message.objects.count(), 3)

        # Execute reset
        result = ResetUserDataUseCase.execute("123456")

        # Verify deletion
        self.assertTrue(result)
        self.assertEqual(Profile.objects.count(), 0)
        self.assertEqual(Message.objects.count(), 0)

    def test_execute_with_nonexistent_user(self):
        """Test that execute returns False for non-existent user."""
        result = ResetUserDataUseCase.execute("nonexistent")

        # Should return False but not raise error
        self.assertFalse(result)

        # Original data should still exist
        self.assertEqual(Profile.objects.count(), 1)
        self.assertEqual(Message.objects.count(), 3)

    def test_execute_is_atomic(self):
        """Test that execute uses transaction and rolls back on error."""
        # This test verifies atomicity by mocking a failure
        with patch.object(Profile, "delete", side_effect=Exception("Test error")):
            with self.assertRaises(Exception):
                ResetUserDataUseCase.execute("123456")

        # Data should still exist due to rollback
        self.assertEqual(Profile.objects.count(), 1)
        self.assertEqual(Message.objects.count(), 3)

    def test_execute_finds_user_by_phone_number(self):
        """Test that execute can find user by phone_number as fallback."""
        # Create a profile without telegram_user_id
        profile2 = Profile.objects.create(
            name="Phone User",
            phone_number="+9876543210",
        )
        Message.objects.create(
            profile=profile2,
            role="user",
            content="Test message",
            channel="telegram",
        )

        # Should find by phone number
        result = ResetUserDataUseCase.execute("+9876543210")

        self.assertTrue(result)
        # profile2 should be deleted
        self.assertFalse(Profile.objects.filter(id=profile2.id).exists())
        # Original profile should still exist
        self.assertTrue(Profile.objects.filter(id=self.profile.id).exists())

    def test_execute_is_idempotent(self):
        """Test that calling execute multiple times doesn't raise errors."""
        # First execution
        result1 = ResetUserDataUseCase.execute("123456")
        self.assertTrue(result1)

        # Second execution (user already deleted)
        result2 = ResetUserDataUseCase.execute("123456")
        self.assertFalse(result2)

        # Should not raise any errors
        self.assertEqual(Profile.objects.count(), 0)


class ResetCommandWebhookTest(TestCase):
    """Tests for the /reset command webhook handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.webhook_url = "/webhooks/telegram/"
        self.webhook_secret = "test-secret"

        # Create a test profile
        self.profile = Profile.objects.create(
            telegram_user_id="123456",
            name="Test User",
        )

    @patch("core.views.TelegramService")
    @patch.dict("os.environ", {"TELEGRAM_WEBHOOK_SECRET": "test-secret"})
    def test_reset_command_requests_confirmation(self, mock_telegram_service):
        """Test that /reset command asks for confirmation."""
        mock_service_instance = Mock()
        mock_telegram_service.return_value = mock_service_instance
        mock_service_instance.send_message.return_value = True

        payload = {
            "update_id": 1,
            "message": {
                "message_id": 1,
                "from": {"id": 123456, "first_name": "Test"},
                "chat": {"id": 123456},
                "text": "/reset",
            },
        }

        response = self.client.post(
            self.webhook_url,
            data=payload,
            content_type="application/json",
            HTTP_X_TELEGRAM_BOT_API_SECRET_TOKEN="test-secret",
        )

        self.assertEqual(response.status_code, 200)

        # Check that confirmation message was sent
        mock_service_instance.send_message.assert_called_once()
        call_args = mock_service_instance.send_message.call_args
        self.assertIn("CONFIRM", call_args[0][1])
        self.assertIn("permanently delete", call_args[0][1])

        # Check that profile has pending confirmation
        self.profile.refresh_from_db()
        self.assertTrue(self.profile.pending_reset_confirmation)
        self.assertIsNotNone(self.profile.reset_confirmation_timestamp)

    @patch("core.views.TelegramService")
    @patch.dict("os.environ", {"TELEGRAM_WEBHOOK_SECRET": "test-secret"})
    def test_reset_confirmation_with_confirm(self, mock_telegram_service):
        """Test that typing CONFIRM completes the reset."""
        mock_service_instance = Mock()
        mock_telegram_service.return_value = mock_service_instance
        mock_service_instance.send_message.return_value = True

        # Set profile to pending confirmation state
        self.profile.pending_reset_confirmation = True
        self.profile.reset_confirmation_timestamp = timezone.now()
        self.profile.save()

        # Create some messages
        Message.objects.create(
            profile=self.profile,
            role="user",
            content="Test message",
        )

        payload = {
            "update_id": 2,
            "message": {
                "message_id": 2,
                "from": {"id": 123456, "first_name": "Test"},
                "chat": {"id": 123456},
                "text": "CONFIRM",
            },
        }

        response = self.client.post(
            self.webhook_url,
            data=payload,
            content_type="application/json",
            HTTP_X_TELEGRAM_BOT_API_SECRET_TOKEN="test-secret",
        )

        self.assertEqual(response.status_code, 200)

        # Check that success message was sent
        mock_service_instance.send_message.assert_called_once()
        call_args = mock_service_instance.send_message.call_args
        self.assertIn("successfully deleted", call_args[0][1])

        # Check that profile and messages were deleted
        self.assertFalse(Profile.objects.filter(id=self.profile.id).exists())
        self.assertEqual(Message.objects.count(), 0)

    @patch("core.views.TelegramService")
    @patch.dict("os.environ", {"TELEGRAM_WEBHOOK_SECRET": "test-secret"})
    def test_reset_confirmation_cancel(self, mock_telegram_service):
        """Test that typing anything other than CONFIRM cancels reset."""
        mock_service_instance = Mock()
        mock_telegram_service.return_value = mock_service_instance
        mock_service_instance.send_message.return_value = True

        # Set profile to pending confirmation state
        self.profile.pending_reset_confirmation = True
        self.profile.reset_confirmation_timestamp = timezone.now()
        self.profile.save()

        payload = {
            "update_id": 2,
            "message": {
                "message_id": 2,
                "from": {"id": 123456, "first_name": "Test"},
                "chat": {"id": 123456},
                "text": "cancel",
            },
        }

        response = self.client.post(
            self.webhook_url,
            data=payload,
            content_type="application/json",
            HTTP_X_TELEGRAM_BOT_API_SECRET_TOKEN="test-secret",
        )

        self.assertEqual(response.status_code, 200)

        # Check that cancellation message was sent
        mock_service_instance.send_message.assert_called_once()
        call_args = mock_service_instance.send_message.call_args
        self.assertIn("cancelled", call_args[0][1])

        # Check that profile still exists
        self.profile.refresh_from_db()
        self.assertFalse(self.profile.pending_reset_confirmation)
        self.assertIsNone(self.profile.reset_confirmation_timestamp)

    @patch("core.views.TelegramService")
    @patch.dict("os.environ", {"TELEGRAM_WEBHOOK_SECRET": "test-secret"})
    def test_reset_confirmation_timeout(self, mock_telegram_service):
        """Test that confirmation times out after 5 minutes."""
        mock_service_instance = Mock()
        mock_telegram_service.return_value = mock_service_instance
        mock_service_instance.send_message.return_value = True

        # Set profile to pending confirmation state with old timestamp
        self.profile.pending_reset_confirmation = True
        self.profile.reset_confirmation_timestamp = timezone.now() - timedelta(
            minutes=6
        )
        self.profile.save()

        payload = {
            "update_id": 2,
            "message": {
                "message_id": 2,
                "from": {"id": 123456, "first_name": "Test"},
                "chat": {"id": 123456},
                "text": "CONFIRM",
            },
        }

        response = self.client.post(
            self.webhook_url,
            data=payload,
            content_type="application/json",
            HTTP_X_TELEGRAM_BOT_API_SECRET_TOKEN="test-secret",
        )

        self.assertEqual(response.status_code, 200)

        # Check that timeout message was sent
        mock_service_instance.send_message.assert_called_once()
        call_args = mock_service_instance.send_message.call_args
        self.assertIn("timeout", call_args[0][1].lower())

        # Check that profile still exists and confirmation was cleared
        self.profile.refresh_from_db()
        self.assertFalse(self.profile.pending_reset_confirmation)
        self.assertIsNone(self.profile.reset_confirmation_timestamp)

    @patch("core.views.TelegramService")
    @patch.dict("os.environ", {"TELEGRAM_WEBHOOK_SECRET": "test-secret"})
    def test_reset_for_nonexistent_user(self, mock_telegram_service):
        """Test /reset for user without profile."""
        mock_service_instance = Mock()
        mock_telegram_service.return_value = mock_service_instance
        mock_service_instance.send_message.return_value = True

        payload = {
            "update_id": 1,
            "message": {
                "message_id": 1,
                "from": {"id": 999999, "first_name": "New User"},
                "chat": {"id": 999999},
                "text": "/reset",
            },
        }

        response = self.client.post(
            self.webhook_url,
            data=payload,
            content_type="application/json",
            HTTP_X_TELEGRAM_BOT_API_SECRET_TOKEN="test-secret",
        )

        self.assertEqual(response.status_code, 200)

        # Check that appropriate message was sent
        mock_service_instance.send_message.assert_called_once()
        call_args = mock_service_instance.send_message.call_args
        self.assertIn("don't have any data", call_args[0][1].lower())

    @patch("core.views.TelegramService")
    @patch.dict("os.environ", {"TELEGRAM_WEBHOOK_SECRET": "test-secret"})
    def test_reset_idempotent(self, mock_telegram_service):
        """Test that calling /reset twice doesn't cause errors."""
        mock_service_instance = Mock()
        mock_telegram_service.return_value = mock_service_instance
        mock_service_instance.send_message.return_value = True

        payload = {
            "update_id": 1,
            "message": {
                "message_id": 1,
                "from": {"id": 123456, "first_name": "Test"},
                "chat": {"id": 123456},
                "text": "/reset",
            },
        }

        # First call
        response1 = self.client.post(
            self.webhook_url,
            data=payload,
            content_type="application/json",
            HTTP_X_TELEGRAM_BOT_API_SECRET_TOKEN="test-secret",
        )
        self.assertEqual(response1.status_code, 200)

        # Set to confirmed and delete data
        self.profile.refresh_from_db()
        ResetUserDataUseCase.execute("123456")

        # Second call (user already deleted)
        payload["update_id"] = 2
        payload["message"]["message_id"] = 2
        response2 = self.client.post(
            self.webhook_url,
            data=payload,
            content_type="application/json",
            HTTP_X_TELEGRAM_BOT_API_SECRET_TOKEN="test-secret",
        )

        self.assertEqual(response2.status_code, 200)
        # Should get "don't have any data" message
