from django.contrib.auth.models import User
from django.test import Client, TestCase
from django.core.cache import cache

from core.models import VirtualFriend, UserSpiritualProfile, Conversation
from service.data_deletion import (
    normalize_phone_number,
    mask_phone_number,
    delete_user_data,
)


class WhatsAppWebhookViewTest(TestCase):
    def setUp(self):
        self.client = Client()

    def test_get_request_returns_ok(self):
        """Test that GET requests to WhatsApp webhook return 200 OK"""
        response = self.client.get("/api/webhooks/whatsapp/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/json")

        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("message", data)


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
