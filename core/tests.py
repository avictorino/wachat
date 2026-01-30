from django.test import Client, TestCase


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
