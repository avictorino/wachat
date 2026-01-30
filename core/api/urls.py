from django.urls import path

from core.views import WhatsAppWebhookView

urlpatterns = [
    path("webhooks/whatsapp/", WhatsAppWebhookView.as_view()),
]
