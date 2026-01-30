from django.urls import path

from core.views import (
    ChatView,
    ConversationDetailView,
    FriendsListCreateView,
    WhatsAppWebhookView,
)

urlpatterns = [
    path("friends/", FriendsListCreateView.as_view()),
    path("friends/<uuid:friend_id>/chat/", ChatView.as_view()),
    path("conversations/<uuid:conversation_id>/", ConversationDetailView.as_view()),
    path("webhooks/whatsapp", WhatsAppWebhookView.as_view()),
]
