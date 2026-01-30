import logging

from django.shortcuts import get_object_or_404
from rest_framework import permissions, status, views
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView

from core.api.serializers import MessageSerializer, VirtualFriendSerializer
from core.llm.factory import get_llm_client
from core.models import Conversation, VirtualFriend
from core.services.orchestrator import chat_with_friend
from core.services.speech_to_text import GroqWhisperSTT
from messaging.dispatcher import dispatch
from messaging.tasks import process_message_task
from messaging.types import IncomingMessage

logger = logging.getLogger(__name__)


class WhatsAppWebhookView(APIView):
    permission_classes = []  # Twilio não usa auth

    def post(self, request):
        data = request.data

        from_number = data.get("From")  # 'whatsapp:+5521967337683'
        to_number = data.get("To")  # whatsapp:+5511999999999
        body = data.get("Body", "").strip()
        num_media = int(data.get("NumMedia", 0))

        user_text_parts = []
        reply_as_audio = False
        # Iterate over all media items and process only audio
        groq_whisper = GroqWhisperSTT()
        for media_index in range(num_media):
            content_type = data.get(f"MediaContentType{media_index}", "")
            media_url = data.get(f"MediaUrl{media_index}")

            if not media_url or not content_type:
                continue

            if content_type.startswith("audio/"):
                transcript = groq_whisper.transcribe_media_url(media_url)

                if transcript:
                    user_text_parts.append(transcript.strip())
                    reply_as_audio = True

        # Append typed text if present
        if body:
            user_text_parts.append(body.strip())

        # Final normalized user text
        user_text = " ".join(user_text_parts).strip()

        msg = IncomingMessage(
            channel="whatsapp",
            from_=from_number,
            to=to_number,
            text=(user_text or "").strip(),
            raw_payload=dict(request.data),
            reply_as_audio=reply_as_audio,
        )

        dispatch(process_message_task, msg)

        return Response({"status": "ok"}, status=status.HTTP_200_OK)


class FriendsListCreateView(views.APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        qs = VirtualFriend.objects.filter(owner=request.user).order_by("-created_at")
        return Response(VirtualFriendSerializer(qs, many=True).data)

    def post(self, request):
        data = request.data or {}
        friend = VirtualFriend.objects.create(
            owner=request.user,
            name=data.get("name", "Amigo Bíblico"),
            persona=data.get(
                "persona", "Amigo bíblico, acolhedor, baseado em Escrituras."
            ),
            tone=data.get("tone", "carinhoso"),
            age=data.get("age") or None,
            background=data.get("background") or {},
        )
        return Response(
            VirtualFriendSerializer(friend).data, status=status.HTTP_201_CREATED
        )


class ChatView(views.APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, friend_id: str):
        friend = get_object_or_404(VirtualFriend, id=friend_id, owner=request.user)
        user_text = (request.data or {}).get("text", "").strip()
        if not user_text:
            return Response(
                {"detail": "text is required"}, status=status.HTTP_400_BAD_REQUEST
            )

        result = chat_with_friend(
            friend=friend, user_text=user_text, llm=get_llm_client()
        )
        return Response(
            {
                "conversation_id": result.conversation_id,
                "assistant_message_id": result.assistant_message_id,
                "text": result.text,
            },
            status=status.HTTP_200_OK,
        )


class ConversationDetailView(views.APIView):
    permission_classes = [permissions.IsAuthenticated]

    @api_view(["GET"])
    def get_conversation(self, conversation_id):
        conversation = Conversation.objects.prefetch_related("messages").get(
            id=conversation_id
        )

        data = {
            "conversation_id": str(conversation.id),
            "mode": conversation.current_mode,
            "messages": MessageSerializer(
                conversation.messages.order_by("created_at"), many=True
            ).data,
        }

        return Response(data)
