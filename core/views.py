import logging

from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

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
