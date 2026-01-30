import logging

from django.http import JsonResponse
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt

from core.services.speech_to_text import GroqWhisperSTT
from messaging.tasks import process_message_task
from messaging.types import IncomingMessage

logger = logging.getLogger(__name__)


@method_decorator(csrf_exempt, name="dispatch")
class WhatsAppWebhookView(View):
    """
    WhatsApp webhook endpoint for receiving messages from Twilio.
    CSRF is exempted because Twilio webhooks don't use CSRF tokens.
    """

    def get(self, request):

        return JsonResponse({})

    def post(self, request):
        data = request.POST

        from_number = data.get("From")  # 'whatsapp:+5521967337683'
        to_number = data.get("To")  # whatsapp:+5511999999999
        body = data.get("Body", "").strip()
        # Safely convert NumMedia to int, default to 0 if missing or invalid
        try:
            num_media = int(data.get("NumMedia", 0))
        except (ValueError, TypeError):
            num_media = 0

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
            raw_payload=dict(request.POST),
            reply_as_audio=reply_as_audio,
        )

        process_message_task(msg)

        return JsonResponse({"status": "ok"}, status=200)
