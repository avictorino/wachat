import json
import logging
import os

from django.http import HttpResponse, JsonResponse
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

        try:
            process_message_task(msg)
        except Exception as ex:
            logger.exception(
                "Error processing webhook message",
                extra={
                    "exception_type": type(ex).__name__,
                    "exception_message": str(ex),
                },
            )

        return JsonResponse({"status": "ok"}, status=200)


@method_decorator(csrf_exempt, name="dispatch")
class FacebookWhatsAppWebhookView(View):
    """
    Facebook WhatsApp webhook endpoint for receiving messages from Facebook Graph API.
    CSRF is exempted because Facebook webhooks don't use CSRF tokens.
    """

    def get(self, request):
        """
        Webhook verification endpoint.
        Facebook sends a GET request with challenge token for verification.
        """
        mode = request.GET.get("hub.mode")
        token = request.GET.get("hub.verify_token")
        challenge = request.GET.get("hub.challenge")

        verify_token = os.environ.get("FACEBOOK_WEBHOOK_VERIFICATION", "")

        if mode == "subscribe" and token == verify_token:
            logger.info("Facebook webhook verified successfully")
            return HttpResponse(challenge, content_type="text/plain")
        else:
            logger.warning("Facebook webhook verification failed")
            return HttpResponse("Forbidden", status=403)

    def post(self, request):
        """
        Webhook endpoint for receiving messages from Facebook WhatsApp.
        """
        try:
            body = json.loads(request.body)
        except json.JSONDecodeError:
            logger.error("Invalid JSON in Facebook webhook")
            return JsonResponse(
                {"status": "error", "message": "Invalid JSON"}, status=400
            )

        # Facebook webhook structure:
        # {
        #   "object": "whatsapp_business_account",
        #   "entry": [{
        #     "id": "WHATSAPP_BUSINESS_ACCOUNT_ID",
        #     "changes": [{
        #       "value": {
        #         "messaging_product": "whatsapp",
        #         "metadata": {
        #           "display_phone_number": "PHONE_NUMBER",
        #           "phone_number_id": "PHONE_NUMBER_ID"
        #         },
        #         "contacts": [{"profile": {"name": "NAME"}, "wa_id": "WHATSAPP_ID"}],
        #         "messages": [{
        #           "from": "WHATSAPP_ID",
        #           "id": "MESSAGE_ID",
        #           "timestamp": "TIMESTAMP",
        #           "type": "text",
        #           "text": {"body": "MESSAGE_BODY"}
        #         }]
        #       },
        #       "field": "messages"
        #     }]
        #   }]
        # }

        # Process each entry
        for entry in body.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})

                # Only process message changes
                if change.get("field") != "messages":
                    continue

                # Get metadata
                metadata = value.get("metadata", {})
                phone_number_id = metadata.get("phone_number_id")
                display_phone_number = metadata.get("display_phone_number", "")

                # Process each message
                for message in value.get("messages", []):
                    from_number = message.get("from")
                    message_type = message.get("type")

                    # Extract message content based on type
                    text = ""
                    media_url = None
                    reply_as_audio = False

                    if message_type == "text":
                        text = message.get("text", {}).get("body", "")
                    elif message_type == "audio":
                        # Handle audio messages
                        audio_data = message.get("audio", {})
                        media_url = audio_data.get(
                            "id"
                        )  # Facebook provides media ID, needs to be fetched
                        # For now, just log that we received audio
                        logger.info(f"Received audio message with ID: {media_url}")
                        text = "[Audio message received]"
                        reply_as_audio = True
                    elif message_type == "image":
                        # Handle image messages
                        logger.info("Received image message")
                        text = "[Image message received]"
                    else:
                        logger.info(
                            f"Received unsupported message type: {message_type}"
                        )
                        text = f"[{message_type} message received]"

                    # Create incoming message
                    msg = IncomingMessage(
                        channel="whatsapp_facebook",
                        from_=from_number,
                        to=display_phone_number or phone_number_id,
                        text=text,
                        media_url=media_url,
                        raw_payload=body,
                        reply_as_audio=reply_as_audio,
                    )

                    # Dispatch message for processing
                    dispatch(process_message_task, msg)

        return JsonResponse({"status": "ok"}, status=200)
