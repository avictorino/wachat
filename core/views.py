import json
import logging
import os

from django.http import HttpResponse, JsonResponse
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt

from messaging.tasks import process_message_task
from messaging.types import IncomingMessage

logger = logging.getLogger(__name__)

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

                    process_message_task(msg)

        return JsonResponse({"status": "ok"}, status=200)
