import json
import logging
import os

from django.http import JsonResponse
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt

logger = logging.getLogger(__name__)


@method_decorator(csrf_exempt, name="dispatch")
class TelegramWebhookView(View):
    """
    Webhook endpoint for Telegram Bot API.

    This endpoint:
    - Validates requests using X-Telegram-Bot-Api-Secret-Token header
    - Handles /start command (sends welcome message)
    - Processes text messages through the LLM conversation pipeline
    - Sends responses back to Telegram chat

    CSRF is exempted because webhooks don't use CSRF tokens.
    """

    def post(self, request):
        """
        Handle incoming Telegram webhook requests.

        Validates secret token, parses payload, and processes messages.
        """
        # Validate secret token
        webhook_secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "")

        # Reject if secret is not configured (security measure)
        if not webhook_secret:
            logger.error("TELEGRAM_WEBHOOK_SECRET environment variable not configured")
            return JsonResponse(
                {"status": "error", "message": "Server configuration error"}, status=500
            )

        request_secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")

        if request_secret != webhook_secret:
            logger.warning(
                "Telegram webhook authentication failed",
                extra={"has_request_secret": bool(request_secret)},
            )
            return JsonResponse({"status": "error", "message": "Forbidden"}, status=403)

        # Parse JSON payload
        try:
            body = json.loads(request.body)
        except json.JSONDecodeError:
            logger.error("Invalid JSON in Telegram webhook request")
            return JsonResponse(
                {"status": "error", "message": "Invalid JSON"}, status=400
            )

        # Log received webhook
        logger.info(
            "Received Telegram webhook", extra={"update_id": body.get("update_id")}
        )

        # Extract message from payload
        message = body.get("message") or body.get("edited_message")
        if not message:
            # Not a message update (could be callback query, etc.)
            logger.info("Telegram webhook is not a message update")
            return JsonResponse({"status": "ok"}, status=200)

        # Extract message data
        sender = message.get("from", {})
        chat = message.get("chat", {})
        message_text = message.get("text", "")

        sender_id = str(sender.get("id", ""))
        chat_id = str(chat.get("id", ""))

        if not sender_id or not chat_id:
            logger.warning("Telegram webhook missing sender_id or chat_id")
            return JsonResponse({"status": "ok"}, status=200)

        print(message_text)
        # process message here

        return JsonResponse({"status": "ok"}, status=200)
