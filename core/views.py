import json
import logging
import os

from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt

from messaging.providers import TelegramAdapter
from messaging.tasks import process_message_task
from messaging.types import IncomingMessage
from service.data_deletion import (
    delete_user_data,
    normalize_phone_number,
    rate_limit_by_ip,
)

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
                extra={"has_request_secret": bool(request_secret)}
            )
            return JsonResponse(
                {"status": "error", "message": "Forbidden"}, status=403
            )
        
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
            "Received Telegram webhook",
            extra={"update_id": body.get("update_id")}
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
        
        # Normalize message using TelegramAdapter
        from messaging.providers import TelegramAdapter
        
        adapter = TelegramAdapter()
        normalized_headers = {}
        normalized_message = adapter.normalize(normalized_headers, body)
        
        if not normalized_message:
            logger.warning("Failed to normalize Telegram message")
            return JsonResponse({"status": "ok"}, status=200)
        
        # Convert to IncomingMessage
        msg = IncomingMessage(
            channel="telegram",
            from_=normalized_message.sender_id,
            to=normalized_message.recipient_id,
            text=normalized_message.message_body,
            media_url=normalized_message.media_url,
            raw_payload=normalized_message.raw_payload,
            reply_as_audio=normalized_message.reply_as_audio,
        )
        
        # Process message (note: process_message_task is synchronous in this codebase)
        process_message_task(msg)
        
        # Return 200 quickly
        return JsonResponse({"status": "ok"}, status=200)


def privacy_policy_view(request):
    """
    View for displaying the Privacy Policy page.
    """
    return render(request, "privacy_policy.html")


def terms_of_service_view(request):
    """
    View for displaying the Terms of Service page.
    """
    return render(request, "terms_of_service.html")


def data_deletion_view(request):
    """
    View for displaying and handling data deletion requests.

    GET: Display the data deletion form
    POST: Process the data deletion request with rate limiting
    """
    if request.method == "GET":
        return render(request, "data_deletion.html")

    elif request.method == "POST":
        # Check rate limit
        result = _handle_data_deletion_post(request)

        if result is None:
            # Rate limit exceeded
            return render(
                request,
                "data_deletion.html",
                {
                    "rate_limited": True,
                },
            )

        return result


@rate_limit_by_ip(max_requests=5, window_seconds=3600)
def _handle_data_deletion_post(request):
    """
    Handle POST request for data deletion.
    Rate limited to 5 requests per hour per IP.
    """
    # Get phone number from form
    phone = request.POST.get("phone", "").strip()

    # Validate phone number format
    if not phone:
        return render(
            request,
            "data_deletion.html",
            {"error": "Por favor, forneça um número de telefone válido."},
        )

    # Normalize phone number
    try:
        normalized_phone = normalize_phone_number(phone)
    except (ValueError, TypeError, AttributeError):
        return render(
            request,
            "data_deletion.html",
            {
                "error": "Formato de número de telefone inválido. "
                "Use o formato E.164 (ex: +5511999999999)."
            },
        )

    # Delete user data
    success, error = delete_user_data(normalized_phone)

    if success:
        # Always return success message (don't reveal if user exists)
        return render(
            request,
            "data_deletion.html",
            {
                "success": True,
            },
        )
    else:
        return render(
            request,
            "data_deletion.html",
            {"error": error or "Ocorreu um erro ao processar sua solicitação."},
        )
