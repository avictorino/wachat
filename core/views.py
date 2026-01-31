import json
import logging
import os

from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt

from messaging.providers import ProviderDetector
from messaging.tasks import process_message_task
from messaging.types import IncomingMessage
from service.data_deletion import (
    delete_user_data,
    normalize_phone_number,
    rate_limit_by_ip,
)

logger = logging.getLogger(__name__)


@method_decorator(csrf_exempt, name="dispatch")
class FacebookWhatsAppWebhookView(View):
    """
    Unified webhook endpoint for receiving messages from multiple providers.
    Supports:
    - WhatsApp (via Facebook/Meta Graph API)
    - Facebook Messenger (via Facebook/Meta Graph API)
    - Twilio (WhatsApp and SMS)
    
    CSRF is exempted because webhooks don't use CSRF tokens.
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
        Unified webhook endpoint for receiving messages from multiple providers.
        Supports WhatsApp (Meta), Facebook Messenger, and Twilio.
        """
        try:
            body = json.loads(request.body)
        except json.JSONDecodeError:
            logger.error("Invalid JSON in webhook request")
            return JsonResponse(
                {"status": "error", "message": "Invalid JSON"}, status=400
            )

        # Convert Django headers to dict for provider detection
        headers = {
            key: value
            for key, value in request.META.items()
            if key.startswith("HTTP_") or key in ["CONTENT_TYPE", "CONTENT_LENGTH"]
        }
        # Normalize header names (remove HTTP_ prefix and convert to title case)
        normalized_headers = {}
        for key, value in headers.items():
            if key.startswith("HTTP_"):
                normalized_key = key[5:].replace("_", "-").title()
            else:
                normalized_key = key.replace("_", "-").title()
            normalized_headers[normalized_key] = value

        # Detect provider and normalize message
        detector = ProviderDetector()
        provider, normalized_message = detector.detect_and_normalize(
            normalized_headers, body
        )

        if not provider or not normalized_message:
            logger.warning(
                "Could not detect provider or normalize message",
                extra={"headers": normalized_headers, "body": body},
            )
            return JsonResponse({"status": "ok"}, status=200)

        logger.info(
            f"Received message from provider: {provider}",
            extra={
                "provider": provider,
                "sender": normalized_message.sender_id,
                "message_type": normalized_message.message_type,
            },
        )

        # Convert normalized message to IncomingMessage for backward compatibility
        # Map provider to channel type
        channel_map = {
            "whatsapp": "whatsapp_facebook",
            "facebook": "facebook",
            "twilio": "twilio",
            "twilio_whatsapp": "twilio_whatsapp",
        }
        
        if provider not in channel_map:
            logger.warning(
                f"Unknown provider '{provider}' not in channel map, using default 'whatsapp_facebook'",
                extra={"provider": provider}
            )
        
        channel = channel_map.get(provider, "whatsapp_facebook")

        msg = IncomingMessage(
            channel=channel,
            from_=normalized_message.sender_id,
            to=normalized_message.recipient_id,
            text=normalized_message.message_body,
            media_url=normalized_message.media_url,
            raw_payload=normalized_message.raw_payload,
            reply_as_audio=normalized_message.reply_as_audio,
        )

        process_message_task(msg)

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
