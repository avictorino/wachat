import json
import logging
import os

from django.http import JsonResponse
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt

from core.models import Message, Profile
from services.groq_service import GroqService
from services.telegram_service import TelegramService

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

        # Handle /start command
        if message_text and message_text.strip().startswith("/start"):
            return self._handle_start_command(sender, chat_id)

        # Handle regular text messages
        if message_text:
            return self._handle_regular_message(
                sender, sender_id, chat_id, message_text
            )

        return JsonResponse({"status": "ok"}, status=200)

    def _handle_start_command(self, sender: dict, chat_id: str):
        """
        Handle the /start command for first-time onboarding.

        This method:
        1. Extracts user data from Telegram
        2. Creates or retrieves user Profile
        3. Infers gender using Groq
        4. Generates personalized welcome message
        5. Persists the welcome message
        6. Sends the message to the user

        Args:
            sender: Telegram sender data (from webhook payload)
            chat_id: Telegram chat ID

        Returns:
            JsonResponse indicating success
        """
        try:
            # Extract user data from Telegram
            telegram_user_id = str(sender.get("id", ""))
            first_name = sender.get("first_name", "")
            last_name = sender.get("last_name", "")
            phone_number = sender.get("phone_number")  # May not be available

            # Construct full name
            name = first_name
            if last_name:
                name = f"{first_name} {last_name}"

            if not telegram_user_id or not name:
                logger.error("Missing required user data from Telegram")
                return JsonResponse({"status": "ok"}, status=200)

            logger.info(f"Handling /start for user: {name} ({telegram_user_id})")

            # Create or get user profile
            profile, created = Profile.objects.get_or_create(
                telegram_user_id=telegram_user_id,
                defaults={"name": name, "phone_number": phone_number},
            )

            # Track if we need to save the profile
            needs_save = False

            # Update profile if it already exists
            if not created:
                profile.name = name
                if phone_number:
                    profile.phone_number = phone_number
                needs_save = True
                logger.info(f"Updated existing profile for {telegram_user_id}")
            else:
                logger.info(f"Created new profile for {telegram_user_id}")

            # Initialize services
            groq_service = GroqService()
            telegram_service = TelegramService()

            # Infer gender if not already done
            if not profile.inferred_gender:
                inferred_gender = groq_service.infer_gender(name)
                profile.inferred_gender = inferred_gender
                needs_save = True
                logger.info(f"Inferred gender for {name}: {inferred_gender}")

            # Save profile if needed (consolidate all updates into one save)
            if needs_save:
                profile.save()

            # Generate welcome message
            welcome_message = groq_service.generate_welcome_message(
                name=name, inferred_gender=profile.inferred_gender
            )

            # Persist the welcome message
            Message.objects.create(
                profile=profile, role="assistant", content=welcome_message
            )
            logger.info(f"Persisted welcome message for profile {profile.id}")

            # Send the message to Telegram
            success = telegram_service.send_message(chat_id, welcome_message)

            if success:
                logger.info(f"Welcome message sent to chat {chat_id}")
            else:
                logger.error(f"Failed to send welcome message to chat {chat_id}")

            return JsonResponse({"status": "ok"}, status=200)

        except Exception as e:
            logger.error(f"Error handling /start command: {str(e)}", exc_info=True)
            return JsonResponse({"status": "error"}, status=500)

    def _handle_regular_message(
        self, sender: dict, sender_id: str, chat_id: str, message_text: str
    ):
        """
        Handle regular text messages from users.

        This method implements the conversational flow after the welcome message:
        1. Retrieves or creates user profile
        2. Persists the user's message
        3. Detects intent from the message
        4. Generates an empathetic response using Groq
        5. Persists the assistant's response
        6. Sends the response back to the user via Telegram

        Args:
            sender: Telegram sender data (from webhook payload)
            sender_id: Telegram user ID as string
            chat_id: Telegram chat ID
            message_text: The text message from the user

        Returns:
            JsonResponse indicating success
        """
        try:
            # Get user profile
            try:
                profile = Profile.objects.get(telegram_user_id=sender_id)
            except Profile.DoesNotExist:
                # If profile doesn't exist, user hasn't used /start yet
                # Create a minimal profile and handle gracefully
                first_name = sender.get("first_name", "")
                last_name = sender.get("last_name", "")
                name = first_name
                if last_name:
                    name = f"{first_name} {last_name}"

                if not name:
                    name = "Amigo"

                profile = Profile.objects.create(telegram_user_id=sender_id, name=name)
                logger.info(
                    f"Created profile for user {sender_id} without /start command"
                )

            # Persist user message
            Message.objects.create(profile=profile, role="user", content=message_text)
            logger.info(f"Persisted user message for profile {profile.id}")

            # Initialize services
            groq_service = GroqService()
            telegram_service = TelegramService()

            # Detect intent if not already detected
            # We only detect intent on the first user message (when there are 2 messages total: welcome + first user message)
            # or on the very first message if user didn't start with /start
            message_count = Message.objects.filter(profile=profile).count()

            if not profile.detected_intent and message_count <= 2:
                # Detect and store intent
                detected_intent = groq_service.detect_intent(message_text)
                profile.detected_intent = detected_intent
                profile.save()
                logger.info(
                    f"Detected and stored intent '{detected_intent}' for profile {profile.id}"
                )
            else:
                # Use previously detected intent or default
                detected_intent = profile.detected_intent or "outro"

            # Generate response based on intent
            response_message = groq_service.generate_intent_response(
                user_message=message_text,
                intent=detected_intent,
                name=profile.name,
                inferred_gender=profile.inferred_gender,
            )

            # Persist assistant response
            Message.objects.create(
                profile=profile, role="assistant", content=response_message
            )
            logger.info(f"Persisted assistant response for profile {profile.id}")

            # Send response to Telegram
            success = telegram_service.send_message(chat_id, response_message)

            if success:
                logger.info(f"Response sent to chat {chat_id}")
            else:
                logger.error(f"Failed to send response to chat {chat_id}")

            return JsonResponse({"status": "ok"}, status=200)

        except Exception as e:
            logger.error(f"Error handling regular message: {str(e)}", exc_info=True)
            return JsonResponse({"status": "error"}, status=500)
