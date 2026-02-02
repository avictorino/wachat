import json
import logging
import os
import time
from datetime import timedelta

from django.http import JsonResponse
from django.utils import timezone
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt

from core.models import Message, Profile
from services.groq_service import GroqService
from services.reset_user_data import ResetUserDataUseCase
from services.simulation_service import SimulationService
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

        # Handle /reset command
        if message_text and message_text.strip().startswith("/reset"):
            return self._handle_reset_command(sender_id, chat_id)

        # Handle /simulate command
        if message_text and message_text.strip().startswith("/simulate"):
            # Parse optional num_messages parameter
            parts = message_text.strip().split()
            num_messages = None
            if len(parts) > 1:
                try:
                    num_messages = int(parts[1])
                except ValueError:
                    pass  # Will use default
            return self._handle_simulate_command(chat_id, num_messages)

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
                profile=profile,
                role="assistant",
                content=welcome_message,
                channel="telegram",
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

    def _handle_reset_command(self, sender_id: str, chat_id: str):
        """
        Handle the /reset command for permanently deleting user data.

        This method implements a two-step confirmation process:
        1. On first /reset: asks for confirmation and sets pending state
        2. Awaits "CONFIRM" response to proceed with deletion

        Args:
            sender_id: Telegram user ID
            chat_id: Telegram chat ID

        Returns:
            JsonResponse indicating success
        """
        try:
            telegram_service = TelegramService()

            # Try to get user profile
            try:
                profile = Profile.objects.get(telegram_user_id=sender_id)
            except Profile.DoesNotExist:
                # User doesn't exist, nothing to reset
                message = (
                    "You don't have any data in our system yet. Use /start to begin."
                )
                telegram_service.send_message(chat_id, message)
                logger.info(f"Reset attempted for non-existent user: {sender_id}")
                return JsonResponse({"status": "ok"}, status=200)

            # Set confirmation pending state
            profile.pending_reset_confirmation = True
            profile.reset_confirmation_timestamp = timezone.now()
            profile.save()

            logger.info(
                f"Reset initiated for profile {profile.id}. Waiting for confirmation."
            )

            # Send confirmation request
            confirmation_message = (
                "⚠️ *WARNING: This action cannot be undone!*\n\n"
                "This will permanently delete:\n"
                "• Your profile\n"
                "• All your conversations\n"
                "• All your messages\n"
                "• All your preferences\n\n"
                "Type *CONFIRM* to proceed with deletion, "
                "or send any other message to cancel."
            )

            telegram_service.send_message(
                chat_id, confirmation_message, parse_mode="Markdown"
            )
            logger.info(f"Confirmation message sent to chat {chat_id}")

            return JsonResponse({"status": "ok"}, status=200)

        except Exception as e:
            logger.error(f"Error handling /reset command: {str(e)}", exc_info=True)
            return JsonResponse({"status": "error"}, status=500)

    def _handle_simulate_command(self, chat_id: str, num_messages: int = None):
        """
        Handle the /simulate command to run a conversation simulation.

        This method:
        1. Creates a new simulation profile
        2. Generates a simulated conversation between ROLE_A (seeker) and ROLE_B (listener)
        3. Persists all messages to the database
        4. Sends the conversation transcript back to Telegram with role identification
        5. Analyzes the conversation emotionally using LLM
        6. Sends the emotional analysis as a final summary message

        Args:
            chat_id: Telegram chat ID
            num_messages: Optional number of messages to generate (6-10, default 8)

        Returns:
            JsonResponse indicating success
        """
        try:
            logger.info(f"Starting /simulate command for chat {chat_id}")

            # Initialize services
            telegram_service = TelegramService()
            groq_api_key = os.environ.get("GROQ_API_KEY")

            if not groq_api_key:
                error_msg = "Simulation service is not available at the moment."
                telegram_service.send_message(chat_id, error_msg)
                logger.error("GROQ_API_KEY not configured for simulation")
                return JsonResponse({"status": "ok"}, status=200)

            simulation_service = SimulationService(groq_api_key)

            # Validate and set num_messages parameter
            if num_messages is None:
                num_messages = 8  # Default value
            elif num_messages < 6 or num_messages > 10:
                error_msg = "❌ Número de mensagens inválido. Use um valor entre 6 e 10.\n\nExemplo: /simulate 8"
                telegram_service.send_message(chat_id, error_msg)
                logger.warning(f"Invalid num_messages parameter: {num_messages}")
                return JsonResponse({"status": "ok"}, status=200)

            # Send initial message
            init_msg = f"🔄 Iniciando simulação de conversa...\n\nGerando {num_messages} mensagens de diálogo entre um buscador espiritual e um ouvinte empático."
            telegram_service.send_message(chat_id, init_msg)

            # Step 1: Create new profile for simulation
            profile = simulation_service.create_simulation_profile()
            logger.info(f"Created simulation profile {profile.id}")

            # Step 2: Generate simulated conversation with specified num_messages
            conversation = simulation_service.generate_simulated_conversation(
                profile, num_messages
            )
            logger.info(f"Generated {len(conversation)} simulated messages")

            # Step 3: Send each message back to Telegram with role prefixes
            for msg in conversation:
                if msg["role"] == "ROLE_A":
                    prefix = "🧑‍💬 Buscador:"
                else:  # ROLE_B
                    prefix = "🌿 Ouvinte:"

                formatted_msg = f"{prefix}\n{msg['content']}"
                telegram_service.send_message(chat_id, formatted_msg)

                # Small pause between messages for readability
                time.sleep(0.8)

            logger.info(f"Sent {len(conversation)} simulated messages to chat {chat_id}")

            # Step 4: Generate critical analysis of conversation
            analysis = simulation_service.analyze_conversation_emotions(conversation)
            logger.info("Generated critical analysis")

            # Step 5: Send critical analysis as final message
            final_msg = f"📊 *Análise Crítica da Conversa*\n\n{analysis}"
            telegram_service.send_message(chat_id, final_msg, parse_mode="Markdown")
            logger.info(f"Sent critical analysis to chat {chat_id}")

            return JsonResponse({"status": "ok"}, status=200)

        except Exception as e:
            logger.error(f"Error handling /simulate command: {str(e)}", exc_info=True)
            # Try to send error message to user
            try:
                telegram_service = TelegramService()
                error_msg = "❌ Erro ao executar a simulação. Por favor, tente novamente mais tarde."
                telegram_service.send_message(chat_id, error_msg)
            except Exception:
                # If we can't send the error message, log and continue
                logger.error("Failed to send error message to user")
            return JsonResponse({"status": "error"}, status=500)

    def _handle_reset_confirmation(
        self,
        profile: Profile,
        sender_id: str,
        chat_id: str,
        message_text: str,
        telegram_service: TelegramService,
    ):
        """
        Handle user's response to reset confirmation request.

        Checks if the user typed "CONFIRM" to proceed with deletion,
        or cancels the reset if they send anything else.

        Args:
            profile: The user's Profile object
            sender_id: Telegram user ID
            chat_id: Telegram chat ID
            message_text: The user's message (should be "CONFIRM" or something else)
            telegram_service: TelegramService instance for sending messages

        Returns:
            JsonResponse indicating success
        """
        try:
            # Check for timeout (5 minutes)
            if profile.reset_confirmation_timestamp:
                timeout_minutes = 5
                elapsed = timezone.now() - profile.reset_confirmation_timestamp
                if elapsed > timedelta(minutes=timeout_minutes):
                    # Timeout expired, cancel reset
                    profile.pending_reset_confirmation = False
                    profile.reset_confirmation_timestamp = None
                    profile.save()

                    message = (
                        "Reset confirmation timeout expired. "
                        "Please use /reset again if you still want to delete your data."
                    )
                    telegram_service.send_message(chat_id, message)
                    logger.info(f"Reset confirmation timeout for profile {profile.id}")

                    # Persist user message and continue with regular flow
                    Message.objects.create(
                        profile=profile, role="user", content=message_text
                    )
                    return JsonResponse({"status": "ok"}, status=200)

            # Check if user confirmed
            if message_text.strip().upper() == "CONFIRM":
                logger.info(
                    f"User {sender_id} confirmed reset. Proceeding with deletion."
                )

                # Execute the reset
                success = ResetUserDataUseCase.execute(sender_id)

                if success:
                    # Send final confirmation message
                    final_message = (
                        "✅ Your data has been successfully deleted. "
                        "You can start over anytime by using /start."
                    )
                    telegram_service.send_message(chat_id, final_message)
                    logger.info(f"Successfully reset data for user {sender_id}")
                else:
                    # This shouldn't happen since we already checked profile exists
                    message = "An error occurred. Please try again later."
                    telegram_service.send_message(chat_id, message)
                    logger.error(
                        f"Reset failed for user {sender_id} despite having profile"
                    )

            else:
                # User cancelled
                profile.pending_reset_confirmation = False
                profile.reset_confirmation_timestamp = None
                profile.save()

                message = "Reset cancelled. Your data is safe."
                telegram_service.send_message(chat_id, message)
                logger.info(f"User {sender_id} cancelled reset")

                # Persist the cancellation message
                Message.objects.create(
                    profile=profile, role="user", content=message_text
                )

            return JsonResponse({"status": "ok"}, status=200)

        except Exception as e:
            logger.error(f"Error handling reset confirmation: {str(e)}", exc_info=True)
            return JsonResponse({"status": "error"}, status=500)

    def _handle_regular_message(
        self, sender: dict, sender_id: str, chat_id: str, message_text: str
    ):
        """
        Handle regular text messages from users.

        This method implements the conversational flow after the welcome message:
        1. Retrieves or creates user profile
        2. Persists the user's message
        3. Detects intent from the message (if not already detected)
        4. Generates a response using Groq:
           - If intent is clear and matches a category: uses intent-based response
           - If intent is "outro" (unclear/ambiguous): uses context-aware fallback
        5. Persists the assistant's response(s)
        6. Sends the response(s) back to the user via Telegram

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

            # Initialize services
            telegram_service = TelegramService()

            # Check if user has pending reset confirmation
            if profile.pending_reset_confirmation:
                return self._handle_reset_confirmation(
                    profile, sender_id, chat_id, message_text, telegram_service
                )

            # Persist user message
            Message.objects.create(profile=profile, role="user", content=message_text)
            logger.info(f"Persisted user message for profile {profile.id}")

            # Initialize Groq service for intent detection and response generation
            groq_service = GroqService()

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

            # Choose response generation strategy based on intent
            if detected_intent == "outro":
                # Use context-aware fallback for ambiguous/unclear intent
                logger.info(
                    f"Using fallback conversational flow for profile {profile.id}"
                )

                # Get conversation context (last 8 messages for continuity)
                context = self._get_conversation_context(profile, limit=8)

                # Generate fallback response (may return multiple messages)
                response_messages = groq_service.generate_fallback_response(
                    user_message=message_text,
                    conversation_context=context,
                    name=profile.name,
                    inferred_gender=profile.inferred_gender,
                )

                # Persist each assistant response separately
                for response_msg in response_messages:
                    Message.objects.create(
                        profile=profile, role="assistant", content=response_msg
                    )
                logger.info(
                    f"Persisted {len(response_messages)} fallback response(s) for profile {profile.id}"
                )

                # Send all responses sequentially with pauses
                success = telegram_service.send_messages(
                    chat_id, response_messages, pause_seconds=1.5
                )

            else:
                # Use intent-based response for clear intent
                response_messages = groq_service.generate_intent_response(
                    user_message=message_text,
                    intent=detected_intent,
                    name=profile.name,
                    inferred_gender=profile.inferred_gender,
                )

                # Persist each assistant response separately
                for response_msg in response_messages:
                    Message.objects.create(
                        profile=profile, role="assistant", content=response_msg
                    )
                logger.info(
                    f"Persisted {len(response_messages)} intent-based response(s) for profile {profile.id}"
                )

                # Send all responses sequentially with pauses
                success = telegram_service.send_messages(
                    chat_id, response_messages, pause_seconds=1.5
                )

            if success:
                logger.info(f"Response sent to chat {chat_id}")
            else:
                logger.error(f"Failed to send response to chat {chat_id}")

            return JsonResponse({"status": "ok"}, status=200)

        except Exception as e:
            logger.error(f"Error handling regular message: {str(e)}", exc_info=True)
            return JsonResponse({"status": "error"}, status=500)

    def _get_conversation_context(self, profile, limit: int = 8) -> list:
        """
        Get recent conversation context for the LLM.

        Retrieves the last N messages (both user and assistant) to provide
        context for generating contextually-aware responses.

        Args:
            profile: The user Profile object
            limit: Maximum number of recent messages to retrieve

        Returns:
            List of dicts with 'role' and 'content' keys
        """
        # Get the most recent messages (excluding system messages)
        recent_messages = (
            Message.objects.filter(profile=profile)
            .exclude(role="system")
            .order_by("-created_at")[:limit]
        )

        # Reverse to get chronological order (oldest to newest)
        context = []
        for msg in reversed(recent_messages):
            context.append({"role": msg.role, "content": msg.content})

        logger.info(f"Retrieved {len(context)} messages for conversation context")
        return context
