import json
import logging
import os
import time
import uuid
from datetime import timedelta

from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils import timezone
from django.utils.decorators import method_decorator
from django.views import View
from django.views.decorators.csrf import csrf_exempt

from core.models import Message, Profile
from services.llm_factory import get_llm_service
from services.message_splitter import split_welcome_message
from services.reset_user_data import ResetUserDataUseCase
from services.simulation_service import SimulationService
from services.telegram_service import TelegramService

logger = logging.getLogger(__name__)

# Constants for simulation timing
MESSAGE_DELAY_SECONDS = 0.6  # Delay between conversation messages
OVERVIEW_DELAY_SECONDS = 1.0  # Delay between overview messages


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
            # Parse theme and optional num_messages parameter
            # Format: /simulate drogas [num_messages]
            parts = message_text.strip().split()
            theme = None
            num_messages = None
            if len(parts) > 1:
                theme = parts[1].lower()
            if len(parts) > 2:
                try:
                    num_messages = int(parts[2])
                except ValueError:
                    # Invalid number format - will use default
                    logger.warning(
                        f"Invalid num_messages format: {parts[2]}, using default"
                    )
            return self._handle_simulate_command(chat_id, theme, num_messages)

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
        3. Infers gender using LLM
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
            llm_service = get_llm_service()
            telegram_service = TelegramService()

            # Infer gender if not already done
            if not profile.inferred_gender:
                inferred_gender = llm_service.infer_gender(name)
                profile.inferred_gender = inferred_gender
                needs_save = True
                logger.info(f"Inferred gender for {name}: {inferred_gender}")

            # Save profile if needed (consolidate all updates into one save)
            if needs_save:
                profile.save()

            # Generate welcome message
            welcome_message = llm_service.generate_welcome_message(
                name=name, inferred_gender=profile.inferred_gender
            )

            # Split the welcome message into greeting and question parts
            greeting_part, question_part = split_welcome_message(welcome_message)

            # Create list of messages to send
            messages_to_send = []
            if greeting_part:
                messages_to_send.append(greeting_part)
            if question_part:
                messages_to_send.append(question_part)

            # If split didn't produce 2 messages, send original message as fallback
            if len(messages_to_send) < 2:
                messages_to_send = [welcome_message]

            # Persist each message part
            for message_content in messages_to_send:
                Message.objects.create(
                    profile=profile,
                    role="assistant",
                    content=message_content,
                    channel="telegram",
                )
            logger.info(
                f"Persisted {len(messages_to_send)} welcome message(s) for profile {profile.id}"
            )

            # Send the message(s) to Telegram
            success = telegram_service.send_messages(chat_id, messages_to_send)

            if success:
                logger.info(f"Welcome message(s) sent to chat {chat_id}")
            else:
                logger.error(f"Failed to send welcome message(s) to chat {chat_id}")

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

    def _handle_simulate_command(
        self, chat_id: str, theme: str = None, num_messages: int = None
    ):
        """
        Handle the /simulate command to run a conversation simulation.

        For 'drogas' theme:
        - No longer supported

        For other themes:
        - Uses the existing single-LLM simulation approach
        - Creates a new simulation profile with the specified theme
        - Generates a simulated conversation between ROLE_A (seeker) and ROLE_B (listener)
        - Persists all messages to the database
        - Sends the conversation transcript back to Telegram with role identification
        - Analyzes the conversation emotionally using LLM
        - Sends the emotional analysis as a final summary message

        Args:
            chat_id: Telegram chat ID
            theme: Optional theme for the conversation (e.g., "doenca", "drogas")
            num_messages: Optional number of messages (only for 'drogas' theme, default: 20)

        Returns:
            JsonResponse indicating success
        """
        try:
            # Special handling for 'drogas' theme - no longer supported
            if theme == "drogas":
                telegram_service = TelegramService()
                error_msg = (
                    "❌ A simulação de dependência química não está mais disponível."
                )
                telegram_service.send_message(chat_id, error_msg)
                return JsonResponse({"status": "ok"}, status=200)

            # Original simulation logic for other themes
            logger.info(
                f"Starting /simulate command for chat {chat_id} with theme: {theme}"
            )

            # Initialize services
            telegram_service = TelegramService()

            if not os.environ.get("LLM_PROVIDER"):
                error_msg = "Simulation service is not available at the moment."
                telegram_service.send_message(chat_id, error_msg)
                logger.error("LLM_PROVIDER not configured for simulation")
                return JsonResponse({"status": "ok"}, status=200)

            simulation_service = SimulationService()
            llm_service = get_llm_service()

            # Approximate theme using LLM if provided
            if theme:
                original_theme = theme
                theme = theme.lower()

                # Use LLM to approximate the theme to one of the valid categories
                approximated_theme = llm_service.approximate_theme(theme)

                if approximated_theme == "outro":
                    # Theme couldn't be clearly mapped
                    error_msg = f"❌ Não consegui identificar o tema '{original_theme}'.\n\nExemplos de temas válidos:\n- doenca / enfermidade\n- ansiedade / medo\n- pecado / culpa\n- desabafar / solidão\n- financeiro / dinheiro\n- religiao / fé\n- redes_sociais\n- drogas (usa simulação especial)\n\nTente usar uma palavra relacionada a esses temas."
                    telegram_service.send_message(chat_id, error_msg)
                    logger.warning(f"Could not approximate theme: {original_theme}")
                    return JsonResponse({"status": "ok"}, status=200)

                theme = approximated_theme
                logger.info(f"Theme approximated: '{original_theme}' -> '{theme}'")
            else:
                # Default theme if none provided
                theme = "desabafar"

            # Send initial message
            init_msg = f"🔄 Iniciando simulação de conversa...\n\nGerando diálogo sobre tema: *{theme}*"
            telegram_service.send_message(chat_id, init_msg, parse_mode="Markdown")

            # Step 1: Create new profile for simulation with theme
            profile = simulation_service.create_simulation_profile(theme)
            logger.info(f"Created simulation profile {profile.id} with theme: {theme}")

            # Step 2: Generate simulated conversation with theme context
            num_messages_default = 8  # Fixed number of messages for simplicity
            conversation = simulation_service.generate_simulated_conversation(
                profile, num_messages_default, theme
            )
            logger.info(f"Generated {len(conversation)} simulated messages")

            # Step 3: Send each message back to Telegram with role prefixes
            for msg in conversation:
                if msg["role"] == "ROLE_A":
                    prefix = "🧑‍💬 Pessoa:"
                else:  # ROLE_B
                    prefix = "🌿 BOT:"

                formatted_msg = f"{prefix}\n{msg['content']}"
                telegram_service.send_message(chat_id, formatted_msg)

                # Small pause between messages for readability
                time.sleep(0.8)

            logger.info(
                f"Sent {len(conversation)} simulated messages to chat {chat_id}"
            )

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
        4. Generates a response using LLM:
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
            actual_message = Message.objects.create(
                profile=profile, role="user", content=message_text
            )
            logger.info(f"Persisted user message for profile {profile.id}")

            # Initialize LLM service for response generation
            llm_service = get_llm_service()

            # Use context-aware conversational flow for response generation
            logger.info(f"Using conversational flow for profile {profile.id}")

            # Get conversation context (last 10 messages for continuity)
            context = self._get_conversation_context(
                profile, actual_message_id=actual_message.id, limit=10
            )

            # Generate response (may return multiple messages)
            # theme_id is passed for logging/analytics only
            response_messages = llm_service.generate_intent_response(
                user_message=message_text,
                conversation_context=context,
                name=profile.name,
                inferred_gender=profile.inferred_gender,
                intent=None,
                theme_id=profile.prompt_theme,
            )

            # Capture and save Ollama prompt payload for observability
            if hasattr(llm_service, 'get_last_prompt_payload'):
                prompt_payload = llm_service.get_last_prompt_payload()
                if prompt_payload and isinstance(prompt_payload, dict):
                    actual_message.ollama_prompt = prompt_payload
                    actual_message.save(update_fields=['ollama_prompt'])
                    logger.info(f"Saved Ollama prompt payload to message {actual_message.id}")

            # Persist each assistant response separately
            for response_msg in response_messages:
                Message.objects.create(
                    profile=profile, role="assistant", content=response_msg
                )
            logger.info(
                f"Persisted {len(response_messages)} response(s) for profile {profile.id}"
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

    def _get_conversation_context(
        self, profile, actual_message_id: int, limit: int = 10
    ) -> list:
        """
        Get recent conversation context for the LLM.

        Retrieves the last N messages (both user and assistant) to provide
        context for generating contextually-aware responses.

        Following the problem statement requirements:
        - Maximum 10 messages (5 user + 5 assistant)
        - Ordered from oldest → newest
        - This is the AI's only memory

        Args:
            profile: The user Profile object
            limit: Maximum number of recent messages to retrieve (default: 10)

        Returns:
            List of dicts with 'role' and 'content' keys
        """
        # Get the most recent messages (excluding system messages)
        recent_messages = (
            Message.objects.filter(profile=profile)
            .exclude(role="system")
            .exclude(
                id=actual_message_id
            )  # Exclude the current message being processed
            .order_by("-created_at")[:limit]
        )

        # Reverse to get chronological order (oldest to newest)
        context = []
        for msg in reversed(recent_messages):
            context.append({"role": msg.role, "content": msg.content})

        logger.info(f"Retrieved {len(context)} messages for conversation context")
        return context


class ChatView(View):
    """
    Single-page chat simulation UI for WhatsApp/Telegram-style conversations.

    Handles:
    - GET: Display chat interface with messages from selected profile
    - POST: Send message, create new profile, or run simulation
    """

    def get(self, request):
        """Render chat interface with messages for selected profile."""
        # Get selected profile ID from query params
        selected_profile_id = request.GET.get("profile_id")

        # Get all profiles for dropdown
        profiles = Profile.objects.all().order_by("-created_at")

        # Select profile
        selected_profile = None
        messages = []

        if selected_profile_id:
            try:
                selected_profile = Profile.objects.get(id=selected_profile_id)
                # Get messages for this profile
                messages = Message.objects.filter(profile=selected_profile).order_by(
                    "created_at"
                )
            except Profile.DoesNotExist:
                pass
        elif profiles.exists():
            # Select most recent profile by default
            selected_profile = profiles.first()
            messages = Message.objects.filter(profile=selected_profile).order_by(
                "created_at"
            )

        context = {
            "profiles": profiles,
            "selected_profile": selected_profile,
            "messages": messages,
        }

        return render(request, "chat.html", context)

    def post(self, request):
        """Handle POST actions: send message, create profile, or simulate."""
        action = request.POST.get("action")

        if action == "send_message":
            return self._handle_send_message(request)
        elif action == "new_profile":
            return self._handle_new_profile(request)
        elif action == "simulate":
            return self._handle_simulate(request)

        # Default: redirect to GET
        return redirect("chat")

    def _handle_send_message(self, request):
        """Send user message and get LLM response."""
        profile_id = request.POST.get("profile_id")
        message_text = request.POST.get("message_text", "").strip()

        if not profile_id or not message_text:
            # If we have profile_id, redirect back to it; otherwise to main chat
            if profile_id:
                return redirect(f"{reverse('chat')}?profile_id={profile_id}")
            return redirect(reverse("chat"))

        try:
            profile = Profile.objects.get(id=profile_id)
        except Profile.DoesNotExist:
            return redirect(reverse("chat"))

        # Save user message
        actual_message = Message.objects.create(
            profile=profile, role="user", content=message_text, channel="other"
        )
        logger.info(f"User message saved for profile {profile.id}")

        # Get LLM service
        llm_service = get_llm_service()

        # Get conversation context
        context = self._get_conversation_context(
            profile, actual_message_id=actual_message.id, limit=10
        )

        # Generate LLM response
        # theme_id is passed for logging/analytics only
        response_messages = llm_service.generate_intent_response(
            user_message=message_text,
            conversation_context=context,
            name=profile.name,
            inferred_gender=profile.inferred_gender,
            intent=None,
            theme_id=profile.prompt_theme,
        )

        # Save assistant responses
        for response_msg in response_messages:
            Message.objects.create(
                profile=profile, role="assistant", content=response_msg, channel="other"
            )
        logger.info(f"Assistant responses saved for profile {profile.id}")

        # Redirect back to chat with selected profile
        return redirect(f"{reverse('chat')}?profile_id={profile.id}")

    def _handle_new_profile(self, request):
        """Create new profile and redirect to it."""
        # Create a new profile with a generated name
        profile_name = f"User_{uuid.uuid4().hex[:8]}"

        profile = Profile.objects.create(name=profile_name, inferred_gender="unknown")
        logger.info(f"Created new profile: {profile.id}")

        # Redirect to chat with new profile selected
        return redirect(f"{reverse('chat')}?profile_id={profile.id}")

    def _handle_simulate(self, request):
        """Run conversation simulation (reuse _handle_simulate_command logic)."""
        # Get theme from request (optional)
        theme = request.POST.get("theme", "").strip().lower()

        if not theme:
            theme = "desabafar"

        logger.info(f"Starting simulation with theme: {theme}")

        # Initialize services
        simulation_service = SimulationService()
        llm_service = get_llm_service()

        # Approximate theme using LLM if provided
        if theme:
            approximated_theme = llm_service.approximate_theme(theme)
            if approximated_theme == "outro":
                # Use default theme if couldn't approximate
                theme = "desabafar"
            else:
                theme = approximated_theme

        # Create new simulation profile
        profile = simulation_service.create_simulation_profile(theme)
        logger.info(f"Created simulation profile {profile.id} with theme: {theme}")

        # Generate simulated conversation
        num_messages = 8
        conversation = simulation_service.generate_simulated_conversation(
            profile, num_messages, theme
        )
        logger.info(f"Generated {len(conversation)} simulated messages")

        # Generate critical analysis
        analysis = simulation_service.analyze_conversation_emotions(conversation)
        logger.info("Generated critical analysis")

        # Save analysis as a system message
        Message.objects.create(
            profile=profile,
            role="system",
            content=f"📊 Análise Crítica da Conversa:\n\n{analysis}",
            channel="other",
        )

        # Redirect to chat with simulation profile selected
        return redirect(f"{reverse('chat')}?profile_id={profile.id}")

    def _get_conversation_context(
        self, profile, actual_message_id, limit: int = 10
    ) -> list:
        """
        Get recent conversation context for the LLM.
        (Reused from TelegramWebhookView)
        """
        recent_messages = (
            Message.objects.filter(profile=profile)
            .exclude(role="system")
            .exclude(id=actual_message_id)
            .order_by("-created_at")[:limit]
        )

        context = []
        for msg in reversed(recent_messages):
            context.append({"role": msg.role, "content": msg.content})

        logger.info(f"Retrieved {len(context)} messages for conversation context")
        return context
