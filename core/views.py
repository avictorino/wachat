import logging
import random
from urllib.parse import urlencode

from django.shortcuts import redirect, render
from django.urls import reverse
from django.views import View
from faker import Faker

from core.models import Message, Profile
from services.chat_service import ChatService
from services.simulation_service import SimulatedUserProfile, SimulationUseCase

logger = logging.getLogger(__name__)

_faker = Faker("pt_BR")

# Gender constants
GENDER_MALE = "male"
GENDER_FEMALE = "female"

# Constants for simulation timing
MESSAGE_DELAY_SECONDS = 0.6  # Delay between conversation messages
OVERVIEW_DELAY_SECONDS = 1.0  # Delay between overview messages


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
        last_assistant_message_id = None

        if selected_profile_id:
            try:
                selected_profile = Profile.objects.get(id=selected_profile_id)
                # Get messages for this profile
                messages = Message.objects.filter(profile=selected_profile).order_by(
                    "created_at"
                )
                last_assistant_message = (
                    Message.objects.filter(profile=selected_profile, role="assistant")
                    .order_by("-created_at")
                    .first()
                )
                if last_assistant_message:
                    last_assistant_message_id = last_assistant_message.id
            except Profile.DoesNotExist:
                pass
        elif profiles.exists():
            # Select most recent profile by default
            selected_profile = profiles.first()
            messages = Message.objects.filter(profile=selected_profile).order_by(
                "created_at"
            )
            last_assistant_message = (
                Message.objects.filter(profile=selected_profile, role="assistant")
                .order_by("-created_at")
                .first()
            )
            if last_assistant_message:
                last_assistant_message_id = last_assistant_message.id

        context = {
            "profiles": profiles,
            "selected_profile": selected_profile,
            "messages": messages,
            "last_assistant_message_id": last_assistant_message_id,
            "simulated_preview": request.GET.get("simulated_preview", "").strip(),
            "simulated_error": request.GET.get("simulated_error", "").strip(),
            "selected_predefined_scenario": request.GET.get(
                "selected_predefined_scenario", ""
            ).strip(),
            "selected_emotional_profile": request.GET.get(
                "selected_emotional_profile", ""
            ).strip(),
            "selected_simulation_theme": request.GET.get(
                "selected_simulation_theme", ""
            ).strip(),
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
        elif action == "analyze":
            return self._handle_analyze(request)
        elif action == "delete_and_regenerate":
            return self._handle_delete_and_regenerate(request)

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

        pending_sim_payload_key = f"pending_simulation_payload_{profile.id}"
        pending_sim_preview_key = f"pending_simulation_preview_{profile.id}"
        pending_simulation_payload = request.session.get(pending_sim_payload_key)
        pending_simulation_preview = request.session.get(pending_sim_preview_key, "")

        user_prompt_payload = None
        if pending_simulation_payload:
            user_prompt_payload = pending_simulation_payload
            if pending_simulation_preview:
                user_prompt_payload = {
                    "source": "simulation_preview",
                    "preview": pending_simulation_preview,
                    "payload": pending_simulation_payload,
                }
            request.session.pop(pending_sim_payload_key, None)
            request.session.pop(pending_sim_preview_key, None)

        Message.objects.create(
            profile=profile,
            role="user",
            content=message_text,
            channel="chat",
            ollama_prompt=user_prompt_payload,
        )

        ChatService().generate_response_message(profile=profile, channel="chat")

        return redirect(f"{reverse('chat')}?profile_id={profile.id}")

    def _handle_new_profile(self, request):
        """Create new profile and redirect to it."""
        # Randomly choose a gender for the profile
        gender = random.choice([GENDER_MALE, GENDER_FEMALE])

        # Generate a realistic name based on the gender using Faker
        if gender == GENDER_MALE:
            profile_name = _faker.first_name_male()
        else:
            profile_name = _faker.first_name_female()

        profile = Profile.objects.create(name=profile_name, inferred_gender=gender)
        logger.info(
            f"Created new profile: {profile.id} with name: {profile_name}, gender: {gender}"
        )

        # Redirect to chat with new profile selected
        return redirect(f"{reverse('chat')}?profile_id={profile.id}")

    def _handle_simulate(self, request):
        profile_id = request.POST.get("profile_id")
        emotional_profile = request.POST.get(
            "emotional_profile", SimulatedUserProfile.AMBIVALENTE.value
        ).strip()
        predefined_scenario = request.POST.get("predefined_scenario", "").strip()
        simulation_theme = request.POST.get("simulation_theme", "").strip()

        if not profile_id:
            return redirect(reverse("chat"))

        try:
            profile = Profile.objects.get(id=int(profile_id))
        except Profile.DoesNotExist:
            return redirect(reverse("chat"))

        simulation_use_case = SimulationUseCase()
        conversation = (
            Message.objects.filter(profile=profile)
            .exclude(role="system")
            .exclude(role="analysis")
            .exclude(exclude_from_context=True)
            .order_by("created_at")
        )
        try:
            simulation_result = (
                simulation_use_case.simulate_next_user_message_with_metadata(
                    conversation=conversation,
                    profile=emotional_profile,
                    predefined_scenario=predefined_scenario,
                    theme=simulation_theme,
                )
            )
        except (ValueError, RuntimeError) as exc:
            logger.exception(
                (
                    "Simulation failed for profile_id=%s "
                    "(emotional_profile=%s, predefined_scenario=%s, simulation_theme=%s)"
                ),
                profile.id,
                emotional_profile,
                predefined_scenario,
                simulation_theme,
            )
            request.session.pop(f"pending_simulation_payload_{profile.id}", None)
            request.session.pop(f"pending_simulation_preview_{profile.id}", None)
            error_query = urlencode(
                {
                    "profile_id": profile.id,
                    "simulated_error": str(exc),
                    "selected_predefined_scenario": predefined_scenario,
                    "selected_emotional_profile": emotional_profile,
                    "selected_simulation_theme": simulation_theme,
                }
            )
            return redirect(f"{reverse('chat')}?{error_query}")

        simulated_preview = simulation_result.get("content", "").strip()
        request.session[
            f"pending_simulation_payload_{profile.id}"
        ] = simulation_result.get("payload")
        request.session[f"pending_simulation_preview_{profile.id}"] = simulated_preview
        query = urlencode(
            {
                "profile_id": profile.id,
                "simulated_preview": simulated_preview,
                "selected_predefined_scenario": predefined_scenario,
                "selected_emotional_profile": emotional_profile,
                "selected_simulation_theme": simulation_theme,
            }
        )
        return redirect(f"{reverse('chat')}?{query}")

    def _handle_analyze(self, request):
        """Analyze conversation emotions for selected profile."""
        profile_id = request.POST.get("profile_id")

        if not profile_id:
            return redirect(reverse("chat"))

        try:
            profile = Profile.objects.get(id=profile_id)
        except Profile.DoesNotExist:
            return redirect(reverse("chat"))

        # Generate analysis using configured LLM service
        analysis = ChatService().analyze_conversation_emotions(profile=profile)

        logger.info("Generated critical analysis")

        # Save analysis message with exclude_from_context flag
        Message.objects.create(
            profile=profile,
            role="analysis",
            content=f"📊 Análise Crítica da Conversa:\n\n{analysis}",
            channel="other",
            exclude_from_context=True,
        )

        # Redirect back to chat with selected profile
        return redirect(f"{reverse('chat')}?profile_id={profile.id}")

    def _handle_delete_and_regenerate(self, request):
        """Delete selected assistant message and generate a new response."""
        profile_id = request.POST.get("profile_id")
        message_id = request.POST.get("message_id")

        if not profile_id or not message_id:
            return redirect(reverse("chat"))

        try:
            profile = Profile.objects.get(id=int(profile_id))
        except Profile.DoesNotExist:
            return redirect(reverse("chat"))

        try:
            message = Message.objects.get(
                id=int(message_id), profile=profile, role="assistant"
            )
        except Message.DoesNotExist:
            return redirect(f"{reverse('chat')}?profile_id={profile.id}")

        message.delete()
        ChatService().generate_response_message(profile=profile, channel="chat")
        return redirect(f"{reverse('chat')}?profile_id={profile.id}")

    def _get_conversation_context(
        self, profile, actual_message_id, limit: int = 5
    ) -> list:
        """
        Get recent conversation context for the LLM.
        (Reused from TelegramWebhookView)
        """
        recent_messages = (
            Message.objects.filter(profile=profile)
            .for_context()
            .exclude(id=actual_message_id)
            .order_by("-created_at")[:limit]
        )

        context = []
        for msg in reversed(recent_messages):
            context.append({"role": msg.role, "content": msg.content})

        logger.info(f"Retrieved {len(context)} messages for conversation context")
        return context
