import logging
import random

from django.shortcuts import redirect, render
from django.urls import reverse
from django.views import View
from faker import Faker

from core.models import Message, Profile
from services.ollama_service import OllamaService
from services.simulation_service import SimulationUseCase

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
            return self._handle_simulate(request, ollama_service=OllamaService())

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

        Message.objects.create(
            profile=profile, role="user", content=message_text, channel="chat"
        )

        OllamaService().generate_response_message(profile=profile, channel="chat")

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

    def _handle_simulate(self, request, ollama_service):

        theme = request.POST.get("theme", "").strip().lower()

        profile_id = SimulationUseCase(ollama_service=ollama_service).handle(
            theme_name=theme, num_messages=4
        )

        # Redirect to chat with simulation profile selected
        return redirect(f"{reverse('chat')}?profile_id={profile_id}")

    def _get_conversation_context(
        self, profile, actual_message_id, limit: int = 5
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
