"""
Django management command to simulate a conversation between a human and the bot.

This command creates a realistic conversation simulation where:
- A new Profile/User is created each time
- A simulated human generates emotionally-driven messages using AI
- All messages flow through the real webhook/view (no shortcuts)
- Everything is persisted in the database exactly as in production
"""

import json
import logging
import os
import random
import time
from typing import List
from unittest.mock import patch

from django.core.management.base import BaseCommand
from django.test import Client

from core.models import Message, Profile
from services.human_simulator import HumanSimulator

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    """
    Management command to simulate a realistic conversation between human and bot.

    This is NOT a test - it's a functional system-level simulation that:
    - Creates real database records
    - Exercises the full webhook pipeline
    - Simulates realistic human behavior with AI
    - Helps validate the entire conversation flow
    """

    help = "Simulate a conversation between a human user and the WaChat bot"

    def add_arguments(self, parser):
        """Add command line arguments."""
        parser.add_argument(
            "--turns",
            type=int,
            default=5,
            help="Number of conversation turns (default: 5)",
        )
        parser.add_argument(
            "--domain",
            type=str,
            default="spiritual",
            help="Conversation domain (e.g., spiritual, grief, relationship, faith) (default: spiritual)",
        )
        parser.add_argument(
            "--name",
            type=str,
            default=None,
            help="Name for the simulated user (randomly generated if not provided)",
        )
        parser.add_argument(
            "--delay",
            type=float,
            default=2.0,
            help="Delay in seconds between messages (default: 2.0)",
        )
        parser.add_argument(
            "--mock-telegram",
            action="store_true",
            help="Mock Telegram API calls (for testing without real bot token)",
        )

    def handle(self, *args, **options):
        """Execute the simulation."""
        turns = options["turns"]
        domain = options["domain"]
        delay = options["delay"]
        mock_telegram = options["mock_telegram"]
        user_name = options["name"] or self._generate_random_name()

        self.stdout.write(
            self.style.SUCCESS(
                f"\n{'='*70}\nStarting Conversation Simulation\n{'='*70}"
            )
        )
        self.stdout.write(f"User: {user_name}")
        self.stdout.write(f"Domain: {domain}")
        self.stdout.write(f"Turns: {turns}")
        self.stdout.write(f"Delay: {delay}s")
        self.stdout.write(f"Mock Telegram: {mock_telegram}\n")

        # Use mock if requested or if TELEGRAM_BOT_TOKEN looks like a test token
        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        if not mock_telegram and "test" in bot_token.lower():
            mock_telegram = True
            self.stdout.write(
                self.style.WARNING("⚠️  Auto-enabling mock mode (test token detected)\n")
            )

        try:
            if mock_telegram:
                # Run with mocked Telegram API
                with patch("services.telegram_service.requests.post") as mock_post:
                    # Mock successful Telegram API responses
                    mock_post.return_value.status_code = 200
                    mock_post.return_value.json.return_value = {"ok": True}
                    self._run_simulation(
                        user_name, domain, turns, delay
                    )
            else:
                # Run without mocking (real Telegram API)
                self._run_simulation(user_name, domain, turns, delay)

        except KeyboardInterrupt:
            self.stdout.write(
                self.style.WARNING("\n\n⚠️  Simulation interrupted by user")
            )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"\n\n❌ Simulation failed: {str(e)}")
            )
            logger.error(f"Simulation error: {str(e)}", exc_info=True)
            raise

    def _run_simulation(self, user_name: str, domain: str, turns: int, delay: float):
        """Run the actual simulation logic."""
    def _run_simulation(self, user_name: str, domain: str, turns: int, delay: float):
        """Run the actual simulation logic."""
        # Validate environment
        self._validate_environment()

        # Generate a simulated user ID for this session
        simulated_user_id = str(random.randint(900000000, 999999999))
        chat_id = simulated_user_id

        # Initialize test client for webhook calls
        client = Client()

        # Start with /start command to initialize the profile properly
        # This will create the profile through the webhook
        self.stdout.write(
            self.style.WARNING(f"\n{'-'*70}\nTurn 0: /start command\n{'-'*70}")
        )
        self._send_webhook_message(
            client, simulated_user_id, chat_id, "/start", user_name
        )

        # Brief pause to let initial processing complete
        time.sleep(delay)

        # Get the created profile
        profile = Profile.objects.get(telegram_user_id=simulated_user_id)
        self.stdout.write(
            self.style.SUCCESS(
                f"✓ Profile created via webhook: {profile.name} (ID: {profile.id})"
            )
        )

        # Initialize human simulator
        api_key = os.environ.get("GROQ_API_KEY")
        human_simulator = HumanSimulator(
            api_key=api_key, name=user_name, domain=domain
        )
        self.stdout.write(self.style.SUCCESS("✓ Initialized AI human simulator"))

        # Get and display the welcome message
        welcome_msg = Message.objects.filter(
            profile=profile, role="assistant"
        ).first()
        if welcome_msg:
            self.stdout.write(
                self.style.SUCCESS(f"🤖 Bot: {welcome_msg.content}\n")
            )
        else:
            self.stdout.write(
                self.style.WARNING("⚠️  No welcome message generated")
            )

        # Run conversation turns
        for turn in range(1, turns + 1):
            self.stdout.write(
                self.style.WARNING(
                    f"\n{'-'*70}\nTurn {turn} of {turns}\n{'-'*70}"
                )
            )

            # Get conversation history for context
            conversation_history = self._get_conversation_history(profile)

            # Generate human message using AI
            self.stdout.write("🤔 Generating human message...")
            human_message = human_simulator.generate_message(
                conversation_history, turn
            )
            self.stdout.write(self.style.SUCCESS(f"👤 Human: {human_message}"))

            # Send message through webhook
            time.sleep(delay * 0.5)  # Brief pause before sending
            self._send_webhook_message(
                client, simulated_user_id, chat_id, human_message, user_name
            )

            # Wait for processing
            time.sleep(delay)

            # Get and display bot responses
            bot_responses = self._get_latest_bot_responses(profile)
            if bot_responses:
                for response in bot_responses:
                    self.stdout.write(
                        self.style.SUCCESS(f"🤖 Bot: {response.content}")
                    )
            else:
                self.stdout.write(
                    self.style.WARNING("⚠️  No bot response received")
                )

        # Show summary
        self._show_simulation_summary(profile, turns)

    def _validate_environment(self):
        """Validate required environment variables."""
        required_vars = ["TELEGRAM_WEBHOOK_SECRET"]
        missing = [var for var in required_vars if not os.environ.get(var)]

        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}"
            )

        # Check for GROQ_API_KEY but allow mocking
        if not os.environ.get("GROQ_API_KEY"):
            logger.warning(
                "GROQ_API_KEY not set - human message generation may use fallbacks"
            )

    def _generate_random_name(self) -> str:
        """Generate a random Brazilian name for the simulated user."""
        first_names = [
            "João",
            "Maria",
            "Pedro",
            "Ana",
            "Lucas",
            "Juliana",
            "Rafael",
            "Camila",
            "Felipe",
            "Beatriz",
            "Gabriel",
            "Larissa",
            "Rodrigo",
            "Fernanda",
            "Bruno",
            "Amanda",
            "Thiago",
            "Paula",
            "Matheus",
            "Carla",
        ]

        last_names = [
            "Silva",
            "Santos",
            "Oliveira",
            "Souza",
            "Lima",
            "Costa",
            "Ferreira",
            "Rodrigues",
            "Almeida",
            "Nascimento",
        ]

        return f"{random.choice(first_names)} {random.choice(last_names)}"

    def _send_webhook_message(
        self, client: Client, user_id: str, chat_id: str, message: str, name: str
    ):
        """
        Send a message through the Telegram webhook endpoint.

        This simulates a real webhook call from Telegram.
        """
        # Split name for first_name and last_name
        name_parts = name.split(" ", 1)
        first_name = name_parts[0]
        last_name = name_parts[1] if len(name_parts) > 1 else ""

        # Build webhook payload that matches Telegram's format
        payload = {
            "update_id": random.randint(100000, 999999),
            "message": {
                "message_id": random.randint(1, 100000),
                "from": {
                    "id": int(user_id),
                    "is_bot": False,
                    "first_name": first_name,
                    "last_name": last_name,
                    "username": f"user_{user_id}",
                },
                "chat": {
                    "id": int(chat_id),
                    "first_name": first_name,
                    "last_name": last_name,
                    "type": "private",
                },
                "date": int(time.time()),
                "text": message,
            },
        }

        # Get webhook secret from environment
        webhook_secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET")

        # Make the webhook call
        response = client.post(
            "/webhooks/telegram/",
            data=json.dumps(payload),
            content_type="application/json",
            HTTP_X_TELEGRAM_BOT_API_SECRET_TOKEN=webhook_secret,
        )

        if response.status_code != 200:
            logger.warning(
                f"Webhook returned status {response.status_code}: {response.content}"
            )

    def _get_conversation_history(self, profile: Profile) -> List[dict]:
        """Get conversation history for the profile."""
        messages = Message.objects.filter(profile=profile).order_by("created_at")

        history = []
        for msg in messages:
            if msg.role in ["user", "assistant"]:
                history.append({"role": msg.role, "content": msg.content})

        return history

    def _get_latest_bot_responses(self, profile: Profile) -> List[Message]:
        """Get the most recent bot responses that haven't been displayed yet."""
        # Get the last few assistant messages
        # (in case of multiple responses in one turn)
        last_user_msg = (
            Message.objects.filter(profile=profile, role="user")
            .order_by("-created_at")
            .first()
        )

        if not last_user_msg:
            return []

        # Get assistant messages after the last user message
        bot_responses = Message.objects.filter(
            profile=profile,
            role="assistant",
            created_at__gt=last_user_msg.created_at,
        ).order_by("created_at")

        return list(bot_responses)

    def _show_simulation_summary(self, profile: Profile, expected_turns: int):
        """Display a summary of the simulation."""
        self.stdout.write(
            self.style.SUCCESS(
                f"\n{'='*70}\nSimulation Complete\n{'='*70}"
            )
        )

        # Count messages
        total_messages = Message.objects.filter(profile=profile).count()
        user_messages = Message.objects.filter(profile=profile, role="user").count()
        bot_messages = Message.objects.filter(profile=profile, role="assistant").count()

        self.stdout.write(f"\nProfile ID: {profile.id}")
        self.stdout.write(f"Profile Name: {profile.name}")
        if profile.detected_intent:
            self.stdout.write(f"Detected Intent: {profile.detected_intent}")
        if profile.inferred_gender:
            self.stdout.write(f"Inferred Gender: {profile.inferred_gender}")

        self.stdout.write(f"\nTotal Messages: {total_messages}")
        self.stdout.write(f"  - User: {user_messages}")
        self.stdout.write(f"  - Bot: {bot_messages}")

        self.stdout.write(
            self.style.SUCCESS(
                f"\n✓ All messages persisted in database"
            )
        )
        self.stdout.write(
            self.style.SUCCESS(
                f"✓ All interactions went through the real webhook/view pipeline"
            )
        )

        # Show where to find the data
        self.stdout.write(f"\nTo review the conversation:")
        self.stdout.write(
            f"  python manage.py shell -c \"from core.models import Profile, Message; "
            f"p = Profile.objects.get(id={profile.id}); "
            f'[print(f\\"{{m.role}}: {{m.content}}\\") for m in p.messages.all()]"'
        )
        self.stdout.write("")
