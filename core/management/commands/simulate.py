"""
Django management command to simulate a conversation between two AI agents.

This command creates a conversation simulation between:
- A simulated human seeker (the "buscador")
- The existing WaChat listener bot (the "ouvinte")

Usage:
    python manage.py simulate
    python manage.py simulate --num-messages 10
    python manage.py simulate --quiet
"""

import logging
import os

from django.core.management.base import BaseCommand

from services.simulation_service import SimulationService

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    """
    Management command to simulate a conversation between two AI agents.

    This command uses the SimulationService to generate a realistic conversation
    between ROLE_A (seeker/buscador) and ROLE_B (listener/ouvinte), displays
    it to the console, and provides a critical analysis of the conversation.
    """

    help = "Simulate a conversation between a seeker (pessoa) and listener (BOT)"

    def add_arguments(self, parser):
        """Add command line arguments."""
        parser.add_argument(
            "--num-messages",
            type=int,
            default=8,
            help="Number of messages to generate (default: 8, range: 6-10)",
        )
        parser.add_argument(
            "--quiet",
            action="store_true",
            help="Minimal output (only show the conversation)",
        )

    def handle(self, *args, **options):
        """Execute the simulation."""
        num_messages = options["num_messages"]
        quiet = options["quiet"]

        # Validate num_messages is within bounds
        if num_messages < 6 or num_messages > 10:
            self.stdout.write(
                self.style.WARNING(
                    f"⚠️  num-messages must be between 6 and 10. Using default (8)."
                )
            )
            num_messages = 8

        # Ensure even number for alternating roles
        if num_messages % 2 != 0:
            num_messages += 1
            if not quiet:
                self.stdout.write(
                    self.style.WARNING(
                        f"⚠️  Adjusted to {num_messages} messages for alternating roles."
                    )
                )

        if not quiet:
            self.stdout.write(
                self.style.SUCCESS(
                    f"\n{'=' * 70}\nSimulação de Conversa\n{'=' * 70}"
                )
            )
            self.stdout.write("Gerando diálogo entre um buscador espiritual e um ouvinte empático.")
            self.stdout.write(f"Mensagens: {num_messages}\n")

        try:
            # Validate environment
            groq_api_key = os.environ.get("GROQ_API_KEY")
            if not groq_api_key:
                self.stdout.write(
                    self.style.ERROR(
                        "❌ GROQ_API_KEY não configurado. Configure a variável de ambiente."
                    )
                )
                return

            # Initialize simulation service
            simulation_service = SimulationService(groq_api_key)

            # Step 1: Create simulation profile
            if not quiet:
                self.stdout.write("\n🔄 Criando perfil de simulação...")
            profile = simulation_service.create_simulation_profile()
            if not quiet:
                self.stdout.write(self.style.SUCCESS(f"✓ Perfil criado: {profile.id}"))

            # Step 2: Generate simulated conversation
            if not quiet:
                self.stdout.write(f"\n🔄 Gerando {num_messages} mensagens...")
            conversation = simulation_service.generate_simulated_conversation(
                profile, num_messages
            )
            if not quiet:
                self.stdout.write(
                    self.style.SUCCESS(
                        f"✓ {len(conversation)} mensagens geradas"
                    )
                )

            # Step 3: Display conversation
            self.stdout.write(
                self.style.SUCCESS(
                    f"\n{'=' * 70}\nConversa Simulada\n{'=' * 70}\n"
                )
            )

            for msg in conversation:
                if msg["role"] == "ROLE_A":
                    prefix = "🧑‍💬 Pessoa:"
                    style = self.style.WARNING  # Yellow/orange for seeker
                else:  # ROLE_B
                    prefix = "🌿 BOT:"
                    style = self.style.SUCCESS  # Green for listener

                self.stdout.write(style(prefix))
                self.stdout.write(msg["content"])
                self.stdout.write("")  # Blank line between messages

            # Step 4: Generate critical analysis
            if not quiet:
                self.stdout.write("\n🔄 Gerando análise crítica da conversa...")
            analysis = simulation_service.analyze_conversation_emotions(conversation)

            # Step 5: Display critical analysis
            self.stdout.write(
                self.style.SUCCESS(
                    f"\n{'=' * 70}\n📊 Análise Crítica da Conversa\n{'=' * 70}\n"
                )
            )
            self.stdout.write(analysis)
            self.stdout.write("")

            # Summary
            if not quiet:
                self.stdout.write(
                    self.style.SUCCESS(
                        f"\n{'=' * 70}\nSimulação Completa\n{'=' * 70}"
                    )
                )
                self.stdout.write(f"✓ Perfil ID: {profile.id}")
                self.stdout.write(f"✓ {len(conversation)} mensagens persistidas no banco")
                self.stdout.write("✓ Análise crítica gerada\n")

        except KeyboardInterrupt:
            self.stdout.write(
                self.style.WARNING("\n\n⚠️  Simulação interrompida pelo usuário")
            )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"\n\n❌ Erro na simulação: {str(e)}")
            )
            logger.error(f"Simulation error: {str(e)}", exc_info=True)
            raise
