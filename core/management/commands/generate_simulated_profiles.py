import logging
import random

from django.core.management.base import BaseCommand
from django.db.models import Q
from faker import Faker

from core.models import Message, Profile, Theme
from services.chat_service import ChatService
from services.simulation_service import (
    PREDEFINED_SCENARIOS,
    SimulatedUserProfile,
    SimulationUseCase,
)

_faker = Faker("pt_BR")

GENDER_MALE = "male"
GENDER_FEMALE = "female"
logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = (
        "Cria N novos perfis e gera conversas simuladas com parametros aleatorios "
        "de 'como eu me sinto', 'o que eu quero' e 'meu problema'. "
        "Cada perfil recebe analise final da conversa."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--count",
            type=int,
            default=5,
            help="Quantidade de novos perfis a criar (padrao: 5).",
        )
        parser.add_argument(
            "--turns",
            type=str,
            default="5",
            help="Quantidade de turnos por conversa (ex.: 5) ou intervalo (ex.: 5-10).",
        )

    def _parse_turns_option(self, raw_turns):
        value = (raw_turns or "").strip()
        if not value:
            raise ValueError("--turns precisa ser informado.")

        if "-" not in value:
            try:
                fixed_turns = int(value)
            except ValueError as exc:
                raise ValueError(
                    "--turns inválido. Use inteiro (ex.: 5) ou intervalo (ex.: 5-10)."
                ) from exc
            if fixed_turns < 1:
                raise ValueError("--turns precisa ser >= 1.")
            return fixed_turns, fixed_turns

        parts = value.split("-")
        if len(parts) != 2:
            raise ValueError(
                "--turns inválido. Use inteiro (ex.: 5) ou intervalo (ex.: 5-10)."
            )

        start_raw, end_raw = parts[0].strip(), parts[1].strip()
        if not start_raw or not end_raw:
            raise ValueError(
                "--turns inválido. Use inteiro (ex.: 5) ou intervalo (ex.: 5-10)."
            )
        try:
            start = int(start_raw)
            end = int(end_raw)
        except ValueError as exc:
            raise ValueError(
                "--turns inválido. Use inteiro (ex.: 5) ou intervalo (ex.: 5-10)."
            ) from exc
        if start < 1 or end < 1:
            raise ValueError("--turns precisa ser >= 1.")
        if start > end:
            raise ValueError("--turns inválido. No intervalo A-B, A deve ser <= B.")
        return start, end

    def handle(self, *args, **options):
        count = int(options["count"])
        turns_min, turns_max = self._parse_turns_option(options["turns"])

        if count < 1:
            raise ValueError("--count precisa ser >= 1.")

        initial_simulation_theme = (
            Theme.objects.filter(
                Q(slug="nao_identificado")
                | Q(name__iexact="Não identificado")
                | Q(name__iexact="Nao identificado")
                | Q(name__iexact="nao_identificado")
            )
            .order_by("id")
            .first()
        )
        if not initial_simulation_theme:
            raise RuntimeError(
                "Theme 'nao_identificado' not found for initial simulation message."
            )

        available_theme_ids = list(
            Theme.objects.exclude(id=initial_simulation_theme.id)
            .order_by("id")
            .values_list("id", flat=True)
        )
        if not available_theme_ids:
            raise RuntimeError(
                "Nenhum theme disponivel para o parametro 'meu problema'."
            )

        emotional_profiles = [item.value for item in SimulatedUserProfile]
        predefined_scenarios = list(PREDEFINED_SCENARIOS.keys())
        if not predefined_scenarios:
            raise RuntimeError("Nenhum cenario predefinido disponivel para simulacao.")

        self.stdout.write(
            self.style.WARNING(
                f"Iniciando geracao: perfis={count} turns={turns_min}-{turns_max}."
            )
        )

        for index in range(1, count + 1):
            profile = self._create_profile()
            selected_turns = random.randint(turns_min, turns_max)
            selected_emotional_profile = random.choice(emotional_profiles)
            selected_predefined_scenario = random.choice(predefined_scenarios)
            selected_theme = Theme.objects.get(id=random.choice(available_theme_ids))

            self.stdout.write(
                (
                    f"[{index}/{count}] profile_id={profile.id} "
                    f"name='{profile.name}' "
                    f"turns='{selected_turns}' "
                    f"como_eu_me_sinto='{selected_predefined_scenario}' "
                    f"o_que_eu_quero='{selected_emotional_profile}' "
                    f"meu_problema='{selected_theme.name}'"
                )
            )

            try:
                self._simulate_full_conversation(
                    profile=profile,
                    turns=selected_turns,
                    initial_simulation_theme=initial_simulation_theme,
                    emotional_profile=selected_emotional_profile,
                    predefined_scenario=selected_predefined_scenario,
                    locked_conversation_theme=selected_theme,
                )
            except RuntimeError as exc:
                logger.exception(
                    "Simulation aborted for profile_id=%s due to runtime error.",
                    profile.id,
                )
                self.stdout.write(
                    self.style.WARNING(
                        f"[{index}/{count}] profile_id={profile.id} abortado: {exc}"
                    )
                )
                continue

            self.stdout.write(
                self.style.SUCCESS(
                    f"[{index}/{count}] concluido profile_id={profile.id} com analise final."
                )
            )

        self.stdout.write(self.style.SUCCESS("Geracao concluida com sucesso."))

    def _create_profile(self):
        gender = random.choice([GENDER_MALE, GENDER_FEMALE])
        if gender == GENDER_MALE:
            profile_name = _faker.first_name_male()
        else:
            profile_name = _faker.first_name_female()
        return Profile.objects.create(name=profile_name, inferred_gender=gender)

    def _simulate_full_conversation(
        self,
        profile,
        turns,
        initial_simulation_theme,
        emotional_profile,
        predefined_scenario,
        locked_conversation_theme,
    ):
        simulation_use_case = SimulationUseCase()
        chat_service = ChatService()
        profile.welcome_message_sent = False
        profile.save(update_fields=["welcome_message_sent", "updated_at"])

        topic_openers = [
            "Mudando um pouco de assunto,",
            "Queria abrir meu coração sobre isso:",
            "Tenho pensado nisso hoje e",
            "Tem uma coisa que está pegando para mim:",
        ]

        for turn in range(1, turns + 1):
            if turn == 1:
                user_text = f"Oi, eu sou {profile.name}. Queria conversar com você."
                user_payload = {
                    "source": "conversation_simulator",
                    "turn": turn,
                    "type": "intro",
                }
            else:
                conversation = (
                    Message.objects.filter(profile=profile)
                    .exclude(role="system")
                    .exclude(role="analysis")
                    .exclude(exclude_from_context=True)
                    .order_by("created_at")
                )
                simulation_result = (
                    simulation_use_case.simulate_next_user_message_with_metadata(
                        conversation=conversation,
                        profile=emotional_profile,
                        predefined_scenario=predefined_scenario,
                        theme=locked_conversation_theme.id,
                        inferred_gender=profile.inferred_gender,
                        force_context_expansion=turn <= 3,
                        profile_instance=profile,
                    )
                )
                user_text = simulation_result.get("content", "").strip()
                if turn == 2:
                    user_text = f"{random.choice(topic_openers)} {user_text}".strip()
                user_payload = {
                    "source": "conversation_simulator",
                    "turn": turn,
                    "payload": simulation_result.get("payload"),
                }

            user_message = Message.objects.create(
                profile=profile,
                role="user",
                content=user_text,
                channel="simulation",
                generated_by_simulator=True,
                ollama_prompt=user_payload,
                theme=(
                    initial_simulation_theme if turn == 1 else locked_conversation_theme
                ),
            )
            user_message.block_root = user_message
            user_message.save(update_fields=["block_root"])

            try:
                chat_service.generate_response_message(
                    profile=profile,
                    channel="chat",
                    forced_theme=locked_conversation_theme if turn != 1 else None,
                )
            except RuntimeError as exc:
                if str(exc) != "Evaluation model returned empty content.":
                    raise
                logger.exception(
                    (
                        "Ignoring evaluation-empty-content and continuing "
                        "simulation profile_id=%s turn=%s"
                    ),
                    profile.id,
                    turn,
                )
                self.stdout.write(
                    self.style.WARNING(
                        (
                            f"[profile_id={profile.id} turn={turn}] "
                            "Evaluation model returned empty content. "
                            "Continuando simulacao."
                        )
                    )
                )
                continue

        report = chat_service.analyze_conversation_emotions(profile=profile)
        profile.last_simulation_report = report
        profile.save(update_fields=["last_simulation_report", "updated_at"])

        analysis_message = Message.objects.create(
            profile=profile,
            role="analysis",
            content=f"📊 Relatório da Simulação:\n\n{report}",
            channel="other",
            exclude_from_context=True,
        )
        analysis_message.block_root = analysis_message
        analysis_message.save(update_fields=["block_root"])
