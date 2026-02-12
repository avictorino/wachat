"""
Simulation service for creating AI-driven conversation simulations.

This service orchestrates simulated conversations between two AI roles
and manages emotional analysis of the conversation.
"""

import logging
import random

from faker import Faker

from core.models import Message, Profile, Theme, ThemeRoleChoices
from services.ollama_service import OllamaService

logger = logging.getLogger(__name__)

_faker = Faker("pt_BR")

# Role labels for analysis output
ROLE_LABEL_SEEKER = "Pessoa"  # Portuguese for "Person"
ROLE_LABEL_LISTENER = "BOT"  # Bot assistant
OLLAMA_SIMULATION_MODEL = "llama3:8b"

# Emotional states for user simulation
ALL_EMOTIONAL_STATES = [
    "CONFUSION",
    "LOSS_OF_CONTROL",
    "RESISTANCE",
    "SHAME",
    "EXHAUSTION",
    "AMBIVALENCE",
]

# State randomness: 30% chance to pick any state for variety
STATE_RANDOMNESS_THRESHOLD = 0.3

# Higher temperature for user simulation to increase natural variation
# 0.95 chosen to produce more diverse, human-like contradictions while maintaining coherence
USER_SIMULATION_TEMPERATURE = 0.95


class SimulationUseCase:
    """
    Service for simulating conversations between two AI roles.

    Simulates realistic, gradual dialogue between:
    - ROLE_A: Introspective and reserved seeker (cautious, building trust slowly)
    - ROLE_B: Patient and relational listener (present, companionable)

    The conversation simulates the beginning of a friendship, not a therapy session.
    """

    def __init__(self, ollama_service: OllamaService):

        self._ollama_service = ollama_service

    def _create_simulation_profile(self, theme_name: str = "desabafar") -> Profile:
        gender = random.choice(["male", "female"])

        # Generate a realistic name based on the gender using Faker
        if gender == "male":
            sim_name = _faker.first_name_male()
        else:
            sim_name = _faker.first_name_female()

        theme, created = Theme.objects.get_or_create(
            name=theme_name, role=ThemeRoleChoices.PERSON_SIMULATOR
        )

        if not theme.prompt:
            theme.prompt = self._ollama_service.build_theme_prompt(theme_name)
            theme.save()

        # Create profile with prompt_theme persisted
        profile = Profile.objects.create(
            name=sim_name,
            inferred_gender=gender,
            theme=theme,
        )

        logger.info(
            f"Created simulation profile: {profile.id} with name: {sim_name}, gender: {gender}, theme: {theme}"
        )
        return profile

    def handle(self, theme_name: str = None, num_messages: int = 8) -> int:

        profile = self._create_simulation_profile(theme_name=theme_name)

        logger.info(
            f"Generating simulated messages for profile {profile.id} with theme: {theme_name}"
        )

        for i in range(num_messages):
            self._generate_person_message(profile=profile)

            if i == 0:
                self._ollama_service.generate_welcome_message(
                    profile=profile, channel="simulation"
                )
            else:
                self._ollama_service.generate_response_message(
                    profile=profile,
                    channel="simulation",
                )

            logger.info(
                f"Saved simulated user message for profile {profile.id}/{profile.name}"
            )

        return profile.id

    def _get_conversation_context(
        self, profile: Profile, limit: int = 5, exclude_message_id: int = None
    ) -> list:
        """
        Get recent conversation context for the LLM.
        Excludes system messages to focus on user-assistant dialogue.

        Args:
            profile: Profile to get messages for
            limit: Maximum number of recent messages to include
            exclude_message_id: Optional message ID to exclude from context (typically the current user message)

        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        query = Message.objects.filter(profile=profile).exclude(role="system").exclude(role="analysis").exclude(exclude_from_context=True)

        # Exclude specific message if provided (to avoid duplication)
        if exclude_message_id:
            query = query.exclude(id=exclude_message_id)

        recent_messages = query.order_by("-created_at")[:limit]

        context = []
        for msg in reversed(recent_messages):
            context.append({"role": msg.role, "content": msg.content})

        logger.info(f"Retrieved {len(context)} messages for conversation context")
        return context

    def _generate_person_message(self, profile: Profile) -> Message:

        WELCOME_MESSAGE = ""
        if profile.messages.count() == 0:
            WELCOME_MESSAGE = f"""
                0. MENSAGEM INICIAL (APENAS NO PRIMEIRO TURNO):
                    - Se esta for a PRIMEIRA mensagem da conversa, comece com uma breve apresentação espontânea.
                    - A apresentação deve soar natural, simples e falada.
                    - Não explique o motivo de estar ali de forma clara demais.
                    - Não organize pensamentos.
                    - Máximo de 1 frase.
                    - Exemplo de tom (não copiar literalmente):
                      - "Meu nome é João, eu nem sei direito por que resolvi falar aqui."
                      - "Sou a Ana… eu precisava falar com alguém agora.
                    - EXEMPLOS:
                      - Eu sou o {profile.name}, nem pensei muito antes de escrever aqui.
                            Eu sei que devia ficar quieto e seguir em frente, mas ao mesmo tempo não aguento
                            mais guardar isso e fico me sentindo errado por querer falar.
                      - Sou a {profile.name}… eu precisava falar com alguém agora.
                            Eu fico tentando segurar tudo sozinha, aí me sinto fraca por isso e
                            mesmo assim continuo empurrando com a barriga.
                      - Meu nome é {profile.name}, eu nem sei direito por que resolvi falar aqui agora.
                            Eu tô com a cabeça cheia, sei que devia dar conta sozinho, mas ao mesmo
                            tempo queria alguém de fé pra ouvir sem me julgar.
                """

        PROMPT = f"""
            Você está simulando uma PESSOA REAL em sofrimento, NÃO um assistente.
            Você está buscando conversar com alguém religioso.
            Você é imperfeito, confuso e contraditório.
            Você NÃO sabe que está sendo simulado.

            Tema: {profile.theme.name or "desabafar"}

            {WELCOME_MESSAGE}

            REGRAS CRÍTICAS — PRIORIDADE MÁXIMA

            1. COMPRIMENTO (LIMITE RÍGIDO):
            - MÁXIMO de 3 frases
            - Prefira 1 ou 2 frases
            - Frases curtas, diretas, faladas

            2. O QUE NUNCA FAZER:
            ❌ Usar linguagem terapêutica ou reflexiva
            ❌ Explicar sentimentos de forma clara
            ❌ Demonstrar autoconsciência emocional
            ❌ Soar racional, organizado ou didático
            ❌ Usar expressões como:
               - "Parece difícil"
               - "Isso soa como"
               - "Estou passando por um momento"
               - "Estou lutando com"
               - "É como se"
               - "Parece que"

            3. O QUE FAZER:
            ✅ Falar como alguém falando, não escrevendo
            ✅ Mostrar confusão, dúvida e contradição
            ✅ Misturar culpa com desejo
            ✅ Pensamento de recaída (especialmente para drogas/álcool)
            ✅ Linguagem informal do português brasileiro
            ✅ Hesitação, repetição leve e inconsistência emocional

            4. TOM E ESTILO:
            - Sempre em primeira pessoa
            - Não seja melodramático
            - Não resuma seu estado emocional
            - Não resolva as contradições
            - Pode ser cru, direto e até duro consigo mesmo

            5. EXEMPLOS DE TOM CORRETO:
            - "Na hora parece que resolve tudo, mas depois eu me odeio por ter feito isso."
            - "Eu sei que isso vai dar ruim, mas quando chega aquele momento eu simplesmente vou."
            - "Eu jurei que não ia mais, e mesmo assim tô aqui de novo."

            Responda APENAS com a fala curta da pessoa (1 a 3 frases).
            Não use aspas.
            Não explique nada.
        """

        for idx, message in enumerate(profile.messages.all().exclude(role="system").exclude(role="analysis").exclude(exclude_from_context=True)):
            if idx == 0:
                PROMPT += "\n\nBASEAR MINHA PERGUNTA NOS ACONTECIMENTOS ANTERIORES:\n\n"
            PROMPT += f"{message.role.upper()}: {message.content}\n\n\n"

        temperature = random.choice([0.4, 0.5, 0.6, 0.7, 0.8])
        result = self._ollama_service.basic_call(
            url_type="generate",
            prompt=PROMPT,
            model=OLLAMA_SIMULATION_MODEL,
            temperature=temperature,
            max_tokens=70,
        )

        message = Message.objects.create(
            profile=profile,
            role="user",
            content=result,
            channel="simulation",
            ollama_prompt=PROMPT,
            ollama_prompt_temperature=temperature,
        )

        return message
