"""Simulation service focused on generating only the next user turn."""

from enum import Enum
from typing import Iterable, List, Optional, Union

from core.models import Message, Profile, Theme
from services.openai_service import OpenAIService
from services.theme_classifier import ThemeClassifier

SIMULATION_MAX_COMPLETION_TOKENS = 1200


class SimulatedUserProfile(Enum):
    AMBIVALENTE = "ambivalente"
    DEFENSIVO = "defensivo"
    CULPA_FORTE = "culpa_forte"
    DESESPERANCA = "desesperanca"
    RACIONAL = "racional"
    PEDIDO_AJUDA = "pedido_ajuda_implicito"
    FECHAMENTO = "fechamento_emocional"


PROFILE_INSTRUCTIONS = {
    SimulatedUserProfile.AMBIVALENTE: "Quer mudar, mas sente que não consegue manter constância.",
    SimulatedUserProfile.DEFENSIVO: "Quer ser ouvido sem julgamento e sem bronca.",
    SimulatedUserProfile.CULPA_FORTE: "Quer aliviar a culpa e encontrar perdão.",
    SimulatedUserProfile.DESESPERANCA: "Quer voltar a ter esperança e força para continuar.",
    SimulatedUserProfile.RACIONAL: "Quer clareza para tomar uma decisão prática.",
    SimulatedUserProfile.PEDIDO_AJUDA: "Quer direção prática para o próximo passo.",
    SimulatedUserProfile.FECHAMENTO: "Quer encerrar o assunto por cansaço emocional.",
}

PROFILE_LABELS = {
    SimulatedUserProfile.AMBIVALENTE: "quer mudar, mas não consegue sustentar a mudança",
    SimulatedUserProfile.DEFENSIVO: "quer ser ouvida sem julgamento",
    SimulatedUserProfile.CULPA_FORTE: "quer aliviar a culpa que está carregando",
    SimulatedUserProfile.DESESPERANCA: "quer voltar a ter esperança",
    SimulatedUserProfile.RACIONAL: "quer clareza para decidir o que fazer",
    SimulatedUserProfile.PEDIDO_AJUDA: "quer orientação prática para o próximo passo",
    SimulatedUserProfile.FECHAMENTO: "quer fugir do assunto por desgaste emocional",
}

PREDEFINED_SCENARIOS = {
    "sobrecarregado": "está sobrecarregada e no limite",
    "ansioso_com_medo": "se sente ansiosa e com medo",
    "culpado_envergonhado": "está culpada e envergonhada",
    "irritado_com_raiva": "está irritada e com raiva",
    "sozinho_sem_apoio": "se sente sozinha e sem apoio",
    "confuso_perdido": "está confusa e perdida",
    "desanimado_sem_forca": "está desanimada e sem força",
    "sem_esperanca": "se sente sem esperança",
}


def _theme_options_from_db() -> dict:
    options = {}
    for theme in Theme.objects.all().order_by("name"):
        options[theme.id] = theme.name
    return options


def _parse_optional_theme_id(theme: Union[int, str, None]) -> Optional[int]:
    if theme is None:
        return None
    raw = str(theme).strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid simulation theme id: '{raw}'.") from exc


def _to_recent_history(conversation: Iterable, limit: int = 5) -> List[dict]:
    normalized = []
    for item in list(conversation)[-limit:]:
        if isinstance(item, dict):
            role = item.get("role", "user")
            content = (item.get("content") or "").strip()
        else:
            role = getattr(item, "role", "user")
            content = (getattr(item, "content", "") or "").strip()
        if content:
            normalized.append({"role": role, "content": content})
    return normalized


def _parse_profile(profile: Union[SimulatedUserProfile, str]) -> SimulatedUserProfile:
    if isinstance(profile, SimulatedUserProfile):
        return profile
    for option in SimulatedUserProfile:
        if option.value == profile:
            return option
    return SimulatedUserProfile.AMBIVALENTE


def _build_simple_simulation_prompt(
    *,
    feeling_context: str,
    desire_context: str,
    problem_context: str,
    history_text: str,
) -> str:

    return f"""
        Gere UMA única frase simulando a mensagem de uma pessoa real, escrita como em uma conversa de WhatsApp.

        Contexto obrigatório desta fala:
        - Como me sinto: {feeling_context}
        - O que eu quero: {desire_context}
        - Meu problema: {problem_context}
        - Procura apoio espiritual

        Histórico recente da conversa (use apenas como referência):
        {history_text if history_text else "Sem histórico relevante."}

        Regras obrigatórias:
        - Responda com uma única mensagem curta, com 1 frase preferencialmente (no máximo 2).
        - Limite total: até 30 palavras.
        - Não use listas.
        - Não copie literalmente os textos de contexto acima.
        - Não explique a situação de forma abstrata.
        - Não use linguagem técnica, psicológica ou conceitual.
        - Evite clichês e frases genéricas.
        - Mostre a situação por sentimentos, pensamentos ou comportamentos concretos.
        - Use tom de confissão para alguém de confiança da igreja, sem mencionar explicitamente "pastor".
        """


def simulate_next_user_message(
    conversation,
    profile: SimulatedUserProfile,
    predefined_scenario: str = "",
    theme: Union[int, str, None] = None,
) -> str:
    """Generate only the next user message based on recent history and emotional profile."""
    return SimulationUseCase().simulate_next_user_message(
        conversation=conversation,
        profile=profile,
        predefined_scenario=predefined_scenario,
        theme=theme,
    )


class SimulationUseCase:
    def __init__(self):
        self._llm_service = OpenAIService()
        self._theme_classifier = ThemeClassifier()

    def simulate_next_user_message(
        self,
        conversation,
        profile: SimulatedUserProfile,
        predefined_scenario: str = "",
        theme: Union[int, str, None] = None,
    ) -> str:
        result = self.simulate_next_user_message_with_metadata(
            conversation=conversation,
            profile=profile,
            predefined_scenario=predefined_scenario,
            theme=theme,
        )
        return result["content"]

    def simulate_next_user_message_with_metadata(
        self,
        conversation,
        profile: SimulatedUserProfile,
        predefined_scenario: str = "",
        theme: Union[int, str, None] = None,
    ) -> dict:
        selected_profile = _parse_profile(profile)
        selected_scenario = (
            predefined_scenario if predefined_scenario in PREDEFINED_SCENARIOS else ""
        )
        available_theme_ids = set(Theme.objects.values_list("id", flat=True))
        selected_theme = _parse_optional_theme_id(theme)
        if selected_theme is not None and selected_theme not in available_theme_ids:
            raise ValueError(f"Theme '{selected_theme}' not found for simulation.")
        theme_options = _theme_options_from_db()
        feeling_label = PREDEFINED_SCENARIOS.get(
            selected_scenario, "está emocionalmente abalada"
        )
        problem_label = theme_options.get(selected_theme, "não ficou claro")
        recent_history = _to_recent_history(conversation=conversation, limit=5)
        history_text = ""
        for message in recent_history:
            history_text += f"{message['role'].upper()}: {message['content']}\n"

        desire_label = PROFILE_LABELS.get(
            selected_profile, selected_profile.value.replace("_", " ")
        )
        prompt = _build_simple_simulation_prompt(
            feeling_context=feeling_label,
            desire_context=desire_label,
            problem_context=problem_label,
            history_text=history_text,
        )
        content = (
            self._llm_service.basic_call(
                url_type="generate",
                prompt=prompt,
                max_tokens=SIMULATION_MAX_COMPLETION_TOKENS,
            )
            or ""
        ).strip()
        payload = self._llm_service.get_last_prompt_payload()
        if not content:
            raise ValueError(
                "Simulador retornou resposta vazia. Ajuste o prompt/modelo e tente novamente."
            )
        return {
            "content": content,
            "prompt": prompt,
            "payload": payload,
        }

    def handle(
        self,
        profile_id: int,
        emotional_profile: Union[SimulatedUserProfile, str],
        predefined_scenario: str = "",
        theme: Union[int, str, None] = None,
    ) -> int:
        profile = Profile.objects.get(id=profile_id)
        conversation = (
            Message.objects.filter(profile=profile)
            .exclude(role="system")
            .exclude(role="analysis")
            .exclude(exclude_from_context=True)
            .order_by("created_at")
        )
        simulation = self.simulate_next_user_message_with_metadata(
            conversation=conversation,
            profile=_parse_profile(emotional_profile),
            predefined_scenario=predefined_scenario,
            theme=theme,
        )
        theme_id = self._theme_classifier.classify(simulation["content"])
        selected_theme = Theme.objects.filter(id=theme_id).first()
        if not selected_theme:
            raise RuntimeError(f"Theme '{theme_id}' not found in database.")

        message = Message.objects.create(
            profile=profile,
            role="user",
            content=simulation["content"],
            channel="simulation",
            generated_by_simulator=True,
            ollama_prompt=simulation.get("payload"),
            theme=selected_theme,
        )
        message.block_root = message
        message.save(update_fields=["block_root"])
        return profile.id
