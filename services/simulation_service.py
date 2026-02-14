"""Simulation service focused on generating only the next user turn."""

import os
from enum import Enum
from typing import Iterable, List, Union

from core.models import Message, Profile
from services.llm_service import LLMService, get_llm_service

_DEFAULT_SIMULATION_MODEL = (
    "gpt-5-mini"
    if os.environ.get("LLM_PROVIDER", "openai").lower() == "openai"
    else os.environ.get("OLLAMA_CHAT_MODEL", "llama3:8b")
)
SIMULATION_MODEL = (
    os.environ.get("LLM_SIMULATION_MODEL")
    or os.environ.get("OPENAI_SIMULATION_MODEL")
    or _DEFAULT_SIMULATION_MODEL
)


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

THEME_OPTIONS = {
    "": "não ficou claro",
    "relacionamento": "conflitos em relacionamento e família",
    "financeiro": "dificuldades com dinheiro e dívidas",
    "vicios": "vícios e recaídas",
    "saude": "saúde fragilizada e cansaço",
    "luto_perda": "luto e perdas recentes",
    "trabalho": "pressão e desgaste no trabalho",
    "solidao": "solidão e falta de suporte",
    "outros": "um problema importante ainda não nomeado",
}


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
        - Responda com uma única mensagem curta, com 1 ou 2 frases no máximo.
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
    theme: str = "",
) -> str:
    """Generate only the next user message based on recent history and emotional profile."""
    return SimulationUseCase(llm_service=get_llm_service()).simulate_next_user_message(
        conversation=conversation,
        profile=profile,
        predefined_scenario=predefined_scenario,
        theme=theme,
    )


class SimulationUseCase:
    def __init__(self, llm_service: LLMService):
        self._llm_service = llm_service

    def simulate_next_user_message(
        self,
        conversation,
        profile: SimulatedUserProfile,
        predefined_scenario: str = "",
        theme: str = "",
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
        theme: str = "",
    ) -> dict:
        selected_profile = _parse_profile(profile)
        selected_scenario = (
            predefined_scenario if predefined_scenario in PREDEFINED_SCENARIOS else ""
        )
        selected_theme = theme if theme in THEME_OPTIONS else ""
        feeling_label = PREDEFINED_SCENARIOS.get(
            selected_scenario, "está emocionalmente abalada"
        )
        problem_label = THEME_OPTIONS.get(selected_theme, "não ficou claro")
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
        temperature = 0.9
        content = (
            self._llm_service.basic_call(
                url_type="generate",
                prompt=prompt,
                model=SIMULATION_MODEL,
                temperature=temperature,
                max_tokens=180,
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
            "temperature": temperature,
        }

    def handle(
        self,
        profile_id: int,
        emotional_profile: Union[SimulatedUserProfile, str],
        predefined_scenario: str = "",
        theme: str = "",
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

        Message.objects.create(
            profile=profile,
            role="user",
            content=simulation["content"],
            channel="simulation",
            generated_by_simulator=True,
            ollama_prompt=simulation.get("payload"),
            ollama_prompt_temperature=simulation["temperature"],
        )
        return profile.id
