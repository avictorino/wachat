"""Simulation service focused on generating only the next user turn."""

import re
from enum import Enum
from typing import Iterable, List, Union

from core.models import Message, Profile
from services.ollama_service import OllamaService

OLLAMA_SIMULATION_MODEL = "llama3:8b"


class SimulatedUserProfile(Enum):
    AMBIVALENTE = "ambivalente"
    DEFENSIVO = "defensivo"
    CULPA_FORTE = "culpa_forte"
    DESESPERANCA = "desesperanca"
    RACIONAL = "racional"
    PEDIDO_AJUDA = "pedido_ajuda_implicito"
    FECHAMENTO = "fechamento_emocional"


PROFILE_INSTRUCTIONS = {
    SimulatedUserProfile.AMBIVALENTE: "Conflito interno: quer mudar, mas resiste.",
    SimulatedUserProfile.DEFENSIVO: "Tom defensivo: justifica ações ou reage ao bot.",
    SimulatedUserProfile.CULPA_FORTE: "Arrependimento intenso e autojulgamento.",
    SimulatedUserProfile.DESESPERANCA: "Baixa esperança de mudança.",
    SimulatedUserProfile.RACIONAL: "Tom lógico, objetivo e menos emocional.",
    SimulatedUserProfile.PEDIDO_AJUDA: "Indica necessidade de orientação prática.",
    SimulatedUserProfile.FECHAMENTO: "Minimiza o problema ou tenta encerrar o assunto.",
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


def _trim_to_three_sentences(text: str) -> str:
    chunks = [part.strip() for part in re.findall(r"[^.!?]+[.!?]?", text or "")]
    chunks = [part for part in chunks if part]
    if not chunks:
        return ""
    return " ".join(chunks[:3]).strip()


def _parse_profile(profile: Union[SimulatedUserProfile, str]) -> SimulatedUserProfile:
    if isinstance(profile, SimulatedUserProfile):
        return profile
    for option in SimulatedUserProfile:
        if option.value == profile:
            return option
    return SimulatedUserProfile.AMBIVALENTE


def simulate_next_user_message(conversation, profile: SimulatedUserProfile) -> str:
    """Generate only the next user message based on recent history and emotional profile."""
    return SimulationUseCase(ollama_service=OllamaService()).simulate_next_user_message(
        conversation=conversation, profile=profile
    )


class SimulationUseCase:
    def __init__(self, ollama_service: OllamaService):
        self._ollama_service = ollama_service

    def simulate_next_user_message(
        self, conversation, profile: SimulatedUserProfile
    ) -> str:
        result = self.simulate_next_user_message_with_metadata(
            conversation=conversation, profile=profile
        )
        return result["content"]

    def simulate_next_user_message_with_metadata(
        self, conversation, profile: SimulatedUserProfile
    ) -> dict:
        selected_profile = _parse_profile(profile)
        recent_history = _to_recent_history(conversation=conversation, limit=5)

        history_text = ""
        for message in recent_history:
            history_text += f"{message['role'].upper()}: {message['content']}\n"

        prompt = f"""
Você está simulando o próximo turno do usuário.
O usuário deve responder de acordo com o perfil emocional selecionado.

Perfil emocional: {selected_profile.value}
Diretriz do perfil: {PROFILE_INSTRUCTIONS[selected_profile]}

Histórico recente:
{history_text if history_text else "Sem histórico disponível."}

Regras obrigatórias:
- Gere apenas UMA próxima mensagem do usuário.
- Máximo 3 frases.
- Linguagem natural e coerente com o perfil.
- Não explique nada.
- Não escreva como assistente.
- Não analise.
- Não use listas ou rótulos.
- Escreva apenas a próxima fala do usuário.
"""
        temperature = 0.5
        result = self._ollama_service.basic_call(
            url_type="generate",
            prompt=prompt,
            model=OLLAMA_SIMULATION_MODEL,
            temperature=temperature,
            max_tokens=90,
        )
        return {
            "content": _trim_to_three_sentences(result),
            "prompt": prompt,
            "temperature": temperature,
        }

    def handle(
        self, profile_id: int, emotional_profile: Union[SimulatedUserProfile, str]
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
        )

        Message.objects.create(
            profile=profile,
            role="user",
            content=simulation["content"],
            channel="simulation",
            generated_by_simulator=True,
            ollama_prompt=simulation["prompt"],
            ollama_prompt_temperature=simulation["temperature"],
        )
        return profile.id
