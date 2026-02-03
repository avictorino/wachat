from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from services.prompts.themes import get_theme_prompt


@dataclass(frozen=True)
class PromptComposer:
    """
    Composes the system prompt as: Optional Theme + Mode.

    The base behavioral prompt is now defined in the Modelfile at the project root,
    which is the single source of truth for conversational behavior.

    This composer only handles dynamic, context-specific prompts:
    - Themes (registered in `services.prompts.themes`)
    - Mode-specific instructions (intent_response, fallback_response)

    Themes are registered in `services.prompts.themes` so adding new ones is
    a small, explicit change.
    """

    @staticmethod
    def compose_system_prompt(*, theme_id: Optional[str], mode: str) -> str:
        theme_prompt = get_theme_prompt(theme_id) if theme_id else ""

        mode_prompt = PromptComposer._mode_prompt(mode)

        parts = []
        if theme_prompt:
            parts.append(theme_prompt.strip())
        parts.append(mode_prompt.strip())

        return "\n\n---\n\n".join(parts).strip() + "\n"

    @staticmethod
    def _mode_prompt(mode: str) -> str:
        if mode == "intent_response":
            return """TAREFA
Responda à mensagem da pessoa com presença e clareza.

REGRAS DE RESPOSTA (CRÍTICO)
- 1–3 frases (prefira 1–2).
- Trate o que foi dito com cuidado; não invente fatos.
- SEMPRE inclua: validação/empatia + convite à continuação.
- O convite pode ser uma declaração reflexiva (ex: "Parece que isso tem sido difícil.") OU NO MÁXIMO uma pergunta suave.
- Prefira declarações reflexivas; use pergunta apenas quando necessário para clarificar ou guiar.
- NUNCA faça mais de uma pergunta na mesma resposta.
- NUNCA termine apenas com validações passivas como "Estou aqui" sem continuação.

SEPARAÇÃO EM MÚLTIPLAS MENSAGENS (quando fizer sentido)
- Você pode responder em até 3 mensagens curtas.
- Use \"|||\" para separar mensagens.
- Se usar múltiplas mensagens, qualquer pergunta deve estar na última mensagem."""

        if mode == "fallback_response":
            return """TAREFA
Continue a conversa usando o histórico recente como contexto.

REGRAS DE RESPOSTA (CRÍTICO)
- 1–3 frases (prefira 1–2).
- Seja específico ao que a pessoa acabou de dizer.
- SEMPRE inclua: validação/empatia + convite à continuação.
- O convite pode ser uma declaração reflexiva (ex: "Parece que isso tem sido difícil.") OU NO MÁXIMO uma pergunta suave.
- Prefira declarações reflexivas; use pergunta apenas quando necessário para clarificar ou guiar.
- NUNCA faça mais de uma pergunta na mesma resposta.
- NUNCA termine apenas com validações passivas como "Estou aqui" sem continuação.

SEPARAÇÃO EM MÚLTIPLAS MENSAGENS (quando fizer sentido)
- Você pode responder em até 3 mensagens curtas.
- Use \"|||\" para separar mensagens.
- Se usar múltiplas mensagens, qualquer pergunta deve estar na última mensagem."""

        raise ValueError(f"Unknown prompt composition mode: {mode}")
