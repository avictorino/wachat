from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from services.prompts.themes.drug_addiction import DRUG_ADDICTION_THEME_ID

_ADDICTION_INTENTS = {"drogas", "alcool", "cigarro", "sexo"}

_ADDICTION_KEYWORDS_RE = re.compile(
    r"(?i)\b("
    r"drogas?|coca[ií]na|crack|maconha|baseado|p[oó]|pedra|subst[âa]ncia[s]?|"
    r"[áa]lcool|alcool|bebida|beber|alcoolismo|b[eê]bado|b[eê]bada|"
    r"cigarro[s]?|fumar|fumo|tabaco|tabagismo|nicotina|"
    r"v[ií]cio|viciado|viciada|depend[êe]ncia|compuls[ãa]o"
    r")\b"
)


@dataclass(frozen=True)
class ThemeSelection:
    theme_id: Optional[str]
    reason: str


def select_theme_from_intent_and_message(
    *, intent: Optional[str], message_text: str, existing_theme_id: Optional[str]
) -> ThemeSelection:
    """
    Decide which thematic prompt (if any) should be active for this user.

    - If a theme is already active, keep it (persistence).
    - Otherwise, select based on detected intent (preferred) and/or a keyword check
      so the theme can activate later in the conversation when the user reveals it.
    """
    if existing_theme_id:
        return ThemeSelection(theme_id=existing_theme_id, reason="existing_theme")

    if intent and intent in _ADDICTION_INTENTS:
        return ThemeSelection(theme_id=DRUG_ADDICTION_THEME_ID, reason="intent_match")

    if _ADDICTION_KEYWORDS_RE.search(message_text or ""):
        return ThemeSelection(theme_id=DRUG_ADDICTION_THEME_ID, reason="keyword_match")

    return ThemeSelection(theme_id=None, reason="no_match")


__all__ = ["select_theme_from_intent_and_message", "ThemeSelection"]
