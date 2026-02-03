from __future__ import annotations

from typing import Optional

from services.prompt.themes.drug_addiction import (
    DRUG_ADDICTION_THEME_ID,
    DRUG_ADDICTION_THEME_PROMPT_PTBR,
)

_THEME_PROMPTS = {
    DRUG_ADDICTION_THEME_ID: DRUG_ADDICTION_THEME_PROMPT_PTBR,
}


def get_theme_prompt(theme_id: Optional[str]) -> str:
    if not theme_id:
        return ""
    return _THEME_PROMPTS.get(theme_id, "")


__all__ = ["get_theme_prompt"]
