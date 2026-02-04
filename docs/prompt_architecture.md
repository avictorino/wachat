# Prompt architecture (Base + Themes)

This project composes LLM system prompts from two layers:

1) **Base prompt** (generalist, shared)
2) **Thematic prompt** (problem-specific, optional, plain text only)

The goal is to keep the base behavior consistent (warmth, brevity, respectful Christian spirituality) while allowing additional, focused guidance when a user problem is identified (e.g., addiction).

## Where prompts live

- Base prompt: `services/prompts/composer.py` (embedded for runtime API calls)
- System prompt: `model/Modelfile` (for Ollama model)
- Themes registry: `services/prompts/themes/__init__.py`
- Theme implementation(s): `services/prompts/themes/*.py`
- Composition logic: `services/prompts/composer.py`

## Theme constraints

Themes MUST:
- Return plain text blocks only
- NOT embed identity, tone, or system rules
- Provide only context-specific guidance relevant to the theme

## Theme selection and persistence

- Theme selection helper: `services/theme_selector.py`
- The selected theme is persisted per user in `Profile.prompt_theme` (see `core/models.py`).
- `core/views.py` activates a theme when:
  - The detected intent matches a known theme, or
  - A keyword-based trigger matches later in the conversation (then intent is re-detected once to confirm).

## Adding a new theme

1. Create a new theme file in `services/prompts/themes/` (e.g., `anxiety.py`) with:
   - a stable `*_THEME_ID`
   - a `*_THEME_PROMPT_PTBR` string (plain text only, no identity/tone/system rules)
2. Register it in `services/prompts/themes/__init__.py`
3. Update `services/theme_selector.py` to map the relevant intents and/or add safe keyword triggers
4. (Optional) Add tests similar to `spec/services/test_prompt_composition.py`
