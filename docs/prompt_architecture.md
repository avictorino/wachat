# Prompt architecture (Modelfile + Themes)

This project composes LLM system prompts from two layers:

1) **Base behavioral prompt** (defined in Modelfile at project root)
2) **Thematic prompt** (problem-specific, optional, defined in application code)

The Modelfile serves as the single source of truth for base conversational behavior 
(warmth, brevity, respectful Christian spirituality). The application code only sends
dynamic, context-specific prompts (themes and modes) to the LLM.

## Where prompts live

- Base behavioral prompt: `Modelfile` at project root
- Themes registry: `services/prompts/themes/__init__.py`
- Theme implementation(s): `services/prompts/themes/*.py`
- Composition logic: `services/prompts/composer.py`

## Theme selection and persistence

- Theme selection helper: `services/theme_selector.py`
- The selected theme is persisted per user in `Profile.prompt_theme` (see `core/models.py`).
- `core/views.py` activates a theme when:
  - The detected intent matches a known theme, or
  - A keyword-based trigger matches later in the conversation (then intent is re-detected once to confirm).

## Adding a new theme

1. Create a new theme file in `services/prompts/themes/` (e.g., `anxiety.py`) with:
   - a stable `*_THEME_ID`
   - a `*_THEME_PROMPT_PTBR` string
2. Register it in `services/prompts/themes/__init__.py`
3. Update `services/theme_selector.py` to map the relevant intents and/or add safe keyword triggers
4. (Optional) Add tests similar to `services/test_prompt_composition.py`
