"""Prompt building blocks for conversation flows.

This package handles dynamic, context-specific prompts:
- Optional thematic prompts (for specific topics like addiction)
- A composer that assembles prompts for specific response modes

The base behavioral prompt is defined in the Modelfile at the project root,
which serves as the single source of truth for conversational behavior.
"""

from services.prompts.composer import PromptComposer

__all__ = ["PromptComposer"]
