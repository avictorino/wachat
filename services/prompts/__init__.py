"""Prompt building blocks for LLM-based conversation flows.

This package is intentionally small and explicit:
- A single shared base prompt (generalist)
- Optional thematic prompts layered on top
- A composer that assembles prompts for specific response modes
"""

from services.prompts.composer import PromptComposer

__all__ = ["PromptComposer"]
