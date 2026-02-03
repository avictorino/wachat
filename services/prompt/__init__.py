"""
LLM service layer with unified prompts.

This package contains:
- All LLM service implementations (GroqService, OllamaService)
- LLM service interface and factory
- Unified prompts for all LLM operations
- Prompt composition for conversation flows
"""

from services.prompt.composer import PromptComposer
from services.prompt.llm_factory import get_llm_service
from services.prompt.llm_service_interface import LLMServiceInterface

__all__ = ["PromptComposer", "get_llm_service", "LLMServiceInterface"]
