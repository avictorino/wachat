# Prompt Architecture

This project uses Ollama Modelfile for system prompts.

## Where prompts live

- System prompt: `model/Modelfile` (for Ollama model)
- RAG system prompt: `model/RagModelfile` (for RAG-based context retrieval)

## How it works

The system prompt is defined in the Ollama Modelfile and loaded when the model is initialized. This approach provides:

- Single source of truth for behavioral rules and system prompts
- No runtime composition needed
- Consistent behavior across all conversations

## Modifying the system prompt

To modify the system prompt:

1. Edit `model/Modelfile` directly
2. Rebuild the Ollama model with the new Modelfile
3. Test the changes in conversation

Note: The `Profile.prompt_theme` field is retained for potential future use but is not currently active in the prompt system.
