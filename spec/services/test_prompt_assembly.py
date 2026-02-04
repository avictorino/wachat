"""
Tests for the strict prompt assembly function.

These tests verify that:
- Theme context is included ONLY when theme_id is provided
- Conversation history is included in chronological order (raw format)
- Current user message is included at the end
- The exact format "CONTEXTO ATUAL DA CONVERSA" is used when theme exists
- The strict order is maintained: theme → history → current message
"""

from django.test import TestCase

from services.prompts.composer import PromptComposer


class PromptAssemblyTest(TestCase):
    def test_assemble_prompt_without_theme(self):
        """Test prompt assembly when no theme is provided."""
        conversation_history = [
            {"role": "user", "content": "Estou com medo"},
            {"role": "assistant", "content": "Como posso te ajudar?"},
        ]
        current_message = "Não sei o que fazer"

        result = PromptComposer.assemble_final_prompt(
            theme_id=None,
            conversation_history=conversation_history,
            current_user_message=current_message,
        )

        # Should not contain theme context or header
        self.assertNotIn("CONTEXTO ATUAL DA CONVERSA", result)

        # Should contain conversation history in order
        self.assertIn("user: Estou com medo", result)
        self.assertIn("assistant: Como posso te ajudar?", result)

        # Should contain current message
        self.assertIn("user: Não sei o que fazer", result)

        # Verify order (history before current message)
        idx_history = result.index("user: Estou com medo")
        idx_current = result.index("user: Não sei o que fazer")
        self.assertLess(idx_history, idx_current)

    def test_assemble_prompt_with_theme(self):
        """Test prompt assembly with drug addiction theme."""
        conversation_history = [
            {"role": "user", "content": "Estou viciado"},
            {"role": "assistant", "content": "Entendo sua dor"},
        ]
        current_message = "Como sair disso?"

        result = PromptComposer.assemble_final_prompt(
            theme_id="drug_addiction",
            conversation_history=conversation_history,
            current_user_message=current_message,
        )

        # Should contain theme context
        self.assertIn("uso de drogas ou dependência química", result)

        # Should contain the exact format header
        self.assertIn("CONTEXTO ATUAL DA CONVERSA", result)

        # Should contain conversation history
        self.assertIn("user: Estou viciado", result)
        self.assertIn("assistant: Entendo sua dor", result)

        # Should contain current message
        self.assertIn("user: Como sair disso?", result)

        # Verify strict order: theme → header → history → current
        idx_theme = result.index("uso de drogas")
        idx_header = result.index("CONTEXTO ATUAL DA CONVERSA")
        idx_history = result.index("user: Estou viciado")
        idx_current = result.index("user: Como sair disso?")

        self.assertLess(idx_theme, idx_header)
        self.assertLess(idx_header, idx_history)
        self.assertLess(idx_history, idx_current)

    def test_assemble_prompt_empty_history(self):
        """Test prompt assembly with empty conversation history."""
        current_message = "Primeira mensagem"

        result = PromptComposer.assemble_final_prompt(
            theme_id=None,
            conversation_history=[],
            current_user_message=current_message,
        )

        # Should only contain current message
        self.assertIn("user: Primeira mensagem", result)
        self.assertNotIn("CONTEXTO ATUAL DA CONVERSA", result)

    def test_assemble_prompt_with_theme_empty_history(self):
        """Test prompt assembly with theme but empty history."""
        current_message = "Primeira mensagem"

        result = PromptComposer.assemble_final_prompt(
            theme_id="drug_addiction",
            conversation_history=[],
            current_user_message=current_message,
        )

        # Should contain theme and header
        self.assertIn("uso de drogas ou dependência química", result)
        self.assertIn("CONTEXTO ATUAL DA CONVERSA", result)

        # Should contain current message
        self.assertIn("user: Primeira mensagem", result)

        # Verify order
        idx_header = result.index("CONTEXTO ATUAL DA CONVERSA")
        idx_current = result.index("user: Primeira mensagem")
        self.assertLess(idx_header, idx_current)

    def test_assemble_prompt_chronological_order(self):
        """Test that conversation history maintains chronological order."""
        conversation_history = [
            {"role": "user", "content": "Mensagem 1"},
            {"role": "assistant", "content": "Resposta 1"},
            {"role": "user", "content": "Mensagem 2"},
            {"role": "assistant", "content": "Resposta 2"},
        ]
        current_message = "Mensagem 3"

        result = PromptComposer.assemble_final_prompt(
            theme_id=None,
            conversation_history=conversation_history,
            current_user_message=current_message,
        )

        # Find positions of all messages
        idx_msg1 = result.index("Mensagem 1")
        idx_resp1 = result.index("Resposta 1")
        idx_msg2 = result.index("Mensagem 2")
        idx_resp2 = result.index("Resposta 2")
        idx_msg3 = result.index("Mensagem 3")

        # Verify chronological order
        self.assertLess(idx_msg1, idx_resp1)
        self.assertLess(idx_resp1, idx_msg2)
        self.assertLess(idx_msg2, idx_resp2)
        self.assertLess(idx_resp2, idx_msg3)

    def test_assemble_prompt_format_when_theme_exists(self):
        """Test exact format structure when theme exists."""
        conversation_history = [
            {"role": "user", "content": "Test message"},
        ]
        current_message = "Current test"

        result = PromptComposer.assemble_final_prompt(
            theme_id="drug_addiction",
            conversation_history=conversation_history,
            current_user_message=current_message,
        )

        # Split by lines to verify structure
        lines = result.split("\n")

        # Find the header line
        header_idx = None
        for i, line in enumerate(lines):
            if "CONTEXTO ATUAL DA CONVERSA" in line:
                header_idx = i
                break

        self.assertIsNotNone(header_idx, "Header not found in prompt")

        # Verify there's content before the header (theme)
        self.assertGreater(header_idx, 0, "Theme should come before header")

        # Verify there's content after the header (history + current message)
        self.assertLess(header_idx, len(lines) - 1, "History should come after header")
