"""
Tests for the Input Sanitization Service.

This module tests the InputSanitizer to ensure it properly:
- Detects harmful content in Portuguese
- Replaces harmful terms with neutral placeholders
- Preserves safe content
- Handles edge cases gracefully
- Logs detections appropriately
"""

from unittest.mock import patch

from django.test import TestCase

from services.input_sanitizer import (
    InputSanitizer,
    get_sanitizer,
    sanitize_input,
)


class InputSanitizerTest(TestCase):
    """Tests for the InputSanitizer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.sanitizer = InputSanitizer()

    def test_sanitize_clean_input(self):
        """Test that clean input passes through unchanged."""
        clean_texts = [
            "Olá, como você está?",
            "Preciso de ajuda com minha vida espiritual",
            "Estou me sentindo sozinho",
            "Quero conversar sobre fé",
            "O que significa amor ao próximo?",
        ]
        
        for text in clean_texts:
            result = self.sanitizer.sanitize(text)
            self.assertEqual(result, text, f"Clean text was modified: {text}")

    def test_sanitize_sexual_content(self):
        """Test detection and sanitization of sexual content."""
        test_cases = [
            ("Quero falar sobre sexo", "Quero falar sobre [tema sensível]"),
            ("Estou tendo problemas sexuais", "Estou tendo problemas [tema sensível]"),
            ("Vi pornografia ontem", "Vi [tema sensível] ontem"),
            ("Fui vítima de abuso sexual", "Fui vítima de [tema sensível]"),
        ]
        
        for input_text, expected_output in test_cases:
            result = self.sanitizer.sanitize(input_text)
            self.assertIn("[tema sensível]", result)
            self.assertNotEqual(result, input_text)

    def test_sanitize_death_content(self):
        """Test detection and sanitization of death-related content."""
        test_cases = [
            ("Penso em morrer", "Penso em [tema sensível]"),
            ("Quero me matar", "Quero me [tema sensível]"),
            ("Tive pensamentos suicidas", "Tive pensamentos [tema sensível]"),
            ("Meu pai faleceu", "Meu pai [tema sensível]"),
        ]
        
        for input_text, expected_output in test_cases:
            result = self.sanitizer.sanitize(input_text)
            self.assertIn("[tema sensível]", result)
            self.assertNotEqual(result, input_text)

    def test_sanitize_controversial_content(self):
        """Test detection and sanitization of controversial topics."""
        test_cases = [
            ("O que pensa sobre aborto?", "O que pensa sobre [tema sensível]?"),
            ("Sou contra o fascismo", "Sou contra o [tema sensível]"),
            ("Há muito racismo no mundo", "Há muito [tema sensível] no mundo"),
            ("Terrorismo é terrível", "[tema sensível] é terrível"),
        ]
        
        for input_text, expected_output in test_cases:
            result = self.sanitizer.sanitize(input_text)
            self.assertIn("[tema sensível]", result)
            self.assertNotEqual(result, input_text)

    def test_sanitize_multiple_harmful_terms(self):
        """Test sanitization when multiple harmful terms are present."""
        input_text = "Quero falar sobre sexo, morte e aborto"
        result = self.sanitizer.sanitize(input_text)
        
        # All harmful terms should be replaced
        self.assertIn("[tema sensível]", result)
        self.assertNotIn("sexo", result.lower())
        self.assertNotIn("morte", result.lower())
        self.assertNotIn("aborto", result.lower())

    def test_sanitize_case_insensitive(self):
        """Test that sanitization is case-insensitive."""
        test_cases = [
            "SEXO",
            "Sexo",
            "sexo",
            "SeXo",
        ]
        
        for text in test_cases:
            result = self.sanitizer.sanitize(text)
            self.assertEqual(result, "[tema sensível]")

    def test_sanitize_preserves_context(self):
        """Test that sanitization preserves surrounding context."""
        input_text = "Olá, estou com dúvidas sobre sexo, pode me ajudar?"
        result = self.sanitizer.sanitize(input_text)
        
        # Should preserve the greeting and structure
        self.assertIn("Olá", result)
        self.assertIn("pode me ajudar", result)
        self.assertIn("[tema sensível]", result)
        self.assertNotIn("sexo", result.lower())

    def test_sanitize_empty_input(self):
        """Test handling of empty input."""
        self.assertEqual(self.sanitizer.sanitize(""), "")
        self.assertEqual(self.sanitizer.sanitize(None), "")

    def test_sanitize_whitespace_only(self):
        """Test handling of whitespace-only input."""
        result = self.sanitizer.sanitize("   ")
        self.assertEqual(result.strip(), "")

    def test_sanitize_non_string_input(self):
        """Test handling of non-string input."""
        # Should convert to string and process
        result = self.sanitizer.sanitize(123)
        self.assertEqual(result, "123")

    @patch('services.input_sanitizer.logger')
    def test_logging_when_harmful_content_detected(self, mock_logger):
        """Test that detections are logged when harmful content is found."""
        self.sanitizer.sanitize("Quero falar sobre sexo", log_detections=True)
        
        # Logger should have been called
        self.assertTrue(mock_logger.info.called)
        call_args = mock_logger.info.call_args
        self.assertIn("Input sanitization performed", call_args[0][0])

    @patch('services.input_sanitizer.logger')
    def test_no_logging_when_clean_input(self, mock_logger):
        """Test that nothing is logged for clean input."""
        self.sanitizer.sanitize("Olá, como você está?", log_detections=True)
        
        # Logger should not have been called for info
        self.assertFalse(mock_logger.info.called)

    @patch('services.input_sanitizer.logger')
    def test_logging_can_be_disabled(self, mock_logger):
        """Test that logging can be disabled."""
        self.sanitizer.sanitize("Quero falar sobre sexo", log_detections=False)
        
        # Logger should not have been called
        self.assertFalse(mock_logger.info.called)

    def test_portuguese_variations(self):
        """Test detection of Portuguese language variations."""
        test_cases = [
            "transando",  # verb variation
            "sexual",     # adjective
            "sexuais",    # plural adjective
            "morrer",     # verb infinitive
            "morrendo",   # gerund
            "morreu",     # past tense
        ]
        
        for text in test_cases:
            result = self.sanitizer.sanitize(text)
            self.assertEqual(result, "[tema sensível]", 
                           f"Failed to detect variation: {text}")

    def test_word_boundary_matching(self):
        """Test that only complete words are matched, not substrings."""
        # "sexta" (Friday) should be safe, it's a different word from "sexo"
        text1 = "Vamos nos encontrar na sexta-feira"
        result1 = self.sanitizer.sanitize(text1)
        # sexta should be preserved because word boundaries prevent matching
        self.assertEqual(result1, text1)
        
        # But a sentence with actual "sexo" should be sanitized
        text2 = "Vamos falar sobre sexo na sexta"
        result2 = self.sanitizer.sanitize(text2)
        self.assertIn("[tema sensível]", result2)
        self.assertIn("sexta", result2)  # sexta should remain

    def test_consecutive_placeholders_are_cleaned(self):
        """Test that consecutive placeholders are merged."""
        input_text = "sexo e morte e aborto"
        result = self.sanitizer.sanitize(input_text)
        
        # Should not have multiple consecutive placeholders
        self.assertNotIn("[tema sensível] [tema sensível] [tema sensível]", result)

    def test_error_handling_returns_original_text(self):
        """Test that errors during sanitization don't break the flow."""
        # This test verifies the error handling mechanism
        # In a real error scenario, the original text should be returned
        # This is a safety measure to avoid breaking the application
        with patch.object(self.sanitizer, '_detect_harmful_content', 
                         side_effect=Exception("Test error")):
            result = self.sanitizer.sanitize("Test input")
            # Should return original text if error occurs
            self.assertEqual(result, "Test input")


class GlobalSanitizerTest(TestCase):
    """Tests for the global sanitizer instance functions."""

    def test_get_sanitizer_returns_singleton(self):
        """Test that get_sanitizer returns the same instance."""
        sanitizer1 = get_sanitizer()
        sanitizer2 = get_sanitizer()
        
        self.assertIs(sanitizer1, sanitizer2)
        self.assertIsInstance(sanitizer1, InputSanitizer)

    def test_sanitize_input_function(self):
        """Test the convenience sanitize_input function."""
        result = sanitize_input("Quero falar sobre sexo")
        
        self.assertIn("[tema sensível]", result)
        self.assertNotIn("sexo", result.lower())

    def test_sanitize_input_with_clean_text(self):
        """Test sanitize_input with clean text."""
        clean_text = "Olá, como você está?"
        result = sanitize_input(clean_text)
        
        self.assertEqual(result, clean_text)


class IntegrationTest(TestCase):
    """Integration tests for realistic scenarios."""

    def test_realistic_user_message(self):
        """Test sanitization of realistic user messages."""
        sanitizer = InputSanitizer()
        
        # Simulate a user trying to discuss a sensitive topic
        input_text = "Olá, estou passando por um momento difícil e penso muito em morte. Também tenho dúvidas sobre sexualidade."
        result = sanitizer.sanitize(input_text)
        
        # Should sanitize harmful parts but keep the rest
        self.assertIn("Olá", result)
        self.assertIn("momento difícil", result)
        self.assertIn("[tema sensível]", result)
        self.assertNotIn("morte", result.lower())
        self.assertNotIn("sexualidade", result.lower())

    def test_emotional_spiritual_context_preserved(self):
        """Test that emotional and spiritual language is preserved."""
        sanitizer = InputSanitizer()
        
        safe_texts = [
            "Sinto uma dor profunda no coração",
            "Preciso de orientação espiritual",
            "Minha fé está abalada",
            "Como encontro paz interior?",
            "Estou em busca de sentido",
        ]
        
        for text in safe_texts:
            result = sanitizer.sanitize(text)
            # Should remain completely unchanged
            self.assertEqual(result, text)

    def test_name_with_problematic_substring(self):
        """Test that names containing substrings don't get sanitized incorrectly."""
        sanitizer = InputSanitizer()
        
        # Note: Our word boundary patterns should handle this correctly
        # Testing to ensure proper regex implementation
        text = "Meu nome é Moreno Silva"  # "More" is substring of "morte"
        result = sanitizer.sanitize(text)
        
        # Name should be preserved (word boundaries prevent false positives)
        self.assertIn("Moreno", result)
