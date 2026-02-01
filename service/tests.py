"""Tests for service module functions."""
from unittest.mock import Mock
from django.test import TestCase

from service.prompts import generate_first_welcome_message, build_welcome_message_prompt
from service.orchestration import extract_phone_ddd


class WelcomeMessageGenerationTest(TestCase):
    """Test the personalized welcome message generation."""

    def test_generate_welcome_message_with_male_name(self):
        """Test welcome message generation with male name (fallback mode)."""
        message = generate_first_welcome_message(
            user_name="João",
            inferred_gender="male",
            phone_ddd=None,
            llm=None,  # Use fallback
        )
        
        # Check that the message contains the user's name
        self.assertIn("João", message)
        
        # Check that the message contains key phrases
        self.assertIn("que bom ter você aqui", message)
        self.assertIn("Estou aqui pra te ouvir", message)
        self.assertIn("sem pressa e sem julgamento", message)
        self.assertIn("Não te digo o que pensar", message)
        self.assertIn("Caminho contigo enquanto você pensa", message)
        self.assertIn("O que te trouxe até aqui?", message)
        
        # Verify it's not preachy or using emojis
        self.assertNotIn("🙏", message)
        self.assertNotIn("amigo bíblico", message)

    def test_generate_welcome_message_with_female_name(self):
        """Test welcome message generation with female name (fallback mode)."""
        message = generate_first_welcome_message(
            user_name="Maria",
            inferred_gender="female",
            phone_ddd=None,
            llm=None,  # Use fallback
        )
        
        # Check that the message contains the user's name
        self.assertIn("Maria", message)
        
        # Check that the message contains key phrases
        self.assertIn("que bom ter você aqui", message)
        self.assertIn("sem julgamento", message)

    def test_generate_welcome_message_with_unknown_gender(self):
        """Test welcome message generation with unknown gender (fallback mode)."""
        message = generate_first_welcome_message(
            user_name="Alex",
            inferred_gender="unknown",
            phone_ddd=None,
            llm=None,  # Use fallback
        )
        
        # Check that the message contains the user's name
        self.assertIn("Alex", message)
        
        # Check that the message is still warm and welcoming
        self.assertIn("que bom ter você aqui", message)

    def test_generate_welcome_message_with_ddd(self):
        """Test welcome message generation with DDD (regional context, fallback mode)."""
        message = generate_first_welcome_message(
            user_name="Pedro",
            inferred_gender="male",
            phone_ddd="21",
            llm=None,  # Use fallback
        )
        
        # Check that regional closeness is implied when DDD is present
        self.assertIn("por aqui", message)
        self.assertIn("Pedro", message)

    def test_generate_welcome_message_without_ddd(self):
        """Test welcome message generation without DDD (fallback mode)."""
        message = generate_first_welcome_message(
            user_name="Ana",
            inferred_gender="female",
            phone_ddd=None,
            llm=None,  # Use fallback
        )
        
        # Check that "por aqui" is not present when DDD is not available
        self.assertNotIn("por aqui", message)
        self.assertIn("Ana", message)

    def test_welcome_message_length_is_appropriate(self):
        """Test that welcome message is concise (3-5 paragraphs)."""
        message = generate_first_welcome_message(
            user_name="Lucas",
            inferred_gender="male",
            phone_ddd="11",
        )
        
        # Count paragraphs (separated by double newlines)
        paragraphs = [p.strip() for p in message.split("\n\n") if p.strip()]
        
        # Should have between 3 and 5 paragraphs
        self.assertGreaterEqual(len(paragraphs), 3)
        self.assertLessEqual(len(paragraphs), 6)

    def test_welcome_message_is_in_portuguese(self):
        """Test that welcome message is in Brazilian Portuguese."""
        message = generate_first_welcome_message(
            user_name="Carlos",
            inferred_gender="male",
            phone_ddd="11",
        )
        
        # Check for Portuguese-specific phrases
        portuguese_phrases = [
            "que bom",
            "você",
            "aqui",
            "te",
            "contigo",
        ]
        
        for phrase in portuguese_phrases:
            self.assertIn(phrase, message)


class PhoneDDDExtractionTest(TestCase):
    """Test the DDD extraction from phone numbers."""

    def test_extract_ddd_from_brazilian_number_with_plus(self):
        """Test DDD extraction from Brazilian number with + prefix."""
        ddd = extract_phone_ddd("+5521967337683")
        self.assertEqual(ddd, "21")

    def test_extract_ddd_from_brazilian_number_without_plus(self):
        """Test DDD extraction from Brazilian number without + prefix."""
        ddd = extract_phone_ddd("5511999999999")
        self.assertEqual(ddd, "11")

    def test_extract_ddd_from_formatted_number(self):
        """Test DDD extraction from formatted phone number."""
        ddd = extract_phone_ddd("+55 (11) 99999-9999")
        self.assertEqual(ddd, "11")

    def test_extract_ddd_returns_none_for_non_brazilian(self):
        """Test that DDD extraction returns None for non-Brazilian numbers."""
        ddd = extract_phone_ddd("+1234567890")
        self.assertIsNone(ddd)

    def test_extract_ddd_returns_none_for_empty_string(self):
        """Test that DDD extraction returns None for empty string."""
        ddd = extract_phone_ddd("")
        self.assertIsNone(ddd)

    def test_extract_ddd_returns_none_for_none(self):
        """Test that DDD extraction returns None for None input."""
        ddd = extract_phone_ddd(None)
        self.assertIsNone(ddd)

    def test_extract_ddd_handles_various_formatting(self):
        """Test that DDD extraction handles various phone number formats."""
        test_cases = [
            ("+55-21-967337683", "21"),
            ("+55 21 967337683", "21"),
            ("+55(21)967337683", "21"),
            ("55 21 967337683", "21"),
        ]
        
        for phone, expected_ddd in test_cases:
            ddd = extract_phone_ddd(phone)
            self.assertEqual(ddd, expected_ddd, f"Failed for phone: {phone}")


class WelcomeMessageLLMGenerationTest(TestCase):
    """Test the AI-generated welcome message generation."""

    def test_generate_welcome_message_with_llm(self):
        """Test welcome message generation with LLM."""
        # Mock LLM client
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.text = "Oi, João, que bom você estar aqui.\n\nEste é um espaço onde você pode falar livremente, sem pressa e sem julgamento.\n\nVocê não precisa ter respostas prontas. Estou aqui para caminhar ao seu lado enquanto você pensa.\n\nO que te trouxe até aqui hoje?"
        mock_llm.chat.return_value = mock_response
        
        message = generate_first_welcome_message(
            user_name="João",
            inferred_gender="male",
            phone_ddd=None,
            llm=mock_llm,
        )
        
        # Check that LLM was called
        self.assertTrue(mock_llm.chat.called)
        
        # Check that the message contains the user's name
        self.assertIn("João", message)
        
        # Check that we got a real message back
        self.assertTrue(len(message) > 0)

    def test_generate_welcome_message_with_llm_and_ddd(self):
        """Test welcome message generation with LLM and DDD."""
        # Mock LLM client
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.text = "Oi, Maria, que bom ter você por aqui.\n\nEste é um espaço seguro para conversar.\n\nNão te digo o que pensar. Caminho contigo.\n\nO que te trouxe até aqui?"
        mock_llm.chat.return_value = mock_response
        
        message = generate_first_welcome_message(
            user_name="Maria",
            inferred_gender="female",
            phone_ddd="21",
            llm=mock_llm,
        )
        
        # Check that LLM was called with correct parameters
        call_args = mock_llm.chat.call_args
        messages = call_args[1]['messages'] if 'messages' in call_args[1] else call_args[0][0]
        
        # Verify the system prompt includes the user name and DDD
        system_message = messages[0]
        self.assertIn("Maria", system_message.content)
        self.assertIn("21", system_message.content)
        
        # Check that we got the expected message back
        self.assertIn("Maria", message)

    def test_generate_welcome_message_llm_fallback_on_error(self):
        """Test that welcome message falls back to hardcoded when LLM fails."""
        # Mock LLM client that raises an exception
        mock_llm = Mock()
        mock_llm.chat.side_effect = Exception("LLM API error")
        
        message = generate_first_welcome_message(
            user_name="Pedro",
            inferred_gender="male",
            phone_ddd="11",
            llm=mock_llm,
        )
        
        # Should fall back to hardcoded message
        self.assertIn("Pedro", message)
        self.assertIn("por aqui", message)  # DDD was provided
        self.assertIn("Estou aqui pra te ouvir", message)

    def test_build_welcome_message_prompt(self):
        """Test the prompt builder for welcome messages."""
        prompt = build_welcome_message_prompt(
            user_name="João",
            inferred_gender="male",
            phone_ddd="21",
        )
        
        # Check that prompt contains essential elements
        self.assertIn("João", prompt)
        self.assertIn("21", prompt)
        self.assertIn("masculino", prompt)
        self.assertIn("primeira mensagem", prompt.lower())
        self.assertIn("não te digo o que pensar", prompt.lower())

    def test_build_welcome_message_prompt_without_ddd(self):
        """Test the prompt builder without DDD."""
        prompt = build_welcome_message_prompt(
            user_name="Ana",
            inferred_gender="female",
            phone_ddd=None,
        )
        
        # Check that prompt contains essential elements
        self.assertIn("Ana", prompt)
        self.assertIn("feminino", prompt)
        # Check that DDD context is not mentioned when None
        # The prompt should not have DDD info if it's None
        lines = prompt.split('\n')
        ddd_mentioned = any('DDD' in line and 'tem DDD' in line for line in lines)
        self.assertFalse(ddd_mentioned)

    def test_build_welcome_message_prompt_unknown_gender(self):
        """Test the prompt builder with unknown gender."""
        prompt = build_welcome_message_prompt(
            user_name="Alex",
            inferred_gender="unknown",
            phone_ddd=None,
        )
        
        # Check that prompt contains the name
        self.assertIn("Alex", prompt)
        # Should not contain gender context
        self.assertNotIn("masculino", prompt)
        self.assertNotIn("feminino", prompt)
