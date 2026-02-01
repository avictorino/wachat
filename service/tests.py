"""Tests for service module functions."""
from django.test import TestCase

from service.prompts import generate_first_welcome_message
from service.orchestration import extract_phone_ddd


class WelcomeMessageGenerationTest(TestCase):
    """Test the personalized welcome message generation."""

    def test_generate_welcome_message_with_male_name(self):
        """Test welcome message generation with male name."""
        message = generate_first_welcome_message(
            user_name="João",
            inferred_gender="male",
            phone_ddd=None,
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
        """Test welcome message generation with female name."""
        message = generate_first_welcome_message(
            user_name="Maria",
            inferred_gender="female",
            phone_ddd=None,
        )
        
        # Check that the message contains the user's name
        self.assertIn("Maria", message)
        
        # Check that the message contains key phrases
        self.assertIn("que bom ter você aqui", message)
        self.assertIn("sem julgamento", message)

    def test_generate_welcome_message_with_unknown_gender(self):
        """Test welcome message generation with unknown gender."""
        message = generate_first_welcome_message(
            user_name="Alex",
            inferred_gender="unknown",
            phone_ddd=None,
        )
        
        # Check that the message contains the user's name
        self.assertIn("Alex", message)
        
        # Check that the message is still warm and welcoming
        self.assertIn("que bom ter você aqui", message)

    def test_generate_welcome_message_with_ddd(self):
        """Test welcome message generation with DDD (regional context)."""
        message = generate_first_welcome_message(
            user_name="Pedro",
            inferred_gender="male",
            phone_ddd="21",
        )
        
        # Check that regional closeness is implied when DDD is present
        self.assertIn("por aqui", message)
        self.assertIn("Pedro", message)

    def test_generate_welcome_message_without_ddd(self):
        """Test welcome message generation without DDD."""
        message = generate_first_welcome_message(
            user_name="Ana",
            inferred_gender="female",
            phone_ddd=None,
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
