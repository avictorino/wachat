"""
Tests for addiction-related intent detection and handling.

These tests verify that addiction-related intents (drogas, alcool, sexo, cigarro)
are properly detected, classified, and handled with empathy and seriousness,
treating them as real conditions rather than moral failures.
"""

from unittest.mock import Mock, patch

from django.test import TestCase


class AddictionIntentDetectionTest(TestCase):
    """Tests for detecting addiction-related intents."""

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    @patch("services.groq_service.Groq")
    def test_detect_drogas_intent(self, mock_groq_client):
        """Test that drug-related messages are detected as 'drogas' intent."""
        from services.groq_service import GroqService

        # Setup mock
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "drogas"

        mock_groq_instance = Mock()
        mock_groq_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_groq_instance

        # Create service and call method
        service = GroqService()
        intent = service.detect_intent("Estou lutando com dependência de drogas")

        # Verify correct intent was detected
        self.assertEqual(intent, "drogas")

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    @patch("services.groq_service.Groq")
    def test_detect_alcool_intent(self, mock_groq_client):
        """Test that alcohol-related messages are detected as 'alcool' intent."""
        from services.groq_service import GroqService

        # Setup mock
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "alcool"

        mock_groq_instance = Mock()
        mock_groq_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_groq_instance

        # Create service and call method
        service = GroqService()
        intent = service.detect_intent("Não consigo parar de beber")

        # Verify correct intent was detected
        self.assertEqual(intent, "alcool")

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    @patch("services.groq_service.Groq")
    def test_detect_sexo_intent(self, mock_groq_client):
        """Test that sexual compulsion messages are detected as 'sexo' intent."""
        from services.groq_service import GroqService

        # Setup mock
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "sexo"

        mock_groq_instance = Mock()
        mock_groq_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_groq_instance

        # Create service and call method
        service = GroqService()
        intent = service.detect_intent("Tenho compulsão sexual")

        # Verify correct intent was detected
        self.assertEqual(intent, "sexo")

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    @patch("services.groq_service.Groq")
    def test_detect_cigarro_intent(self, mock_groq_client):
        """Test that smoking-related messages are detected as 'cigarro' intent."""
        from services.groq_service import GroqService

        # Setup mock
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "cigarro"

        mock_groq_instance = Mock()
        mock_groq_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_groq_instance

        # Create service and call method
        service = GroqService()
        intent = service.detect_intent("Não consigo parar de fumar")

        # Verify correct intent was detected
        self.assertEqual(intent, "cigarro")

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    @patch("services.groq_service.Groq")
    def test_addiction_intents_in_valid_list(self, mock_groq_client):
        """Test that all addiction intents are in the valid intents list."""
        from services.groq_service import GroqService

        # Setup mock to return each addiction intent
        mock_response = Mock()
        mock_response.choices = [Mock()]

        mock_groq_instance = Mock()
        mock_groq_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_groq_instance

        service = GroqService()

        # Test each addiction intent
        addiction_intents = ["drogas", "alcool", "sexo", "cigarro"]
        for addiction_intent in addiction_intents:
            mock_response.choices[0].message.content = addiction_intent
            intent = service.detect_intent(f"Test message for {addiction_intent}")
            # Should not default to 'outro' - the intent should be accepted as valid
            self.assertEqual(intent, addiction_intent)


class AddictionThemeApproximationTest(TestCase):
    """Tests for approximating addiction-related themes."""

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    @patch("services.groq_service.Groq")
    def test_approximate_drogas_variations(self, mock_groq_client):
        """Test that drug-related variations map to 'drogas'."""
        from services.groq_service import GroqService

        # Setup mock
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "drogas"

        mock_groq_instance = Mock()
        mock_groq_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_groq_instance

        service = GroqService()

        # Test variations
        variations = ["cocaína", "maconha", "crack", "dependência química", "vício"]
        for variation in variations:
            theme = service.approximate_theme(variation)
            self.assertEqual(theme, "drogas")

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    @patch("services.groq_service.Groq")
    def test_approximate_alcool_variations(self, mock_groq_client):
        """Test that alcohol-related variations map to 'alcool'."""
        from services.groq_service import GroqService

        # Setup mock
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "alcool"

        mock_groq_instance = Mock()
        mock_groq_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_groq_instance

        service = GroqService()

        # Test variations
        variations = ["bebida", "álcool", "alcoolismo", "beber"]
        for variation in variations:
            theme = service.approximate_theme(variation)
            self.assertEqual(theme, "alcool")

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    @patch("services.groq_service.Groq")
    def test_approximate_cigarro_variations(self, mock_groq_client):
        """Test that smoking-related variations map to 'cigarro'."""
        from services.groq_service import GroqService

        # Setup mock
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "cigarro"

        mock_groq_instance = Mock()
        mock_groq_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_groq_instance

        service = GroqService()

        # Test variations
        variations = ["fumo", "tabaco", "tabagismo", "fumar"]
        for variation in variations:
            theme = service.approximate_theme(variation)
            self.assertEqual(theme, "cigarro")


class AddictionIntentResponseTest(TestCase):
    """Tests for generating responses to addiction-related intents."""

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    @patch("services.groq_service.Groq")
    def test_generate_response_for_drogas_intent(self, mock_groq_client):
        """Test that response generation works for 'drogas' intent."""
        from services.groq_service import GroqService

        # Setup mock
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[
            0
        ].message.content = "Entendo que você está passando por uma luta difícil."

        mock_groq_instance = Mock()
        mock_groq_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_groq_instance

        service = GroqService()
        response_messages = service.generate_intent_response(
            user_message="Estou lutando com drogas",
            intent="drogas",
            name="João",
            inferred_gender="male",
        )

        # Verify response was generated
        self.assertIsNotNone(response_messages)
        self.assertIsInstance(response_messages, list)
        self.assertTrue(len(response_messages) > 0)

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    @patch("services.groq_service.Groq")
    def test_addiction_intent_guidance_exists(self, mock_groq_client):
        """Test that intent guidance exists for all addiction intents."""
        from services.groq_service import GroqService

        # Setup mock
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"

        mock_groq_instance = Mock()
        mock_groq_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_groq_instance

        service = GroqService()

        # Test that each addiction intent has guidance
        addiction_intents = ["drogas", "alcool", "sexo", "cigarro"]
        for addiction_intent in addiction_intents:
            # This should not raise an error or default to 'outro' guidance
            response_messages = service.generate_intent_response(
                user_message=f"Test message for {addiction_intent}",
                intent=addiction_intent,
                name="Test User",
                inferred_gender="unknown",
            )
            self.assertIsNotNone(response_messages)

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    @patch("services.groq_service.Groq")
    def test_addiction_response_guidance_contains_empathy_keywords(
        self, mock_groq_client
    ):
        """Test that addiction intents automatically layer the addiction theme prompt."""
        from services.groq_service import GroqService

        # We can directly check the intent_guidance dict in the method
        # by inspecting the system prompt sent to the LLM
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Empathetic response"

        mock_groq_instance = Mock()
        mock_groq_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_groq_instance

        service = GroqService()

        # Generate response for each addiction intent
        addiction_intents = ["drogas", "alcool", "sexo", "cigarro"]
        for addiction_intent in addiction_intents:
            service.generate_intent_response(
                user_message=f"Test for {addiction_intent}",
                intent=addiction_intent,
                name="Test",
                inferred_gender="unknown",
            )

            call_args = mock_groq_instance.chat.completions.create.call_args
            messages = call_args.kwargs["messages"]
            system_message = next(m for m in messages if m["role"] == "system")
            system_content = system_message["content"].lower()

            # Theme marker
            self.assertIn("tema: drogas / álcool / cigarro / vícios", system_content)

            # Core non-judgment / condition framing
            required_phrases = [
                "condição real",
                "não como falha moral",
                "sem julgamento",
                "não envergonhe",
            ]
            self.assertTrue(
                any(p in system_content for p in required_phrases),
                f"System prompt for {addiction_intent} should contain addiction theme guidance",
            )
