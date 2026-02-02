"""
Tests for the SimulationService, specifically the conversation analysis functionality.
"""

from unittest.mock import MagicMock, patch

from django.test import TestCase

from services.simulation_service import SimulationService


class SimulationServiceProfileTest(TestCase):
    """Tests for the create_simulation_profile method."""

    @patch("services.simulation_service.GroqService")
    @patch("services.simulation_service.Groq")
    def test_create_simulation_profile_has_random_gender(
        self, mock_groq_client, mock_groq_service
    ):
        """Test that created simulation profiles have a randomly assigned gender."""
        service = SimulationService("test-api-key")

        # Create multiple profiles to test randomness
        # With 20 profiles and 3 options, probability of all same is (1/3)^19 ≈ 0.00000000258
        profiles = [service.create_simulation_profile() for _ in range(20)]

        # Verify all profiles have a valid gender
        for profile in profiles:
            self.assertIn(
                profile.inferred_gender,
                ["male", "female", "unknown"],
                "Profile should have one of the valid gender values",
            )

        # Verify profiles have the simulation intent
        for profile in profiles:
            self.assertEqual(profile.detected_intent, "simulation")

        # Verify randomness: with 20 profiles, we should get more than one unique gender
        unique_genders = set(p.inferred_gender for p in profiles)
        self.assertGreater(
            len(unique_genders),
            1,
            "With 20 profiles, randomness should produce more than one gender value",
        )

        # Clean up
        for profile in profiles:
            profile.delete()


class SimulationServiceAnalysisTest(TestCase):
    """Tests for the analyze_conversation_emotions method."""

    @patch("services.simulation_service.GroqService")
    @patch("services.simulation_service.Groq")
    def test_analyze_conversation_structure_with_5_sections(
        self, mock_groq_client, mock_groq_service
    ):
        """Test that analysis returns all 5 required sections."""
        # Setup mock response with proper 5-section structure
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="""**1. O que funcionou bem**
- O Ouvinte manteve tom acolhedor
- Perguntas abertas foram utilizadas

**2. Pontos de possível erro de interpretação**
- Em alguns momentos, o Ouvinte pode ter inferido estados emocionais não explicitados

**3. Problemas de verbosidade e extensão das respostas**
- Algumas respostas foram mais longas que o necessário
- Múltiplas ideias foram introduzidas em uma única resposta

**4. O que poderia ter sido feito diferente**
- Usar respostas mais curtas (1-3 frases)
- Espelhar palavras do Buscador antes de expandir

**5. Ajustes recomendados para próximas interações**
- Reduzir extensão das respostas
- Respeitar brevidade como sinal válido"""
                )
            )
        ]

        mock_groq_instance = MagicMock()
        mock_groq_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_groq_instance

        # Create service and test conversation
        service = SimulationService("test-api-key")
        conversation = [
            {"role": "ROLE_A", "content": "Não sei..."},
            {"role": "ROLE_B", "content": "Entendo. Estou aqui para ouvir."},
        ]

        # Generate analysis
        analysis = service.analyze_conversation_emotions(conversation)

        # Verify all 5 sections are present
        self.assertIn("**1. O que funcionou bem**", analysis)
        self.assertIn("**2. Pontos de possível erro de interpretação**", analysis)
        self.assertIn("**3. Problemas de verbosidade e extensão das respostas**", analysis)
        self.assertIn("**4. O que poderia ter sido feito diferente**", analysis)
        self.assertIn("**5. Ajustes recomendados para próximas interações**", analysis)

    @patch("services.simulation_service.GroqService")
    @patch("services.simulation_service.Groq")
    def test_analyze_conversation_fallback_has_5_sections(
        self, mock_groq_client, mock_groq_service
    ):
        """Test that fallback analysis (on error) returns all 5 required sections."""
        # Setup mock to raise exception
        mock_groq_instance = MagicMock()
        mock_groq_instance.chat.completions.create.side_effect = Exception("API Error")
        mock_groq_client.return_value = mock_groq_instance

        # Create service and test conversation
        service = SimulationService("test-api-key")
        conversation = [
            {"role": "ROLE_A", "content": "Teste"},
            {"role": "ROLE_B", "content": "Teste resposta"},
        ]

        # Generate analysis (should use fallback)
        analysis = service.analyze_conversation_emotions(conversation)

        # Verify all 5 sections are present in fallback
        self.assertIn("**1. O que funcionou bem**", analysis)
        self.assertIn("**2. Pontos de possível erro de interpretação**", analysis)
        self.assertIn("**3. Problemas de verbosidade e extensão das respostas**", analysis)
        self.assertIn("**4. O que poderia ter sido feito diferente**", analysis)
        self.assertIn("**5. Ajustes recomendados para próximas interações**", analysis)

        # Verify fallback is substantial and not empty
        self.assertGreater(len(analysis), 100, "Fallback analysis should be substantial")

    @patch("services.simulation_service.GroqService")
    @patch("services.simulation_service.Groq")
    def test_analyze_conversation_includes_verbosity_focus(
        self, mock_groq_client, mock_groq_service
    ):
        """Test that the analysis prompt emphasizes verbosity and response length."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Test analysis"))
        ]
        mock_groq_instance = MagicMock()
        mock_groq_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_groq_instance

        # Create service and test conversation
        service = SimulationService("test-api-key")
        conversation = [
            {"role": "ROLE_A", "content": "Test"},
            {"role": "ROLE_B", "content": "Test response"},
        ]

        # Generate analysis
        service.analyze_conversation_emotions(conversation)

        # Verify the create method was called
        self.assertTrue(mock_groq_instance.chat.completions.create.called)

        # Get the actual call arguments
        call_args = mock_groq_instance.chat.completions.create.call_args
        messages = call_args[1]["messages"]

        # Verify system prompt includes verbosity-related content
        system_prompt = messages[0]["content"]
        self.assertIn("verbosidade", system_prompt.lower())
        self.assertIn("extensão das respostas", system_prompt.lower())
        self.assertIn("Problemas de verbosidade", system_prompt)

        # Verify user prompt includes verbosity focus
        user_prompt = messages[1]["content"]
        self.assertIn("verbosidade", user_prompt.lower())
        self.assertIn("5 seções", user_prompt.lower())
        self.assertIn("Problemas de verbosidade e extensão das respostas", user_prompt)
