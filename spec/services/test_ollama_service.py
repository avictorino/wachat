"""
Tests for OllamaService.

This module tests the OllamaService implementation, including:
- Request payload formatting
- Error handling for connection failures
- Response parsing
"""

import json
from unittest.mock import Mock, patch

import requests
from django.test import TestCase

from services.ollama_service import OllamaService


class OllamaServiceTest(TestCase):
    """Tests for OllamaService."""

    @patch.dict("os.environ", {"OLLAMA_BASE_URL": "http://localhost:11434", "OLLAMA_MODEL": "llama3.1"})
    def test_ollama_service_initialization_with_defaults(self):
        """Test that OllamaService initializes with default configuration."""
        service = OllamaService()
        self.assertEqual(service.base_url, "http://localhost:11434")
        self.assertEqual(service.model, "llama3.1")
        self.assertEqual(service.api_url, "http://localhost:11434/api/chat")

    @patch.dict("os.environ", {"OLLAMA_BASE_URL": "http://custom:8080", "OLLAMA_MODEL": "llama2"})
    def test_ollama_service_initialization_with_custom_config(self):
        """Test that OllamaService initializes with custom configuration."""
        service = OllamaService()
        self.assertEqual(service.base_url, "http://custom:8080")
        self.assertEqual(service.model, "llama2")
        self.assertEqual(service.api_url, "http://custom:8080/api/chat")

    @patch("services.ollama_service.requests.post")
    @patch("services.ollama_service.sanitize_input")
    def test_infer_gender_sends_correct_payload(self, mock_sanitize, mock_post):
        """Test that infer_gender sends correct payload to Ollama API."""
        # Setup mocks
        mock_sanitize.return_value = "João Silva"
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "male"}
        }
        mock_post.return_value = mock_response

        # Create service and call method
        service = OllamaService()
        result = service.infer_gender("João Silva")

        # Verify result
        self.assertEqual(result, "male")

        # Verify the request was made with correct payload
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        
        # Check URL
        self.assertEqual(call_args[0][0], "http://localhost:11434/api/chat")
        
        # Check payload structure
        payload = call_args[1]["json"]
        self.assertEqual(payload["model"], "wachat-v9")  # Model from .env
        self.assertEqual(payload["stream"], False)
        self.assertIn("messages", payload)
        self.assertEqual(len(payload["messages"]), 2)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertIn("João Silva", payload["messages"][1]["content"])
        
        # Check temperature
        self.assertEqual(payload["options"]["temperature"], 0.3)

    @patch("services.ollama_service.requests.post")
    @patch("services.ollama_service.sanitize_input")
    def test_generate_welcome_message_sends_correct_payload(self, mock_sanitize, mock_post):
        """Test that generate_welcome_message sends correct payload."""
        # Setup mocks
        mock_sanitize.return_value = "Maria Santos"
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "Olá, Maria! Bem-vinda ao nosso espaço."}
        }
        mock_post.return_value = mock_response

        # Create service and call method
        service = OllamaService()
        result = service.generate_welcome_message("Maria Santos", inferred_gender="female")

        # Verify result
        self.assertIn("Maria", result)

        # Verify the request payload
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["options"]["temperature"], 0.8)
        self.assertEqual(payload["options"]["num_predict"], 300)

    @patch("services.ollama_service.requests.post")
    def test_detect_intent_sends_correct_payload(self, mock_post):
        """Test that detect_intent sends correct payload."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "ansiedade"}
        }
        mock_post.return_value = mock_response

        service = OllamaService()
        result = service.detect_intent("Estou com muito medo do futuro")

        self.assertEqual(result, "ansiedade")

        # Verify temperature and max_tokens
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["options"]["temperature"], 0.3)
        self.assertEqual(payload["options"]["num_predict"], 20)

    @patch("services.ollama_service.requests.post")
    @patch("services.ollama_service.sanitize_input")
    def test_approximate_theme_sends_correct_payload(self, mock_sanitize, mock_post):
        """Test that approximate_theme sends correct payload."""
        mock_sanitize.return_value = "enfermidade"
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "doenca"}
        }
        mock_post.return_value = mock_response

        service = OllamaService()
        result = service.approximate_theme("enfermidade")

        self.assertEqual(result, "doenca")

        # Verify temperature
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["options"]["temperature"], 0.2)

    @patch("services.ollama_service.requests.post")
    def test_ollama_service_handles_connection_error(self, mock_post):
        """Test that OllamaService handles connection errors gracefully."""
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

        service = OllamaService()
        result = service.infer_gender("Test Name")

        # Should return "unknown" on error
        self.assertEqual(result, "unknown")

    @patch("services.ollama_service.requests.post")
    def test_ollama_service_handles_timeout_error(self, mock_post):
        """Test that OllamaService handles timeout errors gracefully."""
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

        service = OllamaService()
        result = service.detect_intent("Test message")

        # Should return "outro" on error
        self.assertEqual(result, "outro")

    @patch("services.ollama_service.requests.post")
    def test_ollama_service_handles_http_error(self, mock_post):
        """Test that OllamaService handles HTTP errors gracefully."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Server error")
        mock_post.return_value = mock_response

        service = OllamaService()
        result = service.approximate_theme("teste")

        # Should return "outro" on error
        self.assertEqual(result, "outro")

    @patch("services.ollama_service.requests.post")
    def test_ollama_service_handles_empty_response(self, mock_post):
        """Test that OllamaService handles empty response content."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": ""}
        }
        mock_post.return_value = mock_response

        service = OllamaService()
        result = service.infer_gender("Test")

        # Should return "unknown" on empty response
        self.assertEqual(result, "unknown")

    @patch("services.ollama_service.requests.post")
    @patch("services.ollama_service.sanitize_input")
    def test_generate_intent_response_with_multiple_messages(self, mock_sanitize, mock_post):
        """Test that generate_intent_response splits messages by paragraph."""
        mock_sanitize.return_value = "Preciso de ajuda"
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "Entendo sua preocupação.\n\nComo posso ajudar?\n\nEstou aqui para ouvir."}
        }
        mock_post.return_value = mock_response

        service = OllamaService()
        messages = service.generate_intent_response(
            user_message="Preciso de ajuda",
            conversation_context=[],
            name="João",
            intent="ansiedade"
        )

        # Should split into 3 separate messages by paragraph breaks
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0], "Entendo sua preocupação.")
        self.assertEqual(messages[1], "Como posso ajudar?")
        self.assertEqual(messages[2], "Estou aqui para ouvir.")

    @patch("services.ollama_service.requests.post")
    @patch("services.ollama_service.sanitize_input")
    def test_generate_intent_response_with_context(self, mock_sanitize, mock_post):
        """Test that generate_intent_response uses new 5-layer payload structure."""
        mock_sanitize.return_value = "Obrigado"
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "De nada! Estou aqui para você."}
        }
        mock_post.return_value = mock_response

        service = OllamaService()
        context = [
            {"role": "user", "content": "Estou triste"},
            {"role": "assistant", "content": "Entendo. O que aconteceu?"}
        ]
        
        messages = service.generate_intent_response(
            user_message="Obrigado",
            conversation_context=context,
            name="Maria",
            intent="desabafar"
        )

        # Verify new 5-layer structure: system, memory_user, memory_assistant, (optional rag), user
        payload = mock_post.call_args[1]["json"]
        
        # Should have 4 messages minimum: system + memory_user + memory_assistant + user
        # (rag is optional and not included if no relevant context)
        self.assertGreaterEqual(len(payload["messages"]), 4)
        
        # Layer 1: system
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertIn("companheiro virtual", payload["messages"][0]["content"].lower())
        
        # Layer 2: memory_user
        self.assertEqual(payload["messages"][1]["role"], "system")
        self.assertIn("[MEMORY_USER]", payload["messages"][1]["content"])
        self.assertIn("Maria", payload["messages"][1]["content"])
        
        # Layer 3: memory_assistant
        self.assertEqual(payload["messages"][2]["role"], "system")
        self.assertIn("[MEMORY_ASSISTANT]", payload["messages"][2]["content"])
        self.assertIn("Estou triste", payload["messages"][2]["content"])
        self.assertIn("Entendo. O que aconteceu?", payload["messages"][2]["content"])
        
        # Last layer: user
        self.assertEqual(payload["messages"][-1]["role"], "user")
        self.assertEqual(payload["messages"][-1]["content"], "Obrigado")

    @patch("services.ollama_service.requests.post")
    def test_make_chat_request_with_custom_kwargs(self, mock_post):
        """Test that _make_chat_request accepts and passes through custom kwargs."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "Response"}
        }
        mock_post.return_value = mock_response

        service = OllamaService()
        messages = [{"role": "user", "content": "Test"}]
        
        # Call with custom kwargs
        service._make_chat_request(
            messages,
            temperature=0.5,
            max_tokens=100,
            top_p=0.9,
            repeat_penalty=1.1
        )

        # Verify custom options were passed
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["options"]["temperature"], 0.5)
        self.assertEqual(payload["options"]["num_predict"], 100)
        self.assertEqual(payload["options"]["top_p"], 0.9)
        self.assertEqual(payload["options"]["repeat_penalty"], 1.1)

    @patch("services.ollama_service.requests.post")
    @patch("services.ollama_service.sanitize_input")
    def test_approximate_theme_validates_response(self, mock_sanitize, mock_post):
        """Test that approximate_theme validates and defaults invalid responses."""
        mock_sanitize.return_value = "teste"
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "tema_invalido"}
        }
        mock_post.return_value = mock_response

        service = OllamaService()
        result = service.approximate_theme("teste")

        # Should default to "outro" for invalid theme
        self.assertEqual(result, "outro")

    @patch("services.ollama_service.requests.post")
    @patch("services.ollama_service.sanitize_input")
    @patch("services.ollama_service.get_rag_context")
    def test_five_layer_payload_structure(self, mock_rag, mock_sanitize, mock_post):
        """Test that generate_intent_response uses strict 5-layer payload structure."""
        mock_sanitize.return_value = "Preciso de ajuda"
        mock_rag.return_value = ["RAG context about support"]
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "Estou aqui para você."}
        }
        mock_post.return_value = mock_response

        service = OllamaService()
        context = [
            {"role": "user", "content": "Estou perdido"},
            {"role": "assistant", "content": "O que você está sentindo?"}
        ]
        
        service.generate_intent_response(
            user_message="Preciso de ajuda",
            conversation_context=context,
            name="João",
            inferred_gender="male",
            intent="desabafar"
        )

        # Verify strict 5-layer structure
        payload = mock_post.call_args[1]["json"]
        messages = payload["messages"]
        
        # Should have exactly 5 layers: system, memory_user, memory_assistant, rag, user
        self.assertEqual(len(messages), 5)
        
        # Layer 1: SYSTEM - Fixed identity and global rules
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("companheiro virtual", messages[0]["content"].lower())
        self.assertNotIn("[MEMORY_USER]", messages[0]["content"])
        self.assertNotIn("[MEMORY_ASSISTANT]", messages[0]["content"])
        self.assertNotIn("[RAG]", messages[0]["content"])
        
        # Layer 2: MEMORY_USER - Persistent user facts only
        self.assertEqual(messages[1]["role"], "system")
        self.assertIn("[MEMORY_USER]", messages[1]["content"])
        self.assertIn("João", messages[1]["content"])
        self.assertIn("male", messages[1]["content"])
        self.assertNotIn("Estou perdido", messages[1]["content"])  # No conversation history
        
        # Layer 3: MEMORY_ASSISTANT - Internal conversation state
        self.assertEqual(messages[2]["role"], "system")
        self.assertIn("[MEMORY_ASSISTANT]", messages[2]["content"])
        self.assertIn("Estou perdido", messages[2]["content"])  # Has conversation history
        self.assertIn("O que você está sentindo?", messages[2]["content"])
        self.assertIn("Próximo objetivo", messages[2]["content"])  # Has next objective
        
        # Layer 4: RAG - External knowledge only
        self.assertEqual(messages[3]["role"], "system")
        self.assertIn("[RAG]", messages[3]["content"])
        self.assertIn("RAG context about support", messages[3]["content"])
        
        # Layer 5: USER - Current message only
        self.assertEqual(messages[4]["role"], "user")
        self.assertEqual(messages[4]["content"], "Preciso de ajuda")
        self.assertNotIn("[", messages[4]["content"])  # No markers in user message

    @patch("services.ollama_service.requests.post")
    @patch("services.ollama_service.sanitize_input")
    @patch("services.ollama_service.get_rag_context")
    def test_five_layer_structure_without_rag(self, mock_rag, mock_sanitize, mock_post):
        """Test 5-layer structure when RAG layer is empty."""
        mock_sanitize.return_value = "Olá"
        mock_rag.return_value = []  # No RAG context
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"role": "assistant", "content": "Olá! Como posso ajudar?"}
        }
        mock_post.return_value = mock_response

        service = OllamaService()
        
        service.generate_intent_response(
            user_message="Olá",
            conversation_context=None,
            name="Maria",
            intent="desabafar"
        )

        # Verify structure without RAG
        payload = mock_post.call_args[1]["json"]
        messages = payload["messages"]
        
        # Should have 4 layers when RAG is empty: system, memory_user, memory_assistant, user
        self.assertEqual(len(messages), 4)
        
        # Verify RAG layer is not present
        for msg in messages:
            self.assertNotIn("[RAG]", msg.get("content", ""))
        
        # Last message should still be user
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(messages[-1]["content"], "Olá")
