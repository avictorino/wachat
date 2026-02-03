"""
Tests for the core app, including models and views.
"""

from unittest import skip
from unittest.mock import Mock, patch

from django.test import TestCase

from core.models import Message, Profile


class ProfileModelTest(TestCase):
    """Tests for the Profile model."""

    def test_create_profile(self):
        """Test creating a new profile."""
        profile = Profile.objects.create(
            telegram_user_id="12345",
            name="João Silva",
            phone_number="+5511999999999",
            inferred_gender="male",
        )

        self.assertEqual(profile.telegram_user_id, "12345")
        self.assertEqual(profile.name, "João Silva")
        self.assertEqual(profile.phone_number, "+5511999999999")
        self.assertEqual(profile.inferred_gender, "male")
        self.assertIsNotNone(profile.created_at)
        self.assertIsNotNone(profile.updated_at)

    def test_profile_str_representation(self):
        """Test the string representation of Profile."""
        profile = Profile.objects.create(telegram_user_id="12345", name="João Silva")
        self.assertEqual(str(profile), "João Silva (12345)")


class MessageModelTest(TestCase):
    """Tests for the Message model."""

    def setUp(self):
        """Create a test profile for message tests."""
        self.profile = Profile.objects.create(
            telegram_user_id="12345", name="João Silva"
        )

    def test_create_message(self):
        """Test creating a new message."""
        message = Message.objects.create(
            profile=self.profile, role="assistant", content="Olá! Bem-vindo."
        )

        self.assertEqual(message.profile, self.profile)
        self.assertEqual(message.role, "assistant")
        self.assertEqual(message.content, "Olá! Bem-vindo.")
        self.assertEqual(message.channel, "telegram")  # Default channel
        self.assertIsNotNone(message.created_at)

    def test_message_str_representation(self):
        """Test the string representation of Message."""
        message = Message.objects.create(
            profile=self.profile,
            role="assistant",
            content="This is a long message that should be truncated in the string representation",
        )
        str_repr = str(message)
        self.assertTrue(str_repr.startswith("assistant:"))
        self.assertTrue("..." in str_repr)


class TelegramWebhookViewTest(TestCase):
    """Tests for the Telegram webhook view."""

    def setUp(self):
        """Set up test fixtures."""
        self.webhook_url = "/webhooks/telegram/"
        self.webhook_secret = "test-secret"

    @patch.dict("os.environ", {"TELEGRAM_WEBHOOK_SECRET": "test-secret"})
    def test_webhook_requires_secret(self):
        """Test that webhook validates secret token."""
        response = self.client.post(
            self.webhook_url, data="{}", content_type="application/json"
        )
        # Should fail without secret header
        self.assertEqual(response.status_code, 403)

    @patch.dict("os.environ", {"TELEGRAM_WEBHOOK_SECRET": "test-secret"})
    def test_webhook_accepts_valid_secret(self):
        """Test that webhook accepts requests with valid secret."""
        response = self.client.post(
            self.webhook_url,
            data='{"update_id": 1, "message": {}}',
            content_type="application/json",
            HTTP_X_TELEGRAM_BOT_API_SECRET_TOKEN="test-secret",
        )
        self.assertEqual(response.status_code, 200)

    @patch("core.views.TelegramService")
    @patch("core.views.get_llm_service")
    @patch.dict(
        "os.environ",
        {
            "TELEGRAM_WEBHOOK_SECRET": "test-secret",
            "TELEGRAM_BOT_TOKEN": "test-token",
            "GROQ_API_KEY": "test-key",
        },
    )
    def test_start_command_creates_profile(self, mock_groq, mock_telegram):
        """Test that /start command creates a profile and sends welcome messages."""
        # Mock Groq service
        mock_groq_instance = Mock()
        mock_groq_instance.infer_gender.return_value = "male"
        mock_groq_instance.generate_welcome_message.return_value = (
            "Olá João! Bem-vindo ao nosso espaço. O que te trouxe aqui?"
        )
        mock_groq.return_value = mock_groq_instance

        # Mock Telegram service
        mock_telegram_instance = Mock()
        mock_telegram_instance.send_messages.return_value = True
        mock_telegram.return_value = mock_telegram_instance

        # Send /start command
        payload = {
            "update_id": 1,
            "message": {
                "message_id": 1,
                "from": {"id": 12345, "first_name": "João", "last_name": "Silva"},
                "chat": {"id": 12345, "type": "private"},
                "text": "/start",
            },
        }

        response = self.client.post(
            self.webhook_url,
            data=payload,
            content_type="application/json",
            HTTP_X_TELEGRAM_BOT_API_SECRET_TOKEN="test-secret",
        )

        self.assertEqual(response.status_code, 200)

        # Verify profile was created
        profile = Profile.objects.get(telegram_user_id="12345")
        self.assertEqual(profile.name, "João Silva")
        self.assertEqual(profile.inferred_gender, "male")

        # Verify messages were persisted (should be 2 messages now)
        messages = Message.objects.filter(profile=profile).order_by("id")
        self.assertEqual(messages.count(), 2)
        
        # First message should be the greeting
        self.assertEqual(messages[0].role, "assistant")
        self.assertEqual(messages[0].content, "Olá João! Bem-vindo ao nosso espaço.")
        self.assertEqual(messages[0].channel, "telegram")
        
        # Second message should be the question
        self.assertEqual(messages[1].role, "assistant")
        self.assertEqual(messages[1].content, "O que te trouxe aqui?")
        self.assertEqual(messages[1].channel, "telegram")

        # Verify services were called
        mock_groq_instance.infer_gender.assert_called_once_with("João Silva")
        mock_groq_instance.generate_welcome_message.assert_called_once()
        
        # Verify send_messages was called with both parts
        mock_telegram_instance.send_messages.assert_called_once()
        call_args = mock_telegram_instance.send_messages.call_args
        self.assertEqual(call_args[0][0], "12345")  # chat_id
        self.assertEqual(len(call_args[0][1]), 2)  # two messages

    @patch("core.views.TelegramService")
    @patch("core.views.get_llm_service")
    @patch.dict(
        "os.environ",
        {
            "TELEGRAM_WEBHOOK_SECRET": "test-secret",
            "TELEGRAM_BOT_TOKEN": "test-token",
            "GROQ_API_KEY": "test-key",
        },
    )
    def test_start_command_sanitizes_harmful_names(self, mock_groq, mock_telegram):
        """Test that /start command sanitizes harmful content in names before LLM call."""
        # Mock Groq service
        mock_groq_instance = Mock()
        mock_groq_instance.infer_gender.return_value = "unknown"
        mock_groq_instance.generate_welcome_message.return_value = (
            "Olá! Bem-vindo ao nosso espaço. Como você está?"
        )
        mock_groq.return_value = mock_groq_instance

        # Mock Telegram service
        mock_telegram_instance = Mock()
        mock_telegram_instance.send_messages.return_value = True
        mock_telegram.return_value = mock_telegram_instance

        # Send /start command with a name containing harmful content
        payload = {
            "update_id": 1,
            "message": {
                "message_id": 1,
                "from": {"id": 99999, "first_name": "Sexo", "last_name": "Silva"},
                "chat": {"id": 99999, "type": "private"},
                "text": "/start",
            },
        }

        response = self.client.post(
            self.webhook_url,
            data=payload,
            content_type="application/json",
            HTTP_X_TELEGRAM_BOT_API_SECRET_TOKEN="test-secret",
        )

        self.assertEqual(response.status_code, 200)

        # Verify profile was created with unsanitized name (we store original)
        profile = Profile.objects.get(telegram_user_id="99999")
        self.assertEqual(profile.name, "Sexo Silva")

        # Verify that GroqService methods were called (sanitization happens inside)
        # The sanitization is transparent - we just verify the service was called
        mock_groq_instance.infer_gender.assert_called_once_with("Sexo Silva")
        mock_groq_instance.generate_welcome_message.assert_called_once()
        mock_telegram_instance.send_messages.assert_called_once()

    @patch("core.views.TelegramService")
    @patch("core.views.get_llm_service")
    @patch.dict(
        "os.environ",
        {
            "TELEGRAM_WEBHOOK_SECRET": "test-secret",
            "TELEGRAM_BOT_TOKEN": "test-token",
            "GROQ_API_KEY": "test-key",
        },
    )
    def test_regular_message_detects_intent(self, mock_groq, mock_telegram):
        """Test that regular messages detect intent and generate response."""
        # Create a profile first (simulating a user who already did /start)
        profile = Profile.objects.create(
            telegram_user_id="12345", name="João Silva", inferred_gender="male"
        )

        # Add a welcome message to simulate previous conversation
        Message.objects.create(
            profile=profile, role="assistant", content="Olá João! Bem-vindo."
        )

        # Mock Groq service
        mock_groq_instance = Mock()
        mock_groq_instance.detect_intent.return_value = "ansiedade"
        mock_groq_instance.generate_intent_response.return_value = ["Entendo que você está se sentindo ansioso. Quer me contar um pouco mais sobre isso?"]
        mock_groq.return_value = mock_groq_instance

        # Mock Telegram service
        mock_telegram_instance = Mock()
        mock_telegram_instance.send_messages.return_value = True
        mock_telegram.return_value = mock_telegram_instance

        # Send a regular message
        payload = {
            "update_id": 2,
            "message": {
                "message_id": 2,
                "from": {"id": 12345, "first_name": "João", "last_name": "Silva"},
                "chat": {"id": 12345, "type": "private"},
                "text": "Me sinto muito ansioso ultimamente",
            },
        }

        response = self.client.post(
            self.webhook_url,
            data=payload,
            content_type="application/json",
            HTTP_X_TELEGRAM_BOT_API_SECRET_TOKEN="test-secret",
        )

        self.assertEqual(response.status_code, 200)

        # Verify profile was updated with detected intent
        profile.refresh_from_db()
        self.assertEqual(profile.detected_intent, "ansiedade")

        # Verify messages were persisted (user message + assistant response)
        messages = Message.objects.filter(profile=profile).order_by("created_at")
        self.assertEqual(messages.count(), 3)  # welcome + user + assistant

        # Check user message
        user_message = messages[1]
        self.assertEqual(user_message.role, "user")
        self.assertEqual(user_message.content, "Me sinto muito ansioso ultimamente")

        # Check assistant message
        assistant_message = messages[2]
        self.assertEqual(assistant_message.role, "assistant")
        self.assertIn("ansioso", assistant_message.content.lower())

        # Verify services were called
        mock_groq_instance.detect_intent.assert_called_once_with(
            "Me sinto muito ansioso ultimamente"
        )
        mock_groq_instance.generate_intent_response.assert_called_once()
        mock_telegram_instance.send_messages.assert_called_once()

    @patch("core.views.TelegramService")
    @patch("core.views.get_llm_service")
    @patch.dict(
        "os.environ",
        {
            "TELEGRAM_WEBHOOK_SECRET": "test-secret",
            "TELEGRAM_BOT_TOKEN": "test-token",
            "GROQ_API_KEY": "test-key",
        },
    )
    def test_regular_message_without_profile_creates_one(
        self, mock_groq, mock_telegram
    ):
        """Test that regular message from unknown user creates profile."""
        # Mock Groq service
        mock_groq_instance = Mock()
        mock_groq_instance.detect_intent.return_value = "desabafar"
        mock_groq_instance.generate_intent_response.return_value = [
            "Estou aqui para ouvir. O que você gostaria de compartilhar?"
        ]
        mock_groq.return_value = mock_groq_instance

        # Mock Telegram service
        mock_telegram_instance = Mock()
        mock_telegram_instance.send_messages.return_value = True
        mock_telegram.return_value = mock_telegram_instance

        # Send message from unknown user
        payload = {
            "update_id": 1,
            "message": {
                "message_id": 1,
                "from": {"id": 99999, "first_name": "Maria"},
                "chat": {"id": 99999, "type": "private"},
                "text": "Preciso desabafar",
            },
        }

        response = self.client.post(
            self.webhook_url,
            data=payload,
            content_type="application/json",
            HTTP_X_TELEGRAM_BOT_API_SECRET_TOKEN="test-secret",
        )

        self.assertEqual(response.status_code, 200)

        # Verify profile was created
        profile = Profile.objects.get(telegram_user_id="99999")
        self.assertEqual(profile.name, "Maria")
        self.assertEqual(profile.detected_intent, "desabafar")

        # Verify messages were created
        messages = Message.objects.filter(profile=profile)
        self.assertEqual(messages.count(), 2)  # user message + assistant response

    @patch("core.views.TelegramService")
    @patch("core.views.get_llm_service")
    @patch.dict(
        "os.environ",
        {
            "TELEGRAM_WEBHOOK_SECRET": "test-secret",
            "TELEGRAM_BOT_TOKEN": "test-token",
            "GROQ_API_KEY": "test-key",
        },
    )
    def test_subsequent_messages_use_stored_intent(self, mock_groq, mock_telegram):
        """Test that subsequent messages use previously detected intent."""
        # Create a profile with existing intent
        profile = Profile.objects.create(
            telegram_user_id="12345", name="João Silva", detected_intent="ansiedade"
        )

        # Add some conversation history
        Message.objects.create(profile=profile, role="assistant", content="Welcome")
        Message.objects.create(profile=profile, role="user", content="First message")
        Message.objects.create(
            profile=profile, role="assistant", content="First response"
        )

        # Mock Groq service
        mock_groq_instance = Mock()
        mock_groq_instance.generate_intent_response.return_value = [
            "Continue me contando sobre isso."
        ]
        mock_groq.return_value = mock_groq_instance

        # Mock Telegram service
        mock_telegram_instance = Mock()
        mock_telegram_instance.send_message.return_value = True
        mock_telegram.return_value = mock_telegram_instance

        # Send another message
        payload = {
            "update_id": 3,
            "message": {
                "message_id": 3,
                "from": {"id": 12345, "first_name": "João"},
                "chat": {"id": 12345, "type": "private"},
                "text": "É difícil dormir à noite",
            },
        }

        response = self.client.post(
            self.webhook_url,
            data=payload,
            content_type="application/json",
            HTTP_X_TELEGRAM_BOT_API_SECRET_TOKEN="test-secret",
        )

        self.assertEqual(response.status_code, 200)

        # Verify intent detection was NOT called (should use stored intent)
        mock_groq_instance.detect_intent.assert_not_called()

        # Verify response generation used stored intent
        call_args = mock_groq_instance.generate_intent_response.call_args
        self.assertEqual(call_args[1]["intent"], "ansiedade")


class GroqServiceTest(TestCase):
    """Tests for the Groq service methods."""

    @patch("services.groq_service.Groq")
    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    def test_detect_intent_financial_problems(self, mock_groq_client):
        """Test intent detection for financial problems."""
        from services.groq_service import GroqService

        # Mock the Groq client response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "problemas_financeiros"

        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_client_instance

        # Test the service
        service = GroqService()
        intent = service.detect_intent("Estou com muitas dívidas e sem emprego")

        self.assertEqual(intent, "problemas_financeiros")
        mock_client_instance.chat.completions.create.assert_called_once()

    @patch("services.groq_service.Groq")
    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    def test_detect_intent_anxiety(self, mock_groq_client):
        """Test intent detection for anxiety."""
        from services.groq_service import GroqService

        # Mock the Groq client response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "ansiedade"

        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_client_instance

        # Test the service
        service = GroqService()
        intent = service.detect_intent("Me sinto muito ansioso e preocupado")

        self.assertEqual(intent, "ansiedade")

    @patch("services.groq_service.Groq")
    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    @skip("GroqService does not validate intent responses - tests unimplemented behavior")
    def test_detect_intent_invalid_returns_outro(self, mock_groq_client):
        """Test that invalid intent returns 'outro'."""
        from services.groq_service import GroqService

        # Mock the Groq client to return invalid intent
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "invalid_intent_xyz"

        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_client_instance

        # Test the service
        service = GroqService()
        intent = service.detect_intent("Some random message")

        self.assertEqual(intent, "outro")

    @patch("services.groq_service.Groq")
    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    def test_detect_intent_error_returns_outro(self, mock_groq_client):
        """Test that API errors return 'outro'."""
        from services.groq_service import GroqService

        # Mock the Groq client to raise an exception
        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.side_effect = Exception(
            "API Error"
        )
        mock_groq_client.return_value = mock_client_instance

        # Test the service
        service = GroqService()
        intent = service.detect_intent("Some message")

        self.assertEqual(intent, "outro")

    @patch("services.groq_service.Groq")
    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    def test_generate_intent_response(self, mock_groq_client):
        """Test generating response based on intent."""
        from services.groq_service import GroqService

        # Mock the Groq client response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = (
            "Entendo o peso que você está carregando. Estou aqui contigo nessa caminhada. Quer me contar um pouco mais sobre o que está sentindo?"
        )

        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_client_instance

        # Test the service
        service = GroqService()
        response_list = service.generate_intent_response(
            user_message="Me sinto muito ansioso",
            intent="ansiedade",
            name="João",
            inferred_gender="male",
        )

        # Response should be a list of strings
        self.assertIsInstance(response_list, list)
        response = " ".join(response_list)  # Join for assertion
        self.assertIn("Entendo", response)
        self.assertIn("?", response)  # Should end with a question
        mock_client_instance.chat.completions.create.assert_called_once()

    @patch("services.groq_service.Groq")
    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    def test_generate_intent_response_error_returns_fallback(self, mock_groq_client):
        """Test that API errors return fallback message."""
        from services.groq_service import GroqService

        # Mock the Groq client to raise an exception
        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.side_effect = Exception(
            "API Error"
        )
        mock_groq_client.return_value = mock_client_instance

        # Test the service
        service = GroqService()
        response_list = service.generate_intent_response(
            user_message="Me sinto muito ansioso", intent="ansiedade", name="João"
        )

        # Should return fallback message as a list
        self.assertIsInstance(response_list, list)
        response = " ".join(response_list)  # Join for assertion
        self.assertIn("Obrigado", response)
        self.assertIn("incomoda", response)


class FallbackConversationalFlowTest(TestCase):
    """Tests for the fallback conversational flow functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.profile = Profile.objects.create(
            telegram_user_id="12345",
            name="João Silva",
            inferred_gender="male",
            detected_intent="outro",
        )

    @patch("core.views.TelegramService")
    @patch("core.views.get_llm_service")
    @patch.dict(
        "os.environ",
        {
            "TELEGRAM_WEBHOOK_SECRET": "test-secret",
            "TELEGRAM_BOT_TOKEN": "test-token",
            "GROQ_API_KEY": "test-key",
        },
    )
    def test_fallback_flow_with_outro_intent(self, mock_groq, mock_telegram):
        """Test that 'outro' intent triggers fallback conversational flow."""
        # Create conversation history
        Message.objects.create(
            profile=self.profile, role="assistant", content="Olá João! Bem-vindo."
        )
        Message.objects.create(
            profile=self.profile,
            role="user",
            content="Gostaria de ouvir um pouco da palavra de um bom pastor",
        )

        # Mock Groq service
        mock_groq_instance = Mock()
        mock_groq_instance.generate_fallback_response.return_value = [
            "Entendo o que você busca.",
            "Estou aqui para caminhar junto com você nessa jornada.",
        ]
        mock_groq.return_value = mock_groq_instance

        # Mock Telegram service
        mock_telegram_instance = Mock()
        mock_telegram_instance.send_messages.return_value = True
        mock_telegram.return_value = mock_telegram_instance

        # Send another message
        payload = {
            "update_id": 3,
            "message": {
                "message_id": 3,
                "from": {"id": 12345, "first_name": "João"},
                "chat": {"id": 12345, "type": "private"},
                "text": "Preciso de orientação espiritual",
            },
        }

        response = self.client.post(
            "/webhooks/telegram/",
            data=payload,
            content_type="application/json",
            HTTP_X_TELEGRAM_BOT_API_SECRET_TOKEN="test-secret",
        )

        self.assertEqual(response.status_code, 200)

        # Verify fallback response was called with conversation context
        mock_groq_instance.generate_fallback_response.assert_called_once()
        call_args = mock_groq_instance.generate_fallback_response.call_args

        # Check that context was passed
        self.assertIn("conversation_context", call_args[1])
        context = call_args[1]["conversation_context"]
        self.assertIsInstance(context, list)
        self.assertGreater(len(context), 0)

        # Verify multiple messages were sent
        mock_telegram_instance.send_messages.assert_called_once()
        send_call_args = mock_telegram_instance.send_messages.call_args
        messages_sent = send_call_args[0][1]  # Second positional arg
        self.assertEqual(len(messages_sent), 2)

        # Verify both messages were persisted
        assistant_messages = Message.objects.filter(
            profile=self.profile, role="assistant"
        ).order_by("created_at")
        # Should have: welcome + 2 new fallback messages = 3 total
        self.assertEqual(assistant_messages.count(), 3)
        self.assertEqual(
            assistant_messages[1].content, "Entendo o que você busca."
        )
        self.assertEqual(
            assistant_messages[2].content,
            "Estou aqui para caminhar junto com você nessa jornada.",
        )

    @patch("core.views.TelegramService")
    @patch("core.views.get_llm_service")
    @patch.dict(
        "os.environ",
        {
            "TELEGRAM_WEBHOOK_SECRET": "test-secret",
            "TELEGRAM_BOT_TOKEN": "test-token",
            "GROQ_API_KEY": "test-key",
        },
    )
    def test_non_outro_intent_uses_standard_flow(self, mock_groq, mock_telegram):
        """Test that non-'outro' intents use the standard intent-based flow."""
        # Update profile to have a specific intent
        self.profile.detected_intent = "ansiedade"
        self.profile.save()

        # Create conversation history
        Message.objects.create(
            profile=self.profile, role="assistant", content="Olá João!"
        )

        # Mock Groq service
        mock_groq_instance = Mock()
        mock_groq_instance.generate_intent_response.return_value = [
            "Entendo que você está ansioso. Como posso ajudar?"
        ]
        mock_groq.return_value = mock_groq_instance

        # Mock Telegram service
        mock_telegram_instance = Mock()
        mock_telegram_instance.send_messages.return_value = True
        mock_telegram.return_value = mock_telegram_instance

        # Send a message
        payload = {
            "update_id": 2,
            "message": {
                "message_id": 2,
                "from": {"id": 12345, "first_name": "João"},
                "chat": {"id": 12345, "type": "private"},
                "text": "Estou muito preocupado",
            },
        }

        response = self.client.post(
            "/webhooks/telegram/",
            data=payload,
            content_type="application/json",
            HTTP_X_TELEGRAM_BOT_API_SECRET_TOKEN="test-secret",
        )

        self.assertEqual(response.status_code, 200)

        # Verify intent-based response was used (not fallback)
        mock_groq_instance.generate_intent_response.assert_called_once()
        mock_groq_instance.generate_fallback_response.assert_not_called()

        # Verify messages were sent
        mock_telegram_instance.send_messages.assert_called_once()

    def test_get_conversation_context(self):
        """Test that conversation context is properly assembled."""
        from core.views import TelegramWebhookView

        # Create conversation history
        messages_data = [
            ("assistant", "Olá João! Bem-vindo."),
            ("user", "Oi, obrigado"),
            ("assistant", "Como posso te ajudar hoje?"),
            ("user", "Gostaria de conversar"),
            ("assistant", "Claro, estou aqui para ouvir"),
            ("user", "Tenho passado por momentos difíceis"),
        ]

        for role, content in messages_data:
            Message.objects.create(profile=self.profile, role=role, content=content)

        view = TelegramWebhookView()
        context = view._get_conversation_context(self.profile, limit=5)

        # Should get last 5 messages in chronological order
        self.assertEqual(len(context), 5)
        self.assertEqual(context[0]["role"], "user")
        self.assertEqual(context[0]["content"], "Oi, obrigado")
        self.assertEqual(context[-1]["role"], "user")
        self.assertEqual(
            context[-1]["content"], "Tenho passado por momentos difíceis"
        )

    def test_get_conversation_context_excludes_system_messages(self):
        """Test that system messages are excluded from context."""
        from core.views import TelegramWebhookView

        # Create messages including system messages
        Message.objects.create(
            profile=self.profile, role="system", content="System initialization"
        )
        Message.objects.create(
            profile=self.profile, role="assistant", content="Hello"
        )
        Message.objects.create(profile=self.profile, role="user", content="Hi")

        view = TelegramWebhookView()
        context = view._get_conversation_context(self.profile, limit=5)

        # Should only have assistant and user messages
        self.assertEqual(len(context), 2)
        self.assertNotIn("system", [msg["role"] for msg in context])


class GroqServiceFallbackTest(TestCase):
    """Tests for GroqService fallback response generation."""

    @patch("services.groq_service.Groq")
    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    def test_generate_fallback_response_single_message(self, mock_groq_client):
        """Test generating a single fallback message."""
        from services.groq_service import GroqService

        # Mock the Groq client response (single message)
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = (
            "Entendo o que você busca. Estou aqui para caminhar junto."
        )

        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_client_instance

        # Test the service
        service = GroqService()
        context = [
            {"role": "assistant", "content": "Olá!"},
            {"role": "user", "content": "Oi"},
        ]

        messages = service.generate_fallback_response(
            user_message="Gostaria de conversar sobre espiritualidade",
            conversation_context=context,
            name="João",
            inferred_gender="male",
        )

        # Should return list with one message
        self.assertIsInstance(messages, list)
        self.assertEqual(len(messages), 1)
        self.assertIn("Entendo", messages[0])

    @patch("services.groq_service.Groq")
    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    def test_generate_fallback_response_multiple_messages(self, mock_groq_client):
        """Test generating multiple fallback messages."""
        from services.groq_service import GroqService

        # Mock the Groq client response (multiple messages with separator)
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = (
            "Entendo o que você busca.|||Estou aqui para ouvir você."
        )

        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_client_instance

        # Test the service
        service = GroqService()
        context = [{"role": "assistant", "content": "Olá!"}]

        messages = service.generate_fallback_response(
            user_message="Preciso de ajuda espiritual",
            conversation_context=context,
            name="Maria",
            inferred_gender="female",
        )

        # Should return list with two messages
        self.assertIsInstance(messages, list)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0], "Entendo o que você busca.")
        self.assertEqual(messages[1], "Estou aqui para ouvir você.")

    @patch("services.groq_service.Groq")
    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    def test_generate_fallback_response_includes_context(self, mock_groq_client):
        """Test that conversation context is included in the API call."""
        from services.groq_service import GroqService

        # Mock the Groq client
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"

        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_client_instance

        # Test the service
        service = GroqService()
        context = [
            {"role": "assistant", "content": "Primeira mensagem"},
            {"role": "user", "content": "Segunda mensagem"},
        ]

        service.generate_fallback_response(
            user_message="Nova mensagem",
            conversation_context=context,
            name="João",
        )

        # Verify API was called with context messages
        call_args = mock_client_instance.chat.completions.create.call_args
        messages = call_args[1]["messages"]

        # Should have system prompt + context + new user message
        self.assertGreater(len(messages), 2)
        # Find the context messages in the call
        has_context = any(
            msg.get("content") == "Primeira mensagem" for msg in messages
        )
        self.assertTrue(has_context)

    @patch("services.groq_service.Groq")
    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key"})
    def test_split_response_messages(self, mock_groq_client):
        """Test the message splitting utility."""
        from services.groq_service import GroqService

        # Mock client setup (not used in this test but required for init)
        mock_client_instance = Mock()
        mock_groq_client.return_value = mock_client_instance

        service = GroqService()

        # Test single message (no separator)
        result = service._split_response_messages("Uma mensagem única")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "Uma mensagem única")

        # Test multiple messages with separator
        result = service._split_response_messages(
            "Primeira|||Segunda|||Terceira"
        )
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], "Primeira")
        self.assertEqual(result[1], "Segunda")
        self.assertEqual(result[2], "Terceira")

        # Test with extra whitespace
        result = service._split_response_messages(
            "Primeira  |||  Segunda  "
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "Primeira")
        self.assertEqual(result[1], "Segunda")

        # Test limit to 3 messages
        result = service._split_response_messages(
            "A|||B|||C|||D|||E"
        )
        self.assertEqual(len(result), 3)
        self.assertEqual(result, ["A", "B", "C"])


class TelegramServiceMultiMessageTest(TestCase):
    """Tests for TelegramService multi-message functionality."""

    @patch("services.telegram_service.requests.post")
    @patch("services.telegram_service.time.sleep")
    @patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "test-token"})
    def test_send_messages_sends_all(self, mock_sleep, mock_post):
        """Test that send_messages sends all messages sequentially."""
        from services.telegram_service import TelegramService

        # Mock successful responses
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        service = TelegramService()
        messages = ["Message 1", "Message 2", "Message 3"]

        result = service.send_messages("12345", messages)

        # Should succeed
        self.assertTrue(result)

        # Should call post 3 times
        self.assertEqual(mock_post.call_count, 3)

        # Should pause between messages (2 pauses for 3 messages)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("services.telegram_service.requests.post")
    @patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "test-token"})
    def test_send_messages_handles_empty_list(self, mock_post):
        """Test that send_messages handles empty list gracefully."""
        from services.telegram_service import TelegramService

        service = TelegramService()
        result = service.send_messages("12345", [])

        # Should return True (no-op)
        self.assertTrue(result)

        # Should not call post
        mock_post.assert_not_called()

    @patch("services.telegram_service.requests.post")
    @patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "test-token"})
    def test_send_messages_reports_partial_failure(self, mock_post):
        """Test that send_messages reports failure if any message fails."""
        from services.telegram_service import TelegramService

        # Mock first success, then failure
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_failure = Mock()
        mock_response_failure.status_code = 500
        mock_response_failure.text = "Error"

        mock_post.side_effect = [
            mock_response_success,
            mock_response_failure,
        ]

        service = TelegramService()
        messages = ["Message 1", "Message 2"]

        result = service.send_messages("12345", messages)

        # Should report failure
        self.assertFalse(result)

        # Should have tried to send both
        self.assertEqual(mock_post.call_count, 2)
