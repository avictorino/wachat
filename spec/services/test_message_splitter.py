"""
Tests for message_splitter module.

Tests the splitting logic for welcome messages to ensure they are
properly divided into greeting and question parts.
"""

from django.test import TestCase

from services.message_splitter import split_welcome_message


class MessageSplitterTest(TestCase):
    """Tests for split_welcome_message function."""

    def test_split_simple_message_with_question(self):
        """Test splitting a simple message with one question at the end."""
        message = "Olá João! Bem-vindo ao nosso espaço. O que te trouxe aqui hoje?"
        greeting, question = split_welcome_message(message)

        self.assertEqual(greeting, "Olá João! Bem-vindo ao nosso espaço.")
        self.assertEqual(question, "O que te trouxe aqui hoje?")

    def test_split_message_with_multiple_sentences(self):
        """Test splitting a message with multiple sentences before the question."""
        message = "Bem-vindo. Este é um espaço seguro. Estou aqui para ouvir. O que anda pesando no seu coração?"
        greeting, question = split_welcome_message(message)

        self.assertEqual(greeting, "Bem-vindo. Este é um espaço seguro. Estou aqui para ouvir.")
        self.assertEqual(question, "O que anda pesando no seu coração?")

    def test_split_message_preserves_content(self):
        """Test that splitting preserves all content from original message."""
        message = "João, bem-vindo. Como você está se sentindo hoje?"
        greeting, question = split_welcome_message(message)

        # Verify all words are preserved (normalize punctuation)
        original_words = set(message.replace("?", "").replace(".", "").replace(",", "").split())
        split_words = set((greeting + " " + question).replace("?", "").replace(".", "").replace(",", "").split())

        self.assertEqual(original_words, split_words)

    def test_split_message_with_no_question_mark(self):
        """Test that messages without question marks are split by sentence."""
        message = "Bem-vindo ao espaço. Estou aqui para caminhar contigo. Sinta-se à vontade."
        greeting, question = split_welcome_message(message)

        # Should split somewhere, both parts non-empty
        self.assertTrue(len(greeting) > 0)
        self.assertTrue(len(question) > 0)

    def test_split_message_greeting_no_question_mark(self):
        """Test that greeting part doesn't end with question mark."""
        message = "Olá Maria! Bem-vindo. Este é um lugar seguro. Como você está?"
        greeting, question = split_welcome_message(message)

        self.assertFalse(greeting.endswith("?"))

    def test_split_message_question_has_question_mark(self):
        """Test that question part contains question mark when present."""
        message = "Bem-vindo. O que te trouxe até aqui?"
        greeting, question = split_welcome_message(message)

        self.assertIn("?", question)

    def test_split_message_with_multiple_questions(self):
        """Test that when there are multiple questions, split at the last one."""
        message = "Você está bem? Este lugar é seguro. O que você precisa hoje?"
        greeting, question = split_welcome_message(message)

        # Should split at the LAST question
        self.assertEqual(question, "O que você precisa hoje?")
        self.assertIn("Você está bem?", greeting)

    def test_split_handles_whitespace(self):
        """Test that splitting handles extra whitespace properly."""
        message = "Bem-vindo João.   O que te trouxe aqui?  "
        greeting, question = split_welcome_message(message)

        # Should be trimmed
        self.assertEqual(greeting, "Bem-vindo João.")
        self.assertEqual(question, "O que te trouxe aqui?")

    def test_split_single_question_returns_as_greeting(self):
        """Test that a message that is entirely a question returns it as greeting."""
        message = "O que te trouxe aqui hoje?"
        greeting, question = split_welcome_message(message)

        # Can't split a single question meaningfully
        self.assertEqual(greeting, message)
        self.assertEqual(question, "")

    def test_split_real_world_example_1(self):
        """Test with a real-world style welcome message."""
        message = """Ademar, é bom ter você aqui.

Este é um espaço onde você pode ser quem você é, sem medo ou julgamento. Estou aqui para caminhar ao seu lado nessa jornada.

O que te trouxe até este lugar hoje?"""

        greeting, question = split_welcome_message(message)

        # Should split before the final question
        self.assertNotIn("O que te trouxe", greeting)
        self.assertIn("O que te trouxe", question)
        self.assertTrue(len(greeting) > 0)

    def test_split_real_world_example_2(self):
        """Test with another real-world style message."""
        message = "Maria, seja bem-vinda. Este é um espaço de escuta, reflexão espiritual e acolhimento. Não há julgamento aqui. Em que parte da sua caminhada você sente que precisa de companhia agora?"

        greeting, question = split_welcome_message(message)

        # Greeting should have welcoming content
        self.assertIn("bem-vinda", greeting)
        self.assertIn("acolhimento", greeting)

        # Question should be the reflective question
        self.assertIn("caminhada", question)
        self.assertTrue(question.endswith("?"))
