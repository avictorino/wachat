"""
Message splitter for creating natural, conversational message flow.

This module handles splitting long welcome messages into two sequential messages:
1. A warm, welcoming greeting
2. A shorter reflective question

This creates a more human, less robotic conversational experience.
"""

import logging
import re
from typing import Tuple

logger = logging.getLogger(__name__)


def split_welcome_message(message: str) -> Tuple[str, str]:
    """
    Split a welcome message into greeting and question parts.

    The strategy is to find the last question (identified by '?' at the end of a sentence)
    and split the message there. The greeting part gets everything before the last question,
    and the question part gets the question itself.

    Args:
        message: The complete welcome message to split

    Returns:
        A tuple of (greeting_part, question_part)
        - greeting_part: The warm welcoming content without the final question
        - question_part: The reflective question that invites engagement

    Examples:
        >>> split_welcome_message("Olá João! Bem-vindo. O que te trouxe aqui?")
        ("Olá João! Bem-vindo.", "O que te trouxe aqui?")

        >>> split_welcome_message("Bem-vindo ao espaço. Estou aqui. Como posso ajudar?")
        ("Bem-vindo ao espaço. Estou aqui.", "Como posso ajudar?")
    """
    message = message.strip()

    # Find the last question mark
    last_question_pos = message.rfind("?")

    if last_question_pos == -1:
        # No question mark found, split at last sentence boundary or paragraph
        # Try to split at last period (with or without trailing space)
        sentences = re.split(r'(\.\s*)', message)
        if len(sentences) > 2:
            # Join all but the last sentence as greeting
            # The split includes the delimiters, so we need to handle them carefully
            mid_point = len(sentences) // 2
            greeting = ''.join(sentences[:mid_point]).strip()
            question = ''.join(sentences[mid_point:]).strip()
        else:
            # Can't split meaningfully, split in half by words
            words = message.split()
            mid = len(words) // 2
            greeting = ' '.join(words[:mid]).strip()
            question = ' '.join(words[mid:]).strip()

        logger.warning(f"No question mark found in message, split heuristically")
        return (greeting, question)

    # Find the start of the sentence containing the last question
    # Look backward for sentence boundaries (., !, or start of string)
    question_start = 0
    for i in range(last_question_pos - 1, -1, -1):
        if message[i] in '.!':
            # Found a sentence boundary, question starts after this
            question_start = i + 1
            break

    # Extract the question (from start to the question mark, inclusive)
    question = message[question_start:last_question_pos + 1].strip()

    # Extract the greeting (everything before the question)
    greeting = message[:question_start].strip()

    # Ensure both parts are non-empty
    if not greeting:
        # If greeting is empty, this means the entire message was a question
        # In this case, we can't split meaningfully, so return as-is
        logger.warning("Message is entirely a question, cannot split")
        return (message, "")

    if not question:
        # If question is empty (shouldn't happen given our logic, but be safe)
        logger.warning("Could not extract question, returning full message as greeting")
        return (message, "")

    logger.info(f"Split message into greeting ({len(greeting)} chars) and question ({len(question)} chars)")

    return (greeting, question)
