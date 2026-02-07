"""
Message splitter for creating natural, conversational message flow.

This module handles splitting long welcome messages into two sequential messages:
1. A warm, welcoming greeting
2. A shorter reflective question

This creates a more human, less robotic conversational experience.

It also provides smart response splitting that:
- Splits by logical paragraphs (double newline)
- Trims whitespace from each part
- Discards orphan words/fragments
- Ensures the final message is complete
- Merges incomplete segments with previous messages
"""

import logging
import re
from typing import List, Tuple

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


def _is_complete_sentence(text: str) -> bool:
    """
    Check if text appears to be a complete sentence or question.
    
    A complete sentence should:
    - Have proper ending punctuation (. ! ? :), OR
    - Have multiple words (5+) even without punctuation
    
    Incomplete fragments are:
    - Single words without proper punctuation (e.g., "Você", "está", "e")
    - Very short phrases without punctuation
    
    Args:
        text: The text to check
        
    Returns:
        True if it appears to be a complete sentence, False otherwise
    """
    text = text.strip()
    
    # Empty or too short
    if not text or len(text) < 2:
        return False
    
    # Count words (simple split by space)
    words = text.split()
    num_words = len(words)
    
    # Has sentence-ending punctuation
    has_punctuation = any(text.endswith(p) for p in ['.', '!', '?', ':'])
    
    # Single word cases
    if num_words == 1:
        # Single word WITH punctuation is considered complete
        # (e.g., "Entendo.", "Olá!", "Sim.", "Bem.")
        # These are valid short sentences in Portuguese
        return has_punctuation
    
    # Multiple words - check for proper ending
    # If it ends with sentence-ending punctuation, it's complete
    if has_punctuation:
        return True
    
    # Multiple words without punctuation - could be incomplete
    # but we'll be lenient if it's reasonably long (5+ words)
    return num_words >= 5


def split_response_messages(response: str) -> List[str]:
    """
    Split a bot response into multiple sequential messages based on logical paragraphs.
    
    This function:
    - Splits by double newlines (paragraph breaks)
    - Trims whitespace from each part
    - Discards orphan words or incomplete fragments
    - Ensures the final message is a complete sentence
    - Merges incomplete final segments with the previous message
    
    Args:
        response: The full bot response text
        
    Returns:
        List of message strings to send sequentially
        
    Examples:
        >>> split_response_messages("Olá João!\\n\\nComo você está?")
        ["Olá João!", "Como você está?"]
        
        >>> split_response_messages("Entendo.\\n\\nVocê")
        ["Entendo."]
        
        >>> split_response_messages("Oi.\\n\\nBem-vindo.\\n\\nVocê")
        ["Oi.", "Bem-vindo."]
    """
    if not response or not response.strip():
        return []
    
    # Split by double newline (paragraph breaks)
    parts = re.split(r'\n\n+', response)
    
    # Trim whitespace and filter empty parts
    parts = [part.strip() for part in parts if part.strip()]
    
    if not parts:
        return []
    
    # Process parts to ensure quality
    messages = []
    
    for i, part in enumerate(parts):
        is_last = (i == len(parts) - 1)
        
        # Check if this part is complete
        if _is_complete_sentence(part):
            # If this is the last part, add it normally
            if is_last:
                messages.append(part)
            else:
                # Not the last part, add it
                messages.append(part)
        else:
            # Incomplete fragment
            if is_last:
                # Last fragment is incomplete
                # Check if it's just a single orphan word (no punctuation)
                if len(part.split()) == 1 and not any(part.endswith(p) for p in ['.', '!', '?', ':']):
                    # Single orphan word - discard it
                    logger.info(f"Discarded orphan word at end: '{part}'")
                elif messages:
                    # Not a single orphan - merge with previous message if we have one
                    # Add a space between them for natural flow
                    messages[-1] = f"{messages[-1]} {part}"
                    logger.info(f"Merged incomplete last fragment '{part}' with previous message")
                else:
                    # Only fragment and it's incomplete - still include it
                    # (better than sending nothing)
                    messages.append(part)
                    logger.warning(f"Single incomplete fragment kept: '{part}'")
            else:
                # Not the last part but incomplete - discard it
                logger.info(f"Discarded incomplete fragment: '{part}'")
    
    # If we ended up with no messages, return the original response
    if not messages:
        logger.warning("No valid messages after splitting, returning original")
        return [response.strip()]
    
    logger.info(f"Split response into {len(messages)} messages")
    return messages
