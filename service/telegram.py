"""
Telegram Bot integration and messaging functionality.

This module handles Telegram Bot API message processing, user management,
and integration with Telegram Bot API.
"""
import logging
import os
from typing import Optional

import requests

from messaging.types import OutgoingMessage

logger = logging.getLogger(__name__)


class TelegramProvider:
    """
    Telegram Bot provider using Telegram Bot API.
    
    Supports:
    - Text messages via sendMessage
    - Voice messages via sendVoice
    - Image messages via sendPhoto
    """

    def __init__(self, token: str):
        self.token = token
        self.api_base_url = f"https://api.telegram.org/bot{token}"

    def send(self, message: OutgoingMessage) -> None:
        """
        Send a message using Telegram Bot API.
        
        Args:
            message: The outgoing message to send
        """
        # If reply should be audio, send as voice message
        if message.reply_as_audio:
            self._send_voice_message(message)
        else:
            self._send_text_message(message)

    def _send_text_message(self, message: OutgoingMessage) -> None:
        """
        Send a text message using Telegram sendMessage API.
        
        Args:
            message: The outgoing message to send
        """
        url = f"{self.api_base_url}/sendMessage"
        
        payload = {
            "chat_id": message.to,
            "text": message.text,
            "parse_mode": "Markdown",  # Support basic formatting
        }

        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            logger.info(
                f"Successfully sent text message via Telegram to chat_id {message.to}"
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send message via Telegram: {e}")
            if hasattr(e, "response") and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            raise

    def _send_voice_message(self, message: OutgoingMessage) -> None:
        """
        Send an audio message using Telegram sendVoice API.
        
        This requires first converting text to speech and getting the audio URL.
        
        Args:
            message: The outgoing message to send
        """
        try:
            from service.media_generation import TextToSpeechService
            
            # Convert text to speech and get URL
            audio_url = TextToSpeechService().speak_and_store(
                text=message.text, conversation_mode=message.conversation_mode
            )
            
            url = f"{self.api_base_url}/sendVoice"
            
            payload = {
                "chat_id": message.to,
                "voice": audio_url,
            }

            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            logger.info(
                f"Successfully sent voice message via Telegram to chat_id {message.to}"
            )
        except Exception as e:
            logger.error(
                f"Failed to convert/send voice message via Telegram: {e}"
            )
            # Fall back to sending as text
            logger.info("Falling back to text message")
            self._send_text_message(message)

    def send_photo(self, chat_id: str, photo_url: str, caption: Optional[str] = None) -> None:
        """
        Send an image using Telegram sendPhoto API.
        
        Args:
            chat_id: Telegram chat ID
            photo_url: URL of the photo to send
            caption: Optional caption for the photo
        """
        url = f"{self.api_base_url}/sendPhoto"
        
        payload = {
            "chat_id": chat_id,
            "photo": photo_url,
        }
        
        if caption:
            payload["caption"] = caption

        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            logger.info(
                f"Successfully sent photo via Telegram to chat_id {chat_id}"
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send photo via Telegram: {e}")
            if hasattr(e, "response") and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            raise

    @classmethod
    def from_settings(cls):
        """
        Create a TelegramProvider from environment variables.
        
        Returns:
            TelegramProvider instance
            
        Raises:
            KeyError: If TELEGRAM_BOT_TOKEN is not set
        """
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        if not token:
            raise ValueError(
                "TELEGRAM_BOT_TOKEN environment variable is required for Telegram integration"
            )
        return cls(token=token)


def detect_start_command(message_text: str) -> bool:
    """
    Detect if a message is a /start command.
    
    Args:
        message_text: The message text to check
        
    Returns:
        True if the message is a /start command
    """
    if not message_text:
        return False
    
    return message_text.strip().lower().startswith("/start")


def get_telegram_welcome_message(friend_name: str) -> str:
    """
    Generate a welcome message for new Telegram users.
    
    Args:
        friend_name: Name of the virtual friend
        
    Returns:
        Welcome message text
    """
    return (
        f"Olá! Sou {friend_name}, seu amigo bíblico. 🙏\n\n"
        f"Estou aqui para conversar, ouvir e caminhar com você na fé. "
        f"Pode compartilhar o que está no seu coração - estou aqui para você!"
    )
