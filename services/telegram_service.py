"""
Telegram messaging service.

This module handles sending messages back to users via the Telegram Bot API.
"""

import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class TelegramService:
    """Service class for interacting with Telegram Bot API."""

    def __init__(self):
        """Initialize Telegram service with bot token from environment."""
        self.bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        if not self.bot_token:
            logger.error("TELEGRAM_BOT_TOKEN environment variable is required")
            raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")
        
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

    def send_message(
        self, 
        chat_id: str, 
        text: str,
        parse_mode: Optional[str] = None
    ) -> bool:
        """
        Send a text message to a Telegram chat.
        
        Args:
            chat_id: The Telegram chat ID to send the message to
            text: The message text to send
            parse_mode: Optional parse mode (e.g., 'Markdown', 'HTML')
            
        Returns:
            True if the message was sent successfully, False otherwise
        """
        try:
            url = f"{self.base_url}/sendMessage"
            
            payload = {
                "chat_id": chat_id,
                "text": text
            }
            
            if parse_mode:
                payload["parse_mode"] = parse_mode
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"Message sent successfully to chat {chat_id}")
                return True
            else:
                logger.error(
                    f"Failed to send message to chat {chat_id}. "
                    f"Status: {response.status_code}, Response: {response.text}"
                )
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending message to Telegram: {str(e)}", exc_info=True)
            return False
