"""
Telegram Bot integration and messaging functionality.

This module handles Telegram Bot API message processing, user management,
and integration with Telegram Bot API.
"""

import json
import logging
import os
import random
from typing import Optional

import requests
from django.contrib.auth.models import User

from core.constants import biblical_names, ConversationMode
from core.models import UserSpiritualProfile, VirtualFriend
from messaging.types import IncomingMessage, OutgoingMessage
from service.llm import LLMMessage, get_llm_client
from service.orchestration import chat_with_friend
from service.prompts import build_gender_inference_prompt

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
            logger.error(f"Failed to convert/send voice message via Telegram: {e}")
            # Fall back to sending as text
            logger.info("Falling back to text message")
            self._send_text_message(message)

    def send_photo(
        self, chat_id: str, photo_url: str, caption: Optional[str] = None
    ) -> None:
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
            logger.info(f"Successfully sent photo via Telegram to chat_id {chat_id}")
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
            ValueError: If TELEGRAM_BOT_TOKEN is not set
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


def handle_incoming_message(msg: IncomingMessage) -> OutgoingMessage:
    """
    Handle an incoming message from Telegram.
    
    Args:
        msg: The incoming message
        
    Returns:
        The outgoing response message
    """
    from messaging.types import CHANNEL_TELEGRAM
    
    # Check for /start command
    if detect_start_command(msg.text):
        # Handle /start command
        friend = get_friend_or_init_person(msg)
        
        # Send welcome message
        return OutgoingMessage(
            channel=msg.channel,
            from_=msg.to,
            to=msg.from_,
            text=get_telegram_welcome_message(friend.name),
            reply_as_audio=False,
            conversation_mode=ConversationMode.LISTENING,
        )

    identity = extract_identity(msg)
    friend = get_friend_or_init_person(msg)

    result, conversation = chat_with_friend(
        friend=friend, user_text=msg.text or "", llm=get_llm_client(), identity=identity
    )

    return OutgoingMessage(
        channel=msg.channel,
        from_=msg.to,
        to=msg.from_,
        text=result.text,
        reply_as_audio=msg.reply_as_audio,
        conversation_mode=ConversationMode(conversation.current_mode),
    )


def extract_identity(msg: IncomingMessage) -> dict:
    """
    Extract identity information from an incoming Telegram message.
    
    Args:
        msg: The incoming message
        
    Returns:
        Dictionary with channel-specific identity information
    """
    from messaging.types import CHANNEL_TELEGRAM
    
    # For Telegram, extract from the normalized message or raw payload
    payload = msg.raw_payload or {}
    message = payload.get("message") or payload.get("edited_message") or {}
    sender = message.get("from", {})
    chat = message.get("chat", {})
    
    sender_id = str(sender.get("id", "")) if sender.get("id") else ""
    chat_id = str(chat.get("id", "")) if chat.get("id") else ""
    
    return {
        "channel": CHANNEL_TELEGRAM,
        "user_id": sender_id,
        "chat_id": chat_id,
        "from": msg.from_,
        "to": msg.to,
    }


def get_friend_or_init_person(msg: IncomingMessage) -> VirtualFriend:
    """
    Get or create a VirtualFriend for the user based on the incoming Telegram message.
    
    Args:
        msg: The incoming message
        
    Returns:
        VirtualFriend instance for the user
    """
    from messaging.types import CHANNEL_TELEGRAM
    
    user = User.objects.filter(username=msg.from_).first()
    names = biblical_names
    
    if not user and msg.raw_payload:
        # Try to extract profile name from Telegram format
        first_name = None
        last_name = None

        try:
            message = msg.raw_payload.get("message") or msg.raw_payload.get("edited_message") or {}
            sender = message.get("from", {})
            first_name = sender.get("first_name", "")
            last_name = sender.get("last_name", "")
        except (KeyError, IndexError, AttributeError):
            pass

        user, created = User.objects.get_or_create(
            username=msg.from_,
            defaults={
                "first_name": first_name or "",
                "last_name": last_name or "",
                "is_active": True,
            }
        )

        if created:
            gender_found = infer_gender_from_name(
                name=first_name,
                country="Brasil",
            )

            UserSpiritualProfile.objects.create(user=user, gender=gender_found)

            names = [b for b in biblical_names if b["gender"] == gender_found]

    friend_name = random.choice(names)
    friend, _ = VirtualFriend.objects.get_or_create(
        owner=user,
        defaults={
            "name": friend_name.get("name"),
            "gender": friend_name.get("gender"),
        },
    )

    return friend


def infer_gender_from_name(*, name: str, country: str) -> str:
    """
    Returns: 'male', 'female', or 'unknown'
    """
    if not name:
        return "unknown"

    llm = get_llm_client()
    prompt = build_gender_inference_prompt(
        profile_name=name,
        country=country,
    )

    resp = llm.chat(
        messages=[LLMMessage(role="system", content=prompt)],
        temperature=0.0,
    )

    try:
        data = json.loads(resp.text)
        gender = data.get("gender", "unknown")
    except json.JSONDecodeError:
        return "unknown"

    return gender if gender in {"male", "female", "unknown"} else "unknown"
