"""
WhatsApp integration and messaging functionality.

This module handles WhatsApp message processing, user management,
and integration with Facebook's WhatsApp Business API.
"""
import json
import logging
import os
import random

import requests
from django.contrib.auth.models import User

from core.constants import biblical_names
from core.models import UserSpiritualProfile, VirtualFriend
from service.llm import LLMMessage, get_llm_client
from service.media_generation import TextToSpeechService
from messaging.types import IncomingMessage, OutgoingMessage
from service.orchestration import chat_with_friend
from service.prompts import build_gender_inference_prompt

logger = logging.getLogger(__name__)


# ============================================================================
# WhatsApp Provider
# ============================================================================


class FacebookWhatsAppProvider:
    """
    WhatsApp provider using Facebook's WhatsApp Business API.
    """

    def __init__(self, token: str, phone_number_id: str):
        self.token = token
        self.phone_number_id = phone_number_id
        self.api_url = f"https://graph.facebook.com/v22.0/{phone_number_id}/messages"

    def send(self, message: OutgoingMessage) -> None:
        """
        Send a message using Facebook's WhatsApp Business API.
        """
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

        # If reply should be audio, convert text to speech and send as media
        if message.reply_as_audio:
            try:
                audio_url = TextToSpeechService().speak_and_store(
                    text=message.text, conversation_mode=message.conversation_mode
                )
                # Send audio as media message
                payload = {
                    "messaging_product": "whatsapp",
                    "to": message.to.replace(
                        "whatsapp:", ""
                    ),  # Remove prefix if present
                    "type": "audio",
                    "audio": {
                        "link": audio_url,
                    },
                }
                response = requests.post(self.api_url, json=payload, headers=headers)
                response.raise_for_status()
                logger.info(
                    f"Successfully sent audio message via Facebook WhatsApp to {message.to}"
                )
                return
            except Exception as e:
                logger.error(
                    f"Failed to convert/send audio message via Facebook WhatsApp: {e}"
                )
                # Fall through to send as text instead

        # Default: send plain text message
        payload = {
            "messaging_product": "whatsapp",
            "to": message.to.replace(
                "whatsapp:", ""
            ),  # Remove whatsapp: prefix if present
            "type": "text",
            "text": {
                "body": message.text,
            },
        }

        try:
            response = requests.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()
            logger.info(
                f"Successfully sent text message via Facebook WhatsApp to {message.to}"
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send message via Facebook WhatsApp: {e}")
            if hasattr(e, "response") and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            raise

    @classmethod
    def from_settings(cls):
        """
        Create a FacebookWhatsAppProvider from environment variables.
        """
        return cls(
            token=os.environ["FACEBOOK_TOKEN"],
            phone_number_id=os.environ["FACEBOOK_PHONE_NUMBER_ID"],
        )


# ============================================================================
# Chat Service
# ============================================================================


def extract_whatsapp_identity(payload: dict) -> dict:
    # Facebook WhatsApp API format
    # For Facebook, we extract from the webhook payload structure
    if not payload:
        return {
            "channel": "whatsapp_facebook",
            "wa_id": None,
            "from": None,
            "to": None,
        }

    # Try to extract from Facebook format
    entry = payload.get("entry", [{}])[0]
    changes = entry.get("changes", [{}])[0]
    value = changes.get("value", {})
    messages = value.get("messages", [{}])[0]

    return {
        "channel": "whatsapp_facebook",
        "wa_id": messages.get("from"),
        "from": messages.get("from"),
        "to": value.get("metadata", {}).get("phone_number_id"),
    }


def extract_identity(msg: IncomingMessage) -> dict:
    """
    Extract identity information from an incoming message based on channel.
    
    Args:
        msg: The incoming message
        
    Returns:
        Dictionary with channel-specific identity information
    """
    from messaging.types import CHANNEL_TELEGRAM
    
    if msg.channel == CHANNEL_TELEGRAM:
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
    else:
        # Use existing WhatsApp identity extraction
        identity = extract_whatsapp_identity(msg.raw_payload)
        identity["user_id"] = identity.get("wa_id") or identity.get("from")
        return identity


def handle_incoming_message(msg: IncomingMessage) -> OutgoingMessage:
    """
    Handle an incoming message from any channel.
    
    Supports:
    - WhatsApp (Facebook)
    - Telegram
    - Other channels as configured
    
    All messages, including Telegram /start command, are processed through
    chat_with_friend to ensure proper tracking and database persistence.
    
    Args:
        msg: The incoming message
        
    Returns:
        The outgoing response message
    """
    identity = extract_identity(msg)

    friend = get_friend_or_init_person(msg)

    result, conversation = chat_with_friend(
        friend=friend, user_text=msg.text or "", llm=get_llm_client(), identity=identity
    )

    from core.constants import ConversationMode

    return OutgoingMessage(
        channel=msg.channel,
        from_=msg.to,  # 👈 invertendo
        to=msg.from_,  # 👈 invertendo
        text=result.text,
        reply_as_audio=msg.reply_as_audio,
        conversation_mode=ConversationMode(conversation.current_mode),
    )


def get_friend_or_init_person(msg: IncomingMessage) -> VirtualFriend:
    """
    Get or create a VirtualFriend for the user based on the incoming message.
    
    Supports multiple channels:
    - WhatsApp (Facebook)
    - Telegram
    
    Args:
        msg: The incoming message
        
    Returns:
        VirtualFriend instance for the user
    """
    from messaging.types import CHANNEL_TELEGRAM
    
    user = User.objects.filter(username=msg.from_).first()
    names = biblical_names
    
    if not user and msg.raw_payload:
        # Try to extract profile name based on channel
        first_name = None
        last_name = None

        if msg.channel == CHANNEL_TELEGRAM:
            # Telegram format
            try:
                message = msg.raw_payload.get("message") or msg.raw_payload.get("edited_message") or {}
                sender = message.get("from", {})
                first_name = sender.get("first_name", "")
                last_name = sender.get("last_name", "")
            except (KeyError, IndexError, AttributeError):
                pass
        else:
            # Facebook WhatsApp format
            try:
                entry = msg.raw_payload.get("entry", [{}])[0]
                changes = entry.get("changes", [{}])[0]
                value = changes.get("value", {})
                contacts = value.get("contacts", [])
                if contacts:
                    profile = contacts[0].get("profile", {})
                    name = profile.get("name", "")
                    if name:
                        name_parts = name.split(" ")
                        first_name = name_parts[0]
                        last_name = name_parts[-1] if len(name_parts) > 1 else name_parts[0]
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
