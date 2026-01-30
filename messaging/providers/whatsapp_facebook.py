import logging
import os

import requests

from core.services.text_to_speech import TextToSpeechService
from messaging.providers.base import MessagingProvider
from messaging.types import OutgoingMessage

logger = logging.getLogger(__name__)


class FacebookWhatsAppProvider(MessagingProvider):
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
                    "to": message.to.replace("whatsapp:", ""),  # Remove prefix if present
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
            "to": message.to.replace("whatsapp:", ""),  # Remove whatsapp: prefix if present
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
