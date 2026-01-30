import logging
import os

from twilio.rest import Client

from core.services.text_to_speech import TextToSpeechService
from messaging.providers.base import MessagingProvider
from messaging.types import OutgoingMessage

logger = logging.getLogger(__name__)


class TwilioWhatsAppProvider(MessagingProvider):
    def __init__(self, client: Client):
        self.client = client

    def send(self, message: OutgoingMessage) -> None:
        # If reply should be audio, convert text to speech and send as media
        if message.reply_as_audio:
            try:
                audio_url = TextToSpeechService().speak_and_store(
                    text=message.text, conversation_mode=message.conversation_mode
                )
                self.client.messages.create(
                    from_=message.from_,
                    to=message.to,
                    body="",
                    media_url=[audio_url],
                )
                return
            except Exception as e:
                logger.error(
                    f"Failed to convert audio message via Twilio WhatsApp: {e}"
                )

        # Default: send plain text
        self.client.messages.create(
            from_=message.from_,
            to=message.to,
            body=message.text,
        )

    @classmethod
    def from_settings(cls):
        return TwilioWhatsAppProvider(
            client=Client(
                os.environ["TWILIO_ACCOUNT_SID"],
                os.environ["TWILIO_AUTH_TOKEN"],
            )
        )
