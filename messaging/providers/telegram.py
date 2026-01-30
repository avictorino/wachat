from messaging.providers.base import MessagingProvider
from messaging.types import OutgoingMessage


class TelegramProvider(MessagingProvider):
    def __init__(self, bot):
        self.bot = bot

    def send(self, message: OutgoingMessage) -> None:
        self.bot.send_message(
            chat_id=message.recipient_id,
            text=message.text,
        )
