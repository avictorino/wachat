from messaging.providers.telegram import TelegramProvider
from messaging.providers.whatsapp_twilio import TwilioWhatsAppProvider


def get_provider(channel: str):
    if channel == "whatsapp":
        return TwilioWhatsAppProvider.from_settings()
    if channel == "telegram":
        return TelegramProvider.from_settings()
    if channel == "web":
        raise NotImplementedError("WebSocketProvider is not implemented yet.")

    raise ValueError(f"Unsupported channel: {channel}")
