from dataclasses import dataclass
from typing import Literal, Optional

from core.constants import ConversationMode

ChannelType = Literal["whatsapp_facebook", "facebook", "twilio", "twilio_whatsapp", "telegram", "slack"]

# Channel name constants
CHANNEL_WHATSAPP_FACEBOOK = "whatsapp_facebook"
CHANNEL_FACEBOOK = "facebook"
CHANNEL_TWILIO = "twilio"
CHANNEL_TWILIO_WHATSAPP = "twilio_whatsapp"
CHANNEL_TELEGRAM = "telegram"
CHANNEL_SLACK = "slack"


@dataclass(frozen=True)
class IncomingMessage:
    channel: ChannelType
    to: str
    from_: str
    text: Optional[str]
    media_url: Optional[str] = None
    raw_payload: Optional[dict] = None
    reply_as_audio: bool = False


@dataclass(frozen=True)
class OutgoingMessage:
    channel: ChannelType
    to: str
    from_: str
    text: str
    reply_as_audio: bool = False
    conversation_mode: Optional[ConversationMode] = ConversationMode.LISTENING
