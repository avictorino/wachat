from dataclasses import dataclass
from typing import Literal, Optional

from core.constants import ConversationMode

ChannelType = Literal["whatsapp", "telegram", "web"]


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
