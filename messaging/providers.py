"""
Provider adapters for normalizing webhook requests from Telegram Bot API.

This module implements a strategy pattern for handling requests from Telegram.
Each adapter normalizes the provider-specific request format into a common structure.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple


@dataclass
class NormalizedMessage:
    """
    Common message structure after normalization from any provider.
    
    Attributes:
        sender_id: Unique identifier for the sender
        recipient_id: Unique identifier for the recipient
        message_body: The text content of the message
        message_type: Type of message (text, audio, image, etc.)
        timestamp: Message timestamp
        provider: Source provider (whatsapp, facebook, twilio, telegram, slack)
        media_url: URL for media messages (optional)
        reply_as_audio: Whether to reply with audio (optional)
        raw_payload: Original request data for debugging
    """
    sender_id: str
    recipient_id: str
    message_body: str
    message_type: str
    timestamp: Optional[str]
    provider: str
    media_url: Optional[str] = None
    reply_as_audio: bool = False
    raw_payload: Optional[Dict[str, Any]] = None


class ProviderAdapter(ABC):
    """
    Abstract base class for provider adapters.
    
    Each provider adapter must implement:
    - can_handle: Check if this adapter can handle the request
    - normalize: Convert provider-specific request to NormalizedMessage
    """
    
    @abstractmethod
    def can_handle(self, headers: Dict[str, str], body: Dict[str, Any]) -> bool:
        """
        Determine if this adapter can handle the given request.
        
        Args:
            headers: Request headers
            body: Parsed request body
            
        Returns:
            True if this adapter can handle the request
        """
        pass
    
    @abstractmethod
    def normalize(self, headers: Dict[str, str], body: Dict[str, Any]) -> Optional[NormalizedMessage]:
        """
        Normalize the provider-specific request into a common structure.
        
        Args:
            headers: Request headers
            body: Parsed request body
            
        Returns:
            NormalizedMessage if successful, None if message cannot be processed
        """
        pass


class TelegramAdapter(ProviderAdapter):
    """
    Adapter for Telegram Bot API messages.
    
    Telegram webhook structure:
    {
      "update_id": 123456789,
      "message": {
        "message_id": 123,
        "from": {
          "id": 123456789,
          "is_bot": false,
          "first_name": "John"
        },
        "chat": {
          "id": 123456789,
          "type": "private"
        },
        "date": 1234567890,
        "text": "Hello"
      }
    }
    
    Note: Telegram media files require a separate API call to download.
    The adapter stores the file_id which can be used to fetch the file later.
    """
    
    def can_handle(self, headers: Dict[str, str], body: Dict[str, Any]) -> bool:
        """Check if this is a Telegram webhook request."""
        # Telegram webhooks have update_id field
        if "update_id" not in body:
            return False
        
        # Check for message or edited_message
        return "message" in body or "edited_message" in body
    
    def normalize(self, headers: Dict[str, str], body: Dict[str, Any]) -> Optional[NormalizedMessage]:
        """
        Normalize Telegram webhook payload.
        
        Telegram supports:
        - Text messages
        - Audio messages (voice, audio files)
        - Photo messages
        - Document messages
        """
        # Get the message (could be regular message or edited_message)
        message = body.get("message") or body.get("edited_message")
        if not message:
            return None
        
        sender = message.get("from", {})
        chat = message.get("chat", {})
        
        sender_id = str(sender.get("id", ""))
        recipient_id = str(chat.get("id", ""))
        timestamp = str(message.get("date", ""))
        
        # Default values
        message_body = ""
        message_type = "text"
        media_url = None
        reply_as_audio = False
        
        # Extract message content based on type
        if "text" in message:
            message_body = message.get("text", "")
            message_type = "text"
        
        elif "voice" in message:
            # Voice messages are audio messages
            voice = message.get("voice", {})
            media_url = voice.get("file_id")  # Store file_id for later download
            message_body = "[Audio message received]"
            message_type = "audio"
            reply_as_audio = True
        
        elif "audio" in message:
            # Audio files
            audio = message.get("audio", {})
            media_url = audio.get("file_id")
            message_body = "[Audio message received]"
            message_type = "audio"
            reply_as_audio = True
        
        elif "photo" in message:
            # Photo messages - Telegram sends multiple sizes
            # Get the largest photo (last in the array)
            photos = message.get("photo", [])
            if photos:
                largest_photo = photos[-1]
                media_url = largest_photo.get("file_id")
                message_body = message.get("caption", "[Image message received]")
                message_type = "image"
        
        elif "document" in message:
            # Document messages (could be any file type)
            document = message.get("document", {})
            media_url = document.get("file_id")
            mime_type = document.get("mime_type", "")
            
            if mime_type.startswith("image"):
                message_body = message.get("caption", "[Image message received]")
                message_type = "image"
            elif mime_type.startswith("audio"):
                message_body = message.get("caption", "[Audio message received]")
                message_type = "audio"
            else:
                message_body = message.get("caption", "[Document received]")
                message_type = "document"
        
        else:
            # Unsupported message type
            message_body = "[Unsupported message type]"
            message_type = "unknown"
        
        return NormalizedMessage(
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_body=message_body,
            message_type=message_type,
            timestamp=timestamp,
            provider="telegram",
            media_url=media_url,
            reply_as_audio=reply_as_audio,
            raw_payload=body,
        )


class ProviderDetector:
    """
    Detects the provider from a webhook request and returns the appropriate adapter.
    Only supports Telegram Bot API.
    """
    
    def __init__(self):
        self.adapters = [
            TelegramAdapter(),
        ]
    
    def detect_and_normalize(
        self, headers: Dict[str, str], body: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[NormalizedMessage]]:
        """
        Detect the provider and normalize the message.
        
        Args:
            headers: Request headers
            body: Parsed request body
            
        Returns:
            Tuple of (provider_name, normalized_message)
            Returns (None, None) if no adapter can handle the request
        """
        for adapter in self.adapters:
            if adapter.can_handle(headers, body):
                normalized = adapter.normalize(headers, body)
                if normalized:
                    return (normalized.provider, normalized)
        
        return (None, None)
