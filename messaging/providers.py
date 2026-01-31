"""
Provider adapters for normalizing webhook requests from different messaging platforms.

This module implements a strategy pattern for handling requests from different providers:
- WhatsApp (via Facebook/Meta Graph API)
- Facebook Messenger (via Facebook/Meta Graph API)
- Twilio (WhatsApp and SMS)

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
        provider: Source provider (whatsapp, facebook, twilio)
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


class WhatsAppAdapter(ProviderAdapter):
    """
    Adapter for WhatsApp messages via Facebook/Meta Graph API.
    
    WhatsApp webhook structure:
    {
      "object": "whatsapp_business_account",
      "entry": [{
        "changes": [{
          "value": {
            "messaging_product": "whatsapp",
            "metadata": {"phone_number_id": "...", "display_phone_number": "..."},
            "messages": [{
              "from": "SENDER_ID",
              "type": "text",
              "text": {"body": "MESSAGE"}
            }]
          }
        }]
      }]
    }
    """
    
    def can_handle(self, headers: Dict[str, str], body: Dict[str, Any]) -> bool:
        """Check if this is a WhatsApp webhook request."""
        # WhatsApp webhooks have object="whatsapp_business_account"
        if body.get("object") == "whatsapp_business_account":
            return True
        
        # Also check for messaging_product="whatsapp" in the payload
        entry = body.get("entry", [{}])[0]
        changes = entry.get("changes", [{}])[0]
        value = changes.get("value", {})
        return value.get("messaging_product") == "whatsapp"
    
    def normalize(self, headers: Dict[str, str], body: Dict[str, Any]) -> Optional[NormalizedMessage]:
        """Normalize WhatsApp webhook payload."""
        for entry in body.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                
                # Only process message changes
                if change.get("field") != "messages":
                    continue
                
                metadata = value.get("metadata", {})
                phone_number_id = metadata.get("phone_number_id")
                display_phone_number = metadata.get("display_phone_number", "")
                
                # Process each message
                for message in value.get("messages", []):
                    sender_id = message.get("from")
                    message_type = message.get("type")
                    timestamp = message.get("timestamp")
                    
                    # Extract message content based on type
                    message_body = ""
                    media_url = None
                    reply_as_audio = False
                    
                    if message_type == "text":
                        message_body = message.get("text", {}).get("body", "")
                    elif message_type == "audio":
                        audio_data = message.get("audio", {})
                        media_url = audio_data.get("id")
                        message_body = "[Audio message received]"
                        reply_as_audio = True
                    elif message_type == "image":
                        message_body = "[Image message received]"
                    else:
                        message_body = f"[{message_type} message received]"
                    
                    return NormalizedMessage(
                        sender_id=sender_id,
                        recipient_id=display_phone_number if display_phone_number else phone_number_id,
                        message_body=message_body,
                        message_type=message_type,
                        timestamp=timestamp,
                        provider="whatsapp",
                        media_url=media_url,
                        reply_as_audio=reply_as_audio,
                        raw_payload=body,
                    )
        
        return None


class FacebookMessengerAdapter(ProviderAdapter):
    """
    Adapter for Facebook Messenger messages via Facebook/Meta Graph API.
    
    Facebook Messenger webhook structure:
    {
      "object": "page",
      "entry": [{
        "id": "PAGE_ID",
        "messaging": [{
          "sender": {"id": "SENDER_ID"},
          "recipient": {"id": "PAGE_ID"},
          "timestamp": 1234567890,
          "message": {
            "mid": "MESSAGE_ID",
            "text": "MESSAGE"
          }
        }]
      }]
    }
    """
    
    def can_handle(self, headers: Dict[str, str], body: Dict[str, Any]) -> bool:
        """Check if this is a Facebook Messenger webhook request."""
        # Facebook Messenger webhooks have object="page"
        if body.get("object") == "page":
            # Verify it has the messaging array (not other page events)
            entry = body.get("entry", [{}])[0]
            return "messaging" in entry
        return False
    
    def normalize(self, headers: Dict[str, str], body: Dict[str, Any]) -> Optional[NormalizedMessage]:
        """Normalize Facebook Messenger webhook payload."""
        for entry in body.get("entry", []):
            for messaging_event in entry.get("messaging", []):
                sender = messaging_event.get("sender", {})
                recipient = messaging_event.get("recipient", {})
                timestamp = messaging_event.get("timestamp")
                message = messaging_event.get("message", {})
                
                # Skip if no message (could be delivery receipt, read receipt, etc.)
                if not message:
                    continue
                
                sender_id = sender.get("id")
                recipient_id = recipient.get("id")
                message_body = message.get("text", "")
                
                # Check for attachments
                attachments = message.get("attachments", [])
                media_url = None
                message_type = "text"
                
                if attachments:
                    attachment = attachments[0]
                    attachment_type = attachment.get("type")
                    message_type = attachment_type
                    
                    if attachment_type == "image":
                        media_url = attachment.get("payload", {}).get("url")
                        message_body = message_body or "[Image message received]"
                    elif attachment_type == "audio":
                        media_url = attachment.get("payload", {}).get("url")
                        message_body = message_body or "[Audio message received]"
                    elif attachment_type == "video":
                        media_url = attachment.get("payload", {}).get("url")
                        message_body = message_body or "[Video message received]"
                    elif attachment_type == "file":
                        media_url = attachment.get("payload", {}).get("url")
                        message_body = message_body or "[File received]"
                    else:
                        message_body = message_body or f"[{attachment_type} received]"
                
                return NormalizedMessage(
                    sender_id=sender_id,
                    recipient_id=recipient_id,
                    message_body=message_body,
                    message_type=message_type,
                    timestamp=str(timestamp) if timestamp else None,
                    provider="facebook",
                    media_url=media_url,
                    reply_as_audio=False,
                    raw_payload=body,
                )
        
        return None


class TwilioAdapter(ProviderAdapter):
    """
    Adapter for Twilio messages (WhatsApp and SMS).
    
    Twilio webhook structure (form-encoded):
    {
      "MessageSid": "MESSAGE_ID",
      "From": "whatsapp:+1234567890",
      "To": "whatsapp:+0987654321",
      "Body": "MESSAGE",
      "NumMedia": "0",
      "MediaUrl0": "...",  # if NumMedia > 0
      "MediaContentType0": "..."
    }
    """
    
    def can_handle(self, headers: Dict[str, str], body: Dict[str, Any]) -> bool:
        """Check if this is a Twilio webhook request."""
        # Twilio webhooks include specific headers
        user_agent = headers.get("User-Agent", "").lower()
        if "twilio" in user_agent:
            return True
        
        # Also check for Twilio-specific fields in the body
        # Twilio always sends MessageSid and AccountSid
        return "MessageSid" in body or "AccountSid" in body
    
    def normalize(self, headers: Dict[str, str], body: Dict[str, Any]) -> Optional[NormalizedMessage]:
        """Normalize Twilio webhook payload."""
        sender_id = body.get("From", "")
        recipient_id = body.get("To", "")
        message_body = body.get("Body", "")
        timestamp = body.get("DateCreated")
        
        # Determine provider (whatsapp or sms)
        provider = "twilio"
        if sender_id.startswith("whatsapp:"):
            provider = "twilio_whatsapp"
        
        # Check for media
        num_media = int(body.get("NumMedia", "0"))
        media_url = None
        message_type = "text"
        
        if num_media > 0:
            media_url = body.get("MediaUrl0")
            media_content_type = body.get("MediaContentType0", "")
            
            if media_content_type.startswith("image"):
                message_type = "image"
                message_body = message_body or "[Image message received]"
            elif media_content_type.startswith("audio"):
                message_type = "audio"
                message_body = message_body or "[Audio message received]"
            elif media_content_type.startswith("video"):
                message_type = "video"
                message_body = message_body or "[Video message received]"
            else:
                message_type = "media"
                message_body = message_body or "[Media received]"
        
        return NormalizedMessage(
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_body=message_body,
            message_type=message_type,
            timestamp=timestamp,
            provider=provider,
            media_url=media_url,
            reply_as_audio=False,
            raw_payload=body,
        )


class ProviderDetector:
    """
    Detects the provider from a webhook request and returns the appropriate adapter.
    """
    
    def __init__(self):
        self.adapters = [
            WhatsAppAdapter(),
            FacebookMessengerAdapter(),
            TwilioAdapter(),
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
