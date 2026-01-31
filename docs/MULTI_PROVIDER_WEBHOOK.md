# Multi-Provider Webhook Support

## Overview

The webhook endpoint at `/api/webhooks/whatsapp-facebook/` now supports multiple messaging providers through a unified architecture. This document describes the implementation and how to extend it.

## Supported Providers

1. **WhatsApp** (via Facebook/Meta Graph API)
2. **Facebook Messenger** (via Facebook/Meta Graph API)
3. **Twilio** (WhatsApp and SMS)

## Architecture

The implementation follows a **Strategy Pattern** with three main components:

### 1. Provider Adapters (`messaging/providers.py`)

Each provider has an adapter that:
- Detects if a request belongs to that provider (`can_handle()`)
- Normalizes the provider-specific payload into a common structure (`normalize()`)

```python
class ProviderAdapter(ABC):
    @abstractmethod
    def can_handle(self, headers: Dict[str, str], body: Dict[str, Any]) -> bool:
        """Check if this adapter can handle the request"""
        pass
    
    @abstractmethod
    def normalize(self, headers: Dict[str, str], body: Dict[str, Any]) -> Optional[NormalizedMessage]:
        """Convert provider-specific request to unified format"""
        pass
```

### 2. Normalized Message Structure

All providers are normalized to:

```python
@dataclass
class NormalizedMessage:
    sender_id: str          # Unique sender identifier
    recipient_id: str       # Unique recipient identifier
    message_body: str       # Text content
    message_type: str       # text, audio, image, video, etc.
    timestamp: Optional[str]
    provider: str           # whatsapp, facebook, twilio, twilio_whatsapp
    media_url: Optional[str] = None
    reply_as_audio: bool = False
    raw_payload: Optional[Dict[str, Any]] = None
```

### 3. Provider Detection

The `ProviderDetector` class:
- Iterates through available adapters
- Returns the first adapter that can handle the request
- Normalizes the message using that adapter

```python
detector = ProviderDetector()
provider, normalized_message = detector.detect_and_normalize(headers, body)
```

## Request Flow

1. **Webhook receives request** (`FacebookWhatsAppWebhookView.post()`)
2. **Provider detection** - Identify the source provider
3. **Normalization** - Convert to unified `NormalizedMessage`
4. **Conversion** - Map to existing `IncomingMessage` type
5. **Processing** - Call existing `process_message_task()`

## Provider Detection Logic

### WhatsApp (Facebook)
- Check for `object: "whatsapp_business_account"` in body
- Check for `messaging_product: "whatsapp"` in payload

### Facebook Messenger
- Check for `object: "page"` in body
- Verify presence of `messaging` array in entry

### Twilio
- Check for `User-Agent` header containing "twilio"
- Check for `MessageSid` or `AccountSid` in body
- Distinguish WhatsApp vs SMS by `From` field prefix

## Adding a New Provider

To add support for a new provider (e.g., Telegram):

1. **Create adapter** in `messaging/providers.py`:

```python
class TelegramAdapter(ProviderAdapter):
    def can_handle(self, headers: Dict[str, str], body: Dict[str, Any]) -> bool:
        # Telegram has update_id and message fields
        return "update_id" in body and "message" in body
    
    def normalize(self, headers: Dict[str, str], body: Dict[str, Any]) -> Optional[NormalizedMessage]:
        message = body.get("message", {})
        return NormalizedMessage(
            sender_id=str(message.get("from", {}).get("id")),
            recipient_id=str(message.get("chat", {}).get("id")),
            message_body=message.get("text", ""),
            message_type="text",
            timestamp=str(message.get("date")),
            provider="telegram",
            raw_payload=body,
        )
```

2. **Register adapter** in `ProviderDetector.__init__()`:

```python
self.adapters = [
    WhatsAppAdapter(),
    FacebookMessengerAdapter(),
    TwilioAdapter(),
    TelegramAdapter(),  # Add here
]
```

3. **Update channel types** in `messaging/types.py`:

```python
ChannelType = Literal["whatsapp_facebook", "facebook", "twilio", "twilio_whatsapp", "telegram"]
```

4. **Add channel mapping** in `core/views.py`:

```python
channel_map = {
    "whatsapp": "whatsapp_facebook",
    "facebook": "facebook",
    "twilio": "twilio",
    "twilio_whatsapp": "twilio_whatsapp",
    "telegram": "telegram",  # Add here
}
```

5. **Write tests** in `messaging/tests.py`:

```python
class TelegramAdapterTest(TestCase):
    def test_can_handle_telegram_webhook(self):
        # Test detection logic
        
    def test_normalize_text_message(self):
        # Test normalization
```

## Testing

Run tests for provider adapters:

```bash
python manage.py test messaging.tests
```

Run webhook integration tests:

```bash
python manage.py test core.tests.MultiProviderWebhookViewTest
```

## Backward Compatibility

The existing WhatsApp functionality is **fully preserved**:
- Same endpoint URL
- Same `IncomingMessage` structure internally
- Same processing pipeline
- All existing tests pass

## Security Considerations

1. **CSRF Protection**: Disabled for webhook endpoint (webhooks don't use CSRF tokens)
2. **Verification**: WhatsApp/Facebook webhooks use GET verification with tokens
3. **Unknown Providers**: Gracefully handled (returns 200 OK but doesn't process)
4. **Input Validation**: JSON parsing with error handling

## Environment Variables

No new environment variables are required. Existing variables still work:

- `FACEBOOK_TOKEN` - Facebook/WhatsApp API token
- `FACEBOOK_PHONE_NUMBER_ID` - WhatsApp Business phone number ID
- `FACEBOOK_WEBHOOK_VERIFICATION` - Webhook verification token

## Performance

The provider detection is efficient:
- **Early exit**: Returns on first matching adapter
- **Simple checks**: Most detection uses simple dict lookups
- **No database queries**: Detection happens in-memory

## Monitoring

The webhook logs include provider information:

```python
logger.info(
    f"Received message from provider: {provider}",
    extra={
        "provider": provider,
        "sender": normalized_message.sender_id,
        "message_type": normalized_message.message_type,
    },
)
```

## Future Enhancements

Potential improvements:
1. Provider-specific response formatting
2. Provider-specific rate limiting
3. Provider-specific message queuing
4. Webhook signature verification for each provider
5. Provider-specific error handling and retries
