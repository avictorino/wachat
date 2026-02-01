# Multi-Provider Webhook Support

## Overview

The webhook endpoint at `/api/webhooks/whatsapp-facebook/` now supports multiple messaging providers through a unified architecture. This document describes the implementation and how to extend it.

## Supported Providers

1. **WhatsApp** (via Facebook/Meta Graph API)
2. **Facebook Messenger** (via Facebook/Meta Graph API)
3. **Twilio** (WhatsApp and SMS)
4. **Telegram** (Bot API)
5. **Slack** (Events API)

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
    provider: str           # whatsapp, facebook, twilio, telegram, slack
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

### Telegram
- Check for `update_id` field in body
- Check for `message` or `edited_message` field
- Supports text, voice, audio, photo, and document messages

### Slack
- Check for `type: "event_callback"` in body
- Check for `event.type: "message"` and absence of `bot_id`
- Supports text and file attachments (images, audio, documents)

## Media Support

All providers support audio and image messages with the following capabilities:

### Audio Messages

**WhatsApp**: 
- Receives audio with `file_id` for later download
- Sets `reply_as_audio: true`

**Facebook Messenger**:
- Audio attachments with direct URL
- Supports various audio formats

**Twilio**:
- Audio via `MediaUrl` with content type
- Supports MP3, WAV, etc.

**Telegram**:
- Voice messages via `file_id` (requires separate API call to download)
- Audio files via `file_id`
- Sets `reply_as_audio: true` for voice messages

**Slack**:
- Audio files via `url_private` (requires authentication)
- Stores file metadata in format: `file_id|mime_type|url`
- Sets `reply_as_audio: true` for audio files

### Image Messages

**WhatsApp**:
- Image messages detected by `type: "image"`
- Media ID provided for download

**Facebook Messenger**:
- Image attachments with direct URL
- Supports JPEG, PNG, GIF

**Twilio**:
- Image via `MediaUrl` with content type
- Direct download URLs

**Telegram**:
- Photos as array of sizes (adapter selects largest)
- Images in documents with MIME type check
- Both use `file_id` for download

**Slack**:
- Image files via `url_private`
- Requires bearer token for download
- Metadata stored for later processing

## Provider Limitations

### Telegram
- File downloads require separate API call using `file_id`
- Bot must be authorized to access files
- File size limits apply (20MB for photos)

### Slack
- File URLs require authentication with bot token
- Does not have native voice messages (audio treated as file)
- Files may expire after certain period

## Adding a New Provider

To add support for a new provider (e.g., Discord):

1. **Create adapter** in `messaging/providers.py`:

```python
class DiscordAdapter(ProviderAdapter):
    def can_handle(self, headers: Dict[str, str], body: Dict[str, Any]) -> bool:
        # Discord webhooks have specific structure
        return body.get("type") == 0 and "author" in body
    
    def normalize(self, headers: Dict[str, str], body: Dict[str, Any]) -> Optional[NormalizedMessage]:
        return NormalizedMessage(
            sender_id=str(body.get("author", {}).get("id")),
            recipient_id=str(body.get("channel_id")),
            message_body=body.get("content", ""),
            message_type="text",
            timestamp=body.get("timestamp"),
            provider="discord",
            raw_payload=body,
        )
```

2. **Register adapter** in `ProviderDetector.__init__()`:

```python
self.adapters = [
    WhatsAppAdapter(),
    FacebookMessengerAdapter(),
    TwilioAdapter(),
    TelegramAdapter(),
    SlackAdapter(),
    DiscordAdapter(),  # Add here
]
```

3. **Update channel types** in `messaging/types.py`:

```python
ChannelType = Literal["whatsapp_facebook", "facebook", "twilio", "twilio_whatsapp", "telegram", "slack", "discord"]
```

4. **Add channel mapping** in `core/views.py`:

```python
channel_map = {
    "whatsapp": "whatsapp_facebook",
    "facebook": "facebook",
    "twilio": "twilio",
    "twilio_whatsapp": "twilio_whatsapp",
    "telegram": "telegram",
    "slack": "slack",
    "discord": "discord",  # Add here
}
```

5. **Write tests** in `messaging/tests.py`:

```python
class DiscordAdapterTest(TestCase):
    def test_can_handle_discord_webhook(self):
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
