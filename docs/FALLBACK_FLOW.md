# Fallback Conversational Flow Implementation

## Overview

This implementation provides a graceful, context-aware fallback conversational flow for when a user's message does NOT match any predefined intent category. This is particularly important for maintaining natural conversation when users express spiritual seeking in ways that don't fit into predefined categories.

## Key Features

### 1. Context Awareness
- **Retrieves last N messages** (default: 5) from the conversation history
- Includes both user and assistant messages
- Maintains conversational coherence and continuity
- Does NOT re-explain who the system is

### 2. Script-Driven Response
- **No parameter passing** - doesn't pass flags like `mood` or `topic`
- Behavior is guided by a fixed prompt/script embedded in the LLM call
- The script instructs the LLM how to behave in ambiguity
- Treats ambiguity as a valid conversational state, not an error

### 3. Conversational Script
The LLM:
- Acknowledges the user's message respectfully
- Reflects the intention behind the message (listening, guidance, comfort)
- Avoids labeling the user's state (e.g., "você está distante")
- Avoids assumptions
- Avoids religious authority language
- Uses soft, pastoral, human tone

### 4. Response Length and Structure
- **Short responses**: 1-3 paragraphs OR 2 separate messages
- Messages can be split using `|||` separator
- Feels like natural chat, not written reflection
- Multi-message responses sent sequentially with 1.5s pauses

### 5. Question Strategy
- Questions are **optional**, not forced
- If included:
  - Must be open-ended
  - Must feel like invitation, not prompt
  - Maximum ONE question
- Acceptable to respond with NO question

### 6. Persistence
- Each assistant message saved separately if multiple messages sent
- All messages linked to user Profile
- Chronological order preserved

### 7. Delivery
- Messages sent sequentially to Telegram
- 1.5 second pause between messages
- Natural conversational feel

## Implementation Components

### LLM Service

#### `generate_fallback_response()`
```python
def generate_fallback_response(
    self,
    user_message: str,
    conversation_context: List[dict],
    name: str,
    inferred_gender: Optional[str] = None,
) -> List[str]:
```

- Takes conversation context as list of `{"role": "...", "content": "..."}`
- Returns list of message strings (1-3 messages)
- Script embedded in system prompt guides LLM behavior
- No parameters passed to influence response generation

#### `_split_response_messages()`
```python
def _split_response_messages(self, response: str) -> List[str]:
```

- Splits LLM response on `|||` separator
- Returns list of individual messages
- Limits to 3 messages maximum
- Handles whitespace cleanup

### TelegramService

#### `send_messages()`
```python
def send_messages(
    self,
    chat_id: str,
    messages: List[str],
    pause_seconds: float = 1.0,
    parse_mode: Optional[str] = None,
) -> bool:
```

- Sends multiple messages sequentially
- Pauses between messages (default 1.0s, used as 1.5s in practice)
- Returns success status

### TelegramWebhookView

#### `_get_conversation_context()`
```python
def _get_conversation_context(self, profile, limit: int = 5) -> list:
```

- Retrieves last N messages for a profile
- Excludes system messages
- Returns chronological order (oldest to newest)
- Used to provide context to fallback response generation

#### Updated `_handle_regular_message()`
- Generates conversational responses using fallback flow
- Assembles conversation context
- Generates response (may be multiple messages)
- Persists each message separately
- Sends messages sequentially with pauses

## Example Scenario

### User Message
```
"Gostaria de ouvir um pouco da palavra de um bom pastor"
```

### System Response
**Conversational Flow Activated:**

1. **Context Retrieved** (last 5 messages):
   ```
   ASSISTANT: Olá João! Bem-vindo...
   USER: Oi, obrigado pelo acolhimento.
   ASSISTANT: Fico feliz em ter você aqui...
   USER: Gostaria de ouvir um pouco da palavra...
   ```

2. **LLM Response Generated** (2 messages):
   ```
   Message 1: "Entendo o que você busca."
   Message 2: "Estou aqui para caminhar junto com você nessa jornada espiritual."
   ```

3. **Messages Persisted**:
   - Two separate Message objects created
   - Both linked to user Profile
   - Both with role="assistant"

4. **Messages Delivered**:
   - First message sent to Telegram
   - 1.5s pause
   - Second message sent to Telegram

## Spiritual Tone Guidelines

The fallback script enforces:
- ✅ Soft, pastoral, human presence
- ✅ Listening, not teaching
- ✅ Walking alongside, not leading
- ✅ Questions as invitations, not interrogations
- ❌ No sermons
- ❌ No Bible verses
- ❌ No moral instruction
- ❌ No religious authority language

## Testing

### Unit Tests
- `FallbackConversationalFlowTest`: Tests conversational flow
- `TelegramServiceMultiMessageTest`: Tests sequential message sending

### Manual Testing
Run `/tmp/test_fallback_flow.py` to see a demonstration of:
- Context assembly
- Fallback response generation
- Multi-message persistence
- Natural flow maintenance

## Configuration

### Environment Variables
No additional environment variables required. Uses existing:
- `LLM_PROVIDER`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_WEBHOOK_SECRET`

### Adjustable Parameters
In code:
- Context limit: `limit=5` in `_get_conversation_context()`
- Pause duration: `pause_seconds=1.5` in `send_messages()` call
- Max messages: `messages[:3]` in `_split_response_messages()`
- Temperature: `temperature=0.85` in `generate_fallback_response()`

## Technical Notes

### Message Format
- Messages use plain text (no markdown, no emojis)
- Natural Brazilian Portuguese
- Conversational, not formal

### Error Handling
- If LLM API fails, returns simple fallback message
- If message sending fails, logs error but doesn't crash
- Partial send failures reported but don't block subsequent messages

## Future Enhancements

Possible improvements:
1. Adaptive context window based on conversation length
2. Dynamic pause duration based on message length
3. Sentiment analysis to adjust tone
4. A/B testing different script variations
5. User feedback collection on response quality
