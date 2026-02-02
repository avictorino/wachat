# Simulate Conversation Management Command

## Overview

The `simulate_conversation` management command allows you to simulate realistic conversations between an AI-driven human user and the WaChat bot system. This is useful for:

- Testing the full conversation flow end-to-end
- Validating funnel progression and state management
- Generating test data for development
- Demonstrating the bot's conversational capabilities
- Stress-testing the webhook pipeline

## Features

- **Realistic Human Simulation**: Uses AI (Groq) to generate emotionally-driven, imperfect messages that reflect real spiritual and life struggles
- **Full Pipeline Integration**: All messages flow through the actual webhook/view (no shortcuts or direct service calls)
- **Database Persistence**: Everything is stored in the database exactly as in production
- **Flexible Configuration**: Customize conversation domain, number of turns, delays, and user names
- **Mock Mode**: Test without real API calls using `--mock-telegram` flag

## Usage

### Basic Usage

```bash
python manage.py simulate_conversation
```

This will run a 5-turn conversation with default settings.

### With Custom Options

```bash
python manage.py simulate_conversation --turns 10 --domain grief --name "Ana Costa" --delay 1.5
```

### Mock Mode (for testing)

```bash
python manage.py simulate_conversation --mock-telegram --turns 3
```

## Command Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--turns` | int | 5 | Number of conversation turns |
| `--domain` | str | spiritual | Conversation domain (spiritual, grief, relationship, faith, anxiety, etc.) |
| `--name` | str | (random) | Name for the simulated user |
| `--delay` | float | 2.0 | Delay in seconds between messages |
| `--mock-telegram` | flag | False | Mock Telegram API calls for testing |

## Conversation Domains

The `--domain` parameter influences the type of messages the simulated human generates:

- `spiritual` - Questions about faith and spirituality
- `grief` - Dealing with loss and mourning
- `relationship` - Family and relationship struggles
- `faith` - Doubts and questions about belief
- `anxiety` - Worry, stress, and fear
- `financial` - Money and job concerns

## How It Works

1. **Profile Creation**: A new Profile is created automatically via the `/start` webhook command
2. **Welcome Message**: The bot generates a personalized welcome message using the user's name
3. **Conversation Turns**: For each turn:
   - The `HumanSimulator` generates a realistic message using AI
   - The message is sent through the Telegram webhook endpoint
   - The bot processes the message and generates a response
   - All messages are persisted in the database
4. **Summary**: At the end, a summary shows the profile ID, message counts, and how to review the conversation

## Architecture

### Components

1. **Management Command** (`core/management/commands/simulate_conversation.py`)
   - Main entry point
   - Orchestrates the simulation flow
   - Handles command-line arguments
   - Displays formatted output

2. **Human Simulator** (`services/human_simulator.py`)
   - Uses Groq AI to generate realistic human messages
   - Maintains emotional state across conversation
   - Adapts messages based on conversation history
   - Handles fallbacks if AI is unavailable

3. **Webhook Integration**
   - Uses Django test client to call the actual webhook
   - Simulates Telegram's webhook payload format
   - Ensures messages flow through the real pipeline

### Data Flow

```
simulate_conversation command
    ↓
HumanSimulator generates message
    ↓
Django test client → TelegramWebhookView
    ↓
GroqService processes message
    ↓
TelegramService (mocked or real)
    ↓
Database (Profile, Message models)
```

## Environment Variables

Required:
- `TELEGRAM_WEBHOOK_SECRET` - Webhook authentication secret
- `TELEGRAM_BOT_TOKEN` - Bot token (can be test value with `--mock-telegram`)
- `GROQ_API_KEY` - Groq API key for AI message generation

## Examples

### Quick Test (3 turns, mocked)
```bash
python manage.py simulate_conversation --turns 3 --mock-telegram
```

### Realistic Simulation (10 turns, grief domain)
```bash
python manage.py simulate_conversation --turns 10 --domain grief --delay 2.5
```

### Named User Simulation
```bash
python manage.py simulate_conversation --name "João Santos" --domain anxiety --turns 7
```

## Reviewing Generated Conversations

After running the simulation, you can review the conversation in the database:

```bash
# Using the profile ID from the summary
python manage.py shell -c "from core.models import Profile, Message; \
    p = Profile.objects.get(id=1); \
    [print(f'{m.role}: {m.content}') for m in p.messages.all()]"
```

Or use Django admin:
```bash
python manage.py runserver
# Navigate to http://localhost:8000/admin/core/profile/
```

## Development

### Adding New Conversation Domains

To add new domains, update the `HumanSimulator` class in `services/human_simulator.py` to include domain-specific prompts and emotional patterns.

### Extending the Simulator

The `HumanSimulator` class can be extended to:
- Add more sophisticated emotional state machines
- Include memory of past conversations
- Simulate different personality types
- Add typing delays and message editing

## Troubleshooting

### "Missing required environment variables"
Make sure your `.env` file includes all required variables (see Environment Variables section).

### "No bot response received"
This usually means the webhook processing failed. Check:
- Database migrations are up to date
- Required services (Groq, Telegram) are configured
- Use `--mock-telegram` to isolate issues

### Connection errors with Groq
If you see connection errors, the `HumanSimulator` will use fallback messages. Ensure your `GROQ_API_KEY` is valid or use `--mock-telegram` mode.

## Future Enhancements

Possible improvements:
- Support for WhatsApp and Facebook channel simulations
- Multi-user conversation simulations
- Conversation branching and decision trees
- Export conversations to JSON/CSV
- Conversation analytics and metrics
- Real-time progress visualization
