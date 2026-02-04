# Input Sanitization Service - Documentation

## Overview

The Input Sanitization Service is a reusable safety layer for LLM interactions in the WaChat system. It sanitizes user input by detecting and neutralizing harmful or disallowed topics before the input is sent to the LLM.

## Purpose

This service ensures that:
- Sensitive topics are filtered before reaching the LLM
- User conversations remain safe and appropriate
- The system maintains a protective layer without explicit content moderation messaging
- All LLM calls are consistently protected

## Architecture

### Key Components

1. **InputSanitizer Class** (`services/input_sanitizer.py`)
   - Main class that performs sanitization
   - Contains keyword patterns for harmful content detection
   - Provides the `sanitize()` method

2. **Convenience Functions**
   - `get_sanitizer()`: Returns singleton instance
   - `sanitize_input()`: Direct access to sanitization

3. **Integration Points**
   - `OllamaService.infer_gender()`: Sanitizes names before gender inference
   - `OllamaService.generate_welcome_message()`: Sanitizes names before message generation

## Filtered Content Categories

The service detects and filters three main categories:

### 1. Sexual Content
- Explicit or implicit sexual references
- Inappropriate language
- Abuse-related terms
- Examples: sexo, sexual, pornografia, abuso sexual

### 2. Death-Related Content
- Mentions of death, suicide, or violence
- Self-harm references
- Examples: morte, morrer, suicídio, matar

### 3. Controversial Topics
- Highly polarizing political or social issues
- Extremism and discrimination
- Examples: aborto, fascismo, terrorismo, racismo

## Usage

### Basic Usage

```python
from services.input_sanitizer import sanitize_input

# Sanitize user input
user_input = "Usuário disse algo com conteúdo sensível"
safe_input = sanitize_input(user_input)

# safe_input now has harmful terms replaced with "[tema sensível]"
```

### Using the Singleton

```python
from services.input_sanitizer import get_sanitizer

sanitizer = get_sanitizer()
safe_input = sanitizer.sanitize(user_input)
```

### With Logging Control

```python
from services.input_sanitizer import sanitize_input

# Enable logging (default)
safe_input = sanitize_input(user_input, log_detections=True)

# Disable logging
safe_input = sanitize_input(user_input, log_detections=False)
```

### Integration Example (Already Implemented)

```python
from services.input_sanitizer import sanitize_input

def infer_gender(self, name: str) -> str:
    # Sanitize input before sending to LLM
    sanitized_name = sanitize_input(name)
    
    # Use sanitized_name for LLM prompt
    user_prompt = f"Nome: {sanitized_name}"
    
    # ... rest of LLM call
```

## Behavior

### Safe Input
Clean input passes through unchanged:
```python
sanitize_input("Olá, como você está?")
# Returns: "Olá, como você está?"
```

### Harmful Input
Harmful terms are replaced with `[tema sensível]`:
```python
sanitize_input("Quero falar sobre sexo e morte")
# Returns: "Quero falar sobre [tema sensível] e [tema sensível]"
```

### Preserving Context
The service preserves conversational flow:
```python
sanitize_input("Olá, estou com dúvidas sobre sexualidade")
# Returns: "Olá, estou com dúvidas sobre [tema sensível]"
```

## Key Features

### 1. Language-Aware Filtering
- Designed for Brazilian Portuguese
- Handles word variations (verb conjugations, plurals)
- Uses word boundary matching to avoid false positives

### 2. Non-Intrusive Operation
- Never raises exceptions
- Returns original text if errors occur
- Transparent to users (no explicit moderation messages)

### 3. Observability
- Optional logging of sanitization events
- Logs categories detected, not actual content
- Useful for monitoring and tuning

### 4. Extendable Design
- Easy to add new category patterns
- Centralized keyword management
- Clean separation of concerns

## Error Handling

The service is designed to never break the application:

```python
# Handles None input
sanitize_input(None)  # Returns: ""

# Handles non-string input
sanitize_input(123)  # Returns: "123"

# Handles errors gracefully
# If an error occurs during sanitization, returns original text
```

## Testing

Comprehensive tests are available in:
- `services/test_input_sanitizer.py` - Unit tests for the sanitizer

Run tests:
```bash
python manage.py test services.test_input_sanitizer
```

## Monitoring

When `log_detections=True` (default), sanitization events are logged:

```python
logger.info(
    "Input sanitization performed. Categories detected: ['sexual', 'death']",
    extra={
        'categories': ['sexual', 'death'],
        'input_length': 42,
    }
)
```

**Important**: The actual harmful content is NEVER logged to protect privacy and security.

## Adding New Categories

To add a new content category:

1. Add keyword patterns in `InputSanitizer.__init__()`:
```python
self.new_category_keywords = self._compile_patterns([
    r'\bkeyword1\b',
    r'\bkeyword2\b',
    # ... more patterns
])
```

2. Update `_detect_harmful_content()` to check the new category:
```python
for pattern in self.new_category_keywords:
    matches = pattern.findall(text)
    if matches:
        detections['new_category'].extend(matches)
```

3. Update `sanitize()` to include the new patterns:
```python
all_patterns = (
    self.sexual_keywords +
    self.death_keywords +
    self.controversial_keywords +
    self.new_category_keywords  # Add here
)
```

4. Add tests for the new category

## Design Principles

### Safety by Default
- All LLM calls must go through sanitization
- Favors safety over completeness
- No bypassing mechanism

### Separation of Concerns
- Sanitizer ONLY prepares safe input
- Does NOT generate responses
- Does NOT call the LLM directly

### User-Centric
- Preserves emotional meaning when possible
- No moralizing or judging
- Transparent operation

### Maintainability
- Clean, well-documented code
- Django-style architecture
- Comprehensive test coverage

## Performance Considerations

- Regex patterns are pre-compiled for efficiency
- Singleton pattern reduces initialization overhead
- Minimal overhead for clean input (early return)

## Security Notes

- Never stores raw harmful content in logs
- Operates independently of LLM
- Can be audited and tuned without affecting LLM behavior

## Future Enhancements

Potential improvements:
- Machine learning-based detection
- Context-aware filtering
- Multi-language support
- Severity levels for different categories
- Custom replacement strategies per category

## Support

For issues or questions:
- Review the test files for usage examples
- Check logs for sanitization activity
- Extend keyword lists as needed for your use case

---

**Last Updated**: 2026-02-02
**Version**: 1.0.0
