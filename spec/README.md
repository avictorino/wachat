# Test Directory (`/spec`)

This directory contains all unit tests for the WaChat project.

## Structure

The test directory structure mirrors the source code structure:

```
spec/
├── __init__.py
├── core/              # Tests for core app
│   ├── __init__.py
│   ├── tests.py
│   ├── test_reset.py
│   └── test_simulate_command.py
└── services/          # Tests for services module
    ├── __init__.py
    ├── test_llm_factory.py
    ├── test_simulation_service.py
    ├── test_ollama_service.py
    ├── test_input_sanitizer.py
    ├── test_addiction_intents.py
    ├── test_message_splitter.py
    ├── test_prompt_composition.py
    └── test_groq_service_sanitization.py
```

## Running Tests

```bash
# Run all tests
python manage.py test

# Run tests with verbosity
python manage.py test --verbosity=2

# Run specific test module
python manage.py test spec.core
python manage.py test spec.services

# Run specific test file
python manage.py test spec.core.test_reset
python manage.py test spec.services.test_llm_factory
```

## Test File Naming Convention

- All test files must be named with the prefix `test_` (e.g., `test_models.py`, `test_views.py`)
- Or use the pattern `tests.py` for the main test file of a module
- This follows Django's test discovery pattern

## Enforcement

A pre-commit hook is configured to prevent test files from being created outside this directory:

- Test files matching the pattern `test*.py` are only allowed in `/spec`
- Attempting to commit test files in source directories (`core/`, `services/`, etc.) will fail
- This ensures all tests remain organized in a single location

## Writing Tests

Tests in this project use Django's `TestCase` class:

```python
from django.test import TestCase

class MyTestCase(TestCase):
    def test_something(self):
        # Your test code here
        self.assertEqual(1 + 1, 2)
```

For more information on writing Django tests, see:
https://docs.djangoproject.com/en/4.2/topics/testing/
