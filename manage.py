#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys


def main():
    """Run administrative tasks."""
    # Only load .env file in non-production environments
    # In production (Heroku), use Config Vars instead
    if os.getenv("DYNO") is None:
        # Not on Heroku, load .env file for local development
        try:
            import dotenv
            dotenv.read_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
        except ImportError:
            # django-dotenv not installed (e.g., in production)
            pass

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
