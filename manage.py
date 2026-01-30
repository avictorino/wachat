#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import warnings

# Suppress pkg_resources deprecation warnings from PyCharm debugger
# These warnings are triggered by PyCharm's pydevd plugins using deprecated pkg_resources APIs
# We filter by message content to target only pkg_resources-related deprecation warnings
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=".*pkg_resources.*"
)
warnings.filterwarnings(
    "ignore", 
    category=DeprecationWarning,
    message=".*declare_namespace.*"
)

# Load environment variables from .env file
import dotenv


def main():
    """Run administrative tasks."""
    dotenv.read_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
    
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
