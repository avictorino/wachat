"""
WSGI config for config project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/wsgi/
"""

import os

# Only load .env file in non-production environments
# In production (Heroku), use Config Vars instead
if os.getenv("DYNO") is None:
    # Not on Heroku, load .env file for local development
    try:
        import dotenv
        dotenv.read_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
    except ImportError:
        # django-dotenv not installed (e.g., in production)
        pass

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

application = get_wsgi_application()
