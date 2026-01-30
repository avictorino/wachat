"""
ASGI config for wachat project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/asgi/
"""

import os
import dotenv

from django.core.asgi import get_asgi_application

dotenv.read_dotenv()
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "wachat.settings")

application = get_asgi_application()
