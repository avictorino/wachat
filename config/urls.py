"""
URL configuration for config project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path

from core.views import (
    FacebookWhatsAppWebhookView,
    TelegramWebhookView,
    data_deletion_view,
    privacy_policy_view,
    terms_of_service_view,
)

urlpatterns = [
    path("api/webhooks/whatsapp-facebook/", FacebookWhatsAppWebhookView.as_view()),
    path("webhooks/telegram/", TelegramWebhookView.as_view()),
    path("admin/", admin.site.urls),
    path("privacidade/", privacy_policy_view, name="privacy_policy"),
    path("termos/", terms_of_service_view, name="terms_of_service"),
    path("data-deletion/", data_deletion_view, name="data_deletion"),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
