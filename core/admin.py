from django.contrib import admin

from core.models import Message, Profile


@admin.register(Profile)
class ProfileAdmin(admin.ModelAdmin):
    """Admin interface for Profile model."""

    list_display = [
        "telegram_user_id",
        "name",
        "phone_number",
        "inferred_gender",
        "created_at",
    ]
    list_filter = ["inferred_gender", "created_at"]
    search_fields = ["telegram_user_id", "name", "phone_number"]
    readonly_fields = ["created_at", "updated_at"]
    ordering = ["-created_at"]


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    """Admin interface for Message model."""

    list_display = ["id", "profile", "role", "content_preview", "created_at"]
    list_filter = ["role", "created_at"]
    search_fields = ["content", "profile__name", "profile__telegram_user_id"]
    readonly_fields = ["created_at"]
    ordering = ["-created_at"]

    def content_preview(self, obj):
        """Show a preview of the message content."""
        return obj.content[:50] + "..." if len(obj.content) > 50 else obj.content

    content_preview.short_description = "Content"
