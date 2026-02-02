from django.contrib import admin

from core.models import Message, Profile


class MessageInline(admin.TabularInline):
    """Inline admin for displaying messages within Profile admin."""

    model = Message
    extra = 0
    readonly_fields = ["role", "content", "channel", "created_at"]
    can_delete = False
    fields = ["role", "channel", "content", "created_at"]
    ordering = ["-created_at"]

    def has_add_permission(self, request, obj=None):
        """Disable adding messages directly from inline."""
        return False


@admin.register(Profile)
class ProfileAdmin(admin.ModelAdmin):
    """Admin interface for Profile model."""

    list_display = [
        "id",
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
    inlines = [MessageInline]
