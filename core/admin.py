from django.contrib import admin

from core.models import Message, Profile, RagChunk


class MessageInline(admin.TabularInline):
    """
    Inline admin for displaying messages within Profile admin.

    Messages are read-only in the admin because they should only be created
    through the application logic (webhook handlers, conversation flows, etc.)
    to maintain data integrity and proper channel tracking.
    """

    model = Message
    extra = 0
    readonly_fields = ["role", "content", "channel", "created_at"]
    can_delete = False
    fields = ["role", "channel", "content", "created_at"]
    ordering = ["-created_at"]

    def has_add_permission(self, request, obj=None):
        """Disable adding messages directly from admin - they should be created via app logic."""
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


@admin.register(RagChunk)
class RagChunkAdmin(admin.ModelAdmin):
    """Admin interface for RagChunk model."""

    list_display = ["id", "source", "page", "chunk_index", "type", "created_at"]
    list_filter = ["type", "source", "created_at"]
    search_fields = ["id", "source", "raw_text", "text"]
    readonly_fields = ["id", "created_at"]
    ordering = ["source", "page", "chunk_index"]

    # Exclude vector fields from display
    exclude = ["embedding"]

    fieldsets = (
        ("Identification", {"fields": ("id", "source", "page", "chunk_index", "type")}),
        ("Content", {"fields": ("raw_text", "conversations", "text")}),
        ("Metadata", {"fields": ("created_at",)}),
    )
