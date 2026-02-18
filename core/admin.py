import json

from django.contrib import admin, messages
from django.utils.html import format_html

from core.models import BibleTextFlat, Message, Profile, RagChunk, ThemeV2
from core.theme_prompt_generation import build_theme_prompt_partial


class MessageInline(admin.TabularInline):
    """
    Inline admin for displaying messages within Profile admin.

    Messages are read-only in the admin because they should only be created
    through the application logic (webhook handlers, conversation flows, etc.)
    to maintain data integrity and proper channel tracking.
    """

    model = Message
    extra = 0
    readonly_fields = ["role", "content", "channel", "theme", "created_at"]
    can_delete = False
    fields = ["role", "channel", "theme", "content", "created_at"]
    ordering = ["-created_at"]

    def has_add_permission(self, request, obj=None):
        """Disable adding messages directly from admin - they should be created via app logic."""
        return False


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    """Admin interface for Message model."""

    list_display = ["id", "profile", "role", "theme", "content_preview", "created_at"]
    list_filter = ["role", "channel", "theme", "created_at"]
    search_fields = ["content", "profile__name"]
    readonly_fields = ["created_at", "ollama_prompt_display"]
    ordering = ["-created_at"]

    fieldsets = (
        (
            "Message Info",
            {"fields": ("profile", "role", "content", "channel", "theme")},
        ),
        (
            "Ollama Prompt",
            {"fields": ("ollama_prompt_display",), "classes": ("collapse",)},
        ),
        ("Metadata", {"fields": ("created_at",)}),
    )

    def content_preview(self, obj):
        """Show truncated content in list view."""
        return obj.content[:50] + "..." if len(obj.content) > 50 else obj.content

    content_preview.short_description = "Content"

    def ollama_prompt_display(self, obj):
        """Display ollama_prompt as formatted JSON."""
        if obj.ollama_prompt:
            formatted_json = json.dumps(obj.ollama_prompt, indent=2, ensure_ascii=False)
            return format_html(
                """
                <pre style="
                    background-color: #0f172a;
                    color: #e5e7eb;
                    padding: 16px;
                    border-radius: 8px;
                    max-height: 600px;
                    overflow: auto;
                    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
                    font-size: 13px;
                    line-height: 1.6;
                    white-space: pre-wrap;
                    word-break: break-word;
                    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.08);
                ">
                {}</pre>
                """,
                formatted_json,
            )
        return "No prompt payload available"

    ollama_prompt_display.short_description = "Ollama Prompt Payload"

    def has_add_permission(self, request):
        """Disable adding messages directly from admin."""
        return False

    def has_delete_permission(self, request, obj=None):
        """Disable deleting messages from admin."""
        return True


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

    list_display = [
        "id",
        "source",
        "page",
        "chunk_index",
        "type",
        "theme",
        "created_at",
    ]
    list_filter = ["type", "theme", "source", "created_at"]
    search_fields = ["id", "source", "raw_text", "text"]
    readonly_fields = ["id", "created_at", "conversations_display"]
    ordering = ["source", "page", "chunk_index"]

    # Exclude vector fields from display
    exclude = ["embedding"]

    fieldsets = (
        (
            "Identification",
            {"fields": ("id", "source", "page", "chunk_index", "type", "theme")},
        ),
        ("Content", {"fields": ("raw_text", "conversations_display", "text")}),
        ("Metadata", {"fields": ("created_at",)}),
    )

    def conversations_display(self, obj):
        """Display conversations as formatted JSON."""
        if obj.conversations:
            formatted_json = json.dumps(obj.conversations, indent=2, ensure_ascii=False)
            return format_html(
                '<pre style="background: #f4f4f4; padding: 10px; overflow-x: auto; font-family: monospace;">{}</pre>',
                formatted_json,
            )
        return "No conversations available"

    conversations_display.short_description = "Conversations (Formatted)"


@admin.register(ThemeV2)
class ThemeV2Admin(admin.ModelAdmin):
    """Admin interface for ThemeV2 model."""

    list_display = ["id", "name", "prompt_short"]
    search_fields = ["id", "name", "prompt"]
    actions = ["regenerate_prompts_from_runtime"]

    def prompt_short(self, obj):
        """Show truncated prompt in list view."""
        return obj.prompt[:50] + "..." if len(obj.prompt) > 50 else obj.prompt

    @admin.action(description="Regenerar prompt dos temas selecionados (runtime)")
    def regenerate_prompts_from_runtime(self, request, queryset):
        updated_count = 0
        for theme in queryset:
            theme.prompt = build_theme_prompt_partial(
                theme_name=theme.name,
            )
            theme.save(update_fields=["prompt"])
            updated_count += 1
        self.message_user(
            request,
            f"{updated_count} tema(s) atualizados com prompt gerado.",
            level=messages.SUCCESS,
        )


@admin.register(BibleTextFlat)
class BibleTextFlatAdmin(admin.ModelAdmin):
    list_display = [
        "id",
        "translation",
        "testament",
        "book",
        "chapter",
        "verse",
        "reference",
        "theme",
    ]
    list_filter = ["translation", "testament", "book", "chapter", "theme"]
    search_fields = ["reference", "text", "book"]
    ordering = ["book_order", "chapter", "verse"]
    readonly_fields = ["embedding"]
