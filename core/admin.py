import json

from django.contrib import admin, messages
from django.db.models import Avg, Count
from django.utils.html import format_html

from core.models import Message, Profile, SocialMediaExport, Theme
from core.theme_prompt_generation import build_theme_prompt_partial
from services.social_media_export_service import SocialMediaExportService


class MessageInline(admin.TabularInline):
    """
    Inline admin for displaying messages within Profile admin.

    Messages are read-only in the admin because they should only be created
    through the application logic (webhook handlers, conversation flows, etc.)
    to maintain data integrity and proper channel tracking.
    """

    model = Message
    extra = 0
    readonly_fields = ["role", "content", "channel", "theme", "score", "created_at"]
    can_delete = False
    fields = ["role", "channel", "theme", "score", "content", "created_at"]
    ordering = ["-created_at"]

    def has_add_permission(self, request, obj=None):
        """Disable adding messages directly from admin - they should be created via app logic."""
        return False


class MessageScoreBandFilter(admin.SimpleListFilter):
    title = "score range"
    parameter_name = "score_band"

    def lookups(self, request, model_admin):
        return (
            ("bad", "Ruins (<= 6.0)"),
            ("medium", "Médias (> 6.0 e < 8.0)"),
            ("good", "Boas (>= 8.0)"),
            ("empty", "Sem score"),
        )

    def queryset(self, request, queryset):
        value = self.value()
        if value == "bad":
            return queryset.filter(score__lte=6.0)
        if value == "medium":
            return queryset.filter(score__gt=6.0, score__lt=8.0)
        if value == "good":
            return queryset.filter(score__gte=8.0)
        if value == "empty":
            return queryset.filter(score__isnull=True)
        return queryset


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    """Admin interface for Message model."""

    list_display = [
        "id",
        "profile",
        "role",
        "score",
        "theme",
        "content_preview",
        "created_at",
    ]
    list_filter = ["role", "channel", "theme", MessageScoreBandFilter, "created_at"]
    search_fields = ["content", "profile__name"]
    readonly_fields = ["created_at", "score", "ollama_prompt_display"]
    ordering = ["-created_at"]

    fieldsets = (
        (
            "Message Info",
            {"fields": ("profile", "role", "content", "channel", "theme", "score")},
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
        "name",
        "messages_count",
        "messages_score_avg",
    ]
    list_filter = []
    search_fields = ["telegram_user_id", "name", "phone_number"]
    readonly_fields = ["created_at", "updated_at"]
    ordering = ["-created_at"]
    inlines = [MessageInline]
    actions = ["export_social_media_snippets"]

    def get_queryset(self, request):
        queryset = super().get_queryset(request)
        return queryset.annotate(
            messages_total=Count("messages"),
            messages_score_avg_value=Avg("messages__score"),
        )

    def messages_count(self, obj):
        return obj.messages_total

    messages_count.short_description = "Messages"
    messages_count.admin_order_field = "messages_total"

    def messages_score_avg(self, obj):
        if obj.messages_score_avg_value is None:
            return "-"
        return f"{obj.messages_score_avg_value:.2f}"

    messages_score_avg.short_description = "Avg score"
    messages_score_avg.admin_order_field = "messages_score_avg_value"

    @admin.action(description="Exportar trechos para social media")
    def export_social_media_snippets(self, request, queryset):
        service = SocialMediaExportService()
        exported_count = 0

        for profile in queryset:
            exported_count += service.export_profile_messages(profile=profile)

        self.message_user(
            request,
            f"{exported_count} trecho(s) exportado(s) para social media.",
            level=messages.SUCCESS,
        )


@admin.register(Theme)
class ThemeAdmin(admin.ModelAdmin):
    """Admin interface for Theme model."""

    list_display = ["id", "slug", "name", "score"]
    search_fields = ["id", "slug", "name", "meta_prompt"]
    readonly_fields = ["id", "score", "improvement"]
    fields = ["id", "slug", "name", "meta_prompt", "score", "improvement"]
    actions = ["regenerate_prompts_from_runtime"]

    @admin.action(description="Regenerar prompt dos temas selecionados (runtime)")
    def regenerate_prompts_from_runtime(self, request, queryset):
        updated_count = 0
        for theme in queryset:
            build_theme_prompt_partial(theme=theme)
            updated_count += 1
        self.message_user(
            request,
            f"{updated_count} tema(s) atualizados com prompt gerado.",
            level=messages.SUCCESS,
        )


@admin.register(SocialMediaExport)
class SocialMediaExportAdmin(admin.ModelAdmin):
    list_display = [
        "id",
        "status",
        "score",
        "image_price_usd",
        "image_thumb",
        "adapted_text_full",
        "image_reference_full",
        "religous_reference_full",
    ]
    list_filter = []
    search_fields = [
        "original_text",
        "adapted_text",
        "image_summary",
        "religous_reference",
        "original_message__content",
        "original_message__profile__name",
    ]
    readonly_fields = [
        "original_text",
        "adapted_text",
        "image_summary",
        "religous_reference",
        "generated_image",
        "image_preview",
        "image_generation_usage_pretty",
    ]
    exclude = ["original_message"]
    ordering = ["-id"]
    actions = ["generate_selected_images"]

    @admin.action(description="Gerar imagem social dos itens selecionados")
    def generate_selected_images(self, request, queryset):
        service = SocialMediaExportService()
        generated_count = 0

        for item in queryset:
            service.generate_image_for_export(export_item=item)
            generated_count += 1

        self.message_user(
            request,
            f"{generated_count} imagem(ns) gerada(s) com sucesso.",
            level=messages.SUCCESS,
        )

    def image_thumb(self, obj):
        if not obj.generated_image:
            return "-"
        return format_html(
            '<a href="{0}" target="_blank" rel="noopener noreferrer">'
            '<img src="{0}" style="width: 120px; height: auto; object-fit: contain; border-radius: 6px;" />'
            "</a>",
            obj.generated_image.url,
        )

    image_thumb.short_description = "Image"

    def image_price_usd(self, obj):
        usage_payload = obj.image_generation_usage or {}
        pricing = usage_payload.get("pricing") or {}
        estimated_cost = pricing.get("estimated_cost_usd")
        if isinstance(estimated_cost, (int, float)):
            return f"${estimated_cost:.3f}"

        estimated_range = pricing.get("estimated_range_usd") or {}
        min_cost = estimated_range.get("min")
        max_cost = estimated_range.get("max")
        if isinstance(min_cost, (int, float)) and isinstance(max_cost, (int, float)):
            return f"${min_cost:.3f} - ${max_cost:.3f}"

        return "-"

    image_price_usd.short_description = "Price (USD)"

    def adapted_text_full(self, obj):
        return format_html(
            '<div style="white-space: pre-wrap; min-width: 420px;">{}</div>',
            obj.adapted_text,
        )

    adapted_text_full.short_description = "Adapted text"

    def image_reference_full(self, obj):
        return format_html(
            '<div style="white-space: pre-wrap; min-width: 320px;">{}</div>',
            obj.image_summary,
        )

    image_reference_full.short_description = "Image reference"

    def religous_reference_full(self, obj):
        return format_html(
            '<div style="white-space: pre-wrap; min-width: 320px;">{}</div>',
            obj.religous_reference or "",
        )

    religous_reference_full.short_description = "Religious reference"

    def image_preview(self, obj):
        if not obj.generated_image:
            return "No generated image"
        return format_html(
            '<a href="{0}" target="_blank" rel="noopener noreferrer">'
            '<img src="{0}" style="max-width: 280px; height: auto; border-radius: 8px;" />'
            "</a>",
            obj.generated_image.url,
        )

    image_preview.short_description = "Image preview"

    def image_generation_usage_pretty(self, obj):
        if not obj.image_generation_usage:
            return "No usage payload"
        formatted_json = json.dumps(
            obj.image_generation_usage, indent=2, ensure_ascii=False
        )
        return format_html(
            '<pre style="white-space: pre-wrap; min-width: 420px;">{}</pre>',
            formatted_json,
        )

    image_generation_usage_pretty.short_description = "Image usage / cost (USD)"
