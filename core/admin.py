from django.contrib import admin
from django.db import transaction
from django.db.models import Count

from core.constants import ConversationMode
from core.models import (
    UserSpiritualProfile,
    VirtualFriend,
    Conversation,
    Message,
    FriendMemory,
    PrayerRequest,
    ReadingPlan,
)


# -----------------------------
# User Spiritual Profile
# -----------------------------

@admin.register(UserSpiritualProfile)
class UserSpiritualProfileAdmin(admin.ModelAdmin):
    list_display = (
        "user",
        "preferred_translation",
        "doctrine_profile",
        "created_at",
    )
    search_fields = ("user__email", "user__username")
    list_filter = ("doctrine_profile", "preferred_translation")
    readonly_fields = ("created_at", "updated_at")

    fieldsets = (
        ("Usuário", {"fields": ("user",)}),
        ("Preferências Espirituais", {
            "fields": (
                "preferred_translation",
                "doctrine_profile",
            )
        }),
        ("Onboarding e Restrições", {
            "classes": ("collapse",),
            "fields": (
                "extracted_profile",
                "allowed_topics",
                "blocked_topics",
            )
        }),
        ("Auditoria", {
            "classes": ("collapse",),
            "fields": ("created_at", "updated_at"),
        }),
    )


# -----------------------------
# Message Inline (Chat style)
# -----------------------------

class MessageInline(admin.TabularInline):
    model = Message
    extra = 0
    can_delete = True
    readonly_fields = ("role", "content", "created_at")
    ordering = ("created_at",)

    fields = ("role", "content", "created_at")

    def has_add_permission(self, request, obj=None):
        return False


# -----------------------------
# Conversation
# -----------------------------


@admin.action(description="🧹 Reset conversation (messages, profile, memories)")
def reset_conversation_state(modeladmin, request, queryset):
    """
    Deletes:
    - All messages in the conversation
    - All memories linked to the friend
    Resets:
    - extracted_profile of UserSpiritualProfile
    """

    with transaction.atomic():
        for conversation in queryset:
            friend = conversation.friend
            user = friend.owner

            # 1. Delete all messages from the conversation
            Message.objects.filter(conversation=conversation).delete()

            # 2. Reset extracted spiritual profile
            profile = UserSpiritualProfile.objects.filter(user=user).first()
            if profile:
                profile.extracted_profile = {}
                profile.save(update_fields=["extracted_profile", "updated_at"])

            # 3. Delete all memories related to the friend
            FriendMemory.objects.filter(friend=friend).delete()

            # 4. Optional: reset conversation mode (uncomment if desired)
            conversation.current_mode = ConversationMode.LISTENING.value
            conversation.save(update_fields=["current_mode"])

@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    actions = [reset_conversation_state]
    list_display = (
        "id",
        "friend",
        "short_title",
        "message_count",
        "is_closed",
        "created_at",
    )
    list_filter = ("is_closed", "created_at")
    search_fields = (
        "title",
        "friend__name",
        "friend__owner__email",
    )
    readonly_fields = ("created_at", "updated_at")
    inlines = [MessageInline]
    ordering = ("-created_at",)

    def short_title(self, obj):
        return obj.title[:50] if obj.title else "—"
    short_title.short_description = "Título"

    def message_count(self, obj):
        return obj.messages.count()
    message_count.short_description = "Mensagens"


# -----------------------------
# Virtual Friend
# -----------------------------

@admin.register(VirtualFriend)
class VirtualFriendAdmin(admin.ModelAdmin):
    list_display = (
        "name",
        "owner",
        "tone",
        "is_active",
        "conversation_count",
        "created_at",
    )
    list_filter = ("is_active", "tone", "created_at")
    search_fields = (
        "name",
        "owner__email",
        "owner__username",
    )
    readonly_fields = ("created_at", "updated_at")

    fieldsets = (
        ("Identidade do Amigo", {
            "fields": (
                "owner",
                "name",
                "persona",
                "tone",
                "is_active",
            )
        }),
        ("Parâmetros pessoais", {
            "classes": ("collapse",),
            "fields": (
                "age",
                "background",
            )
        }),
        ("Auditoria", {
            "classes": ("collapse",),
            "fields": (
                "created_at",
                "updated_at",
            )
        }),
    )

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        return qs.annotate(convo_count=Count("conversations"))

    def conversation_count(self, obj):
        return obj.convo_count
    conversation_count.short_description = "Conversas"


# -----------------------------
# Friend Memory
# -----------------------------

@admin.register(FriendMemory)
class FriendMemoryAdmin(admin.ModelAdmin):
    list_display = (
        "friend",
        "kind",
        "key",
        "short_value",
        "confidence",
        "is_active",
        "updated_at",
    )
    list_filter = ("kind", "is_active")
    search_fields = (
        "friend__name",
        "friend__owner__email",
        "key",
        "value",
    )
    readonly_fields = ("created_at", "updated_at")

    fieldsets = (
        ("Vínculo", {
            "fields": ("friend",)
        }),
        ("Conteúdo da Memória", {
            "fields": (
                "kind",
                "key",
                "value",
                "confidence",
                "is_active",
            )
        }),
        ("Origem", {
            "classes": ("collapse",),
            "fields": ("source",)
        }),
        ("Auditoria", {
            "classes": ("collapse",),
            "fields": ("created_at", "updated_at"),
        }),
    )

    def short_value(self, obj):
        return obj.value[:80] + "..." if len(obj.value) > 80 else obj.value
    short_value.short_description = "Valor"


# -----------------------------
# Prayer Request
# -----------------------------

@admin.register(PrayerRequest)
class PrayerRequestAdmin(admin.ModelAdmin):
    list_display = (
        "title",
        "friend",
        "is_answered",
        "created_at",
    )
    list_filter = ("is_answered", "created_at")
    search_fields = (
        "title",
        "details",
        "friend__name",
        "friend__owner__email",
    )
    readonly_fields = ("created_at", "updated_at")

    fieldsets = (
        ("Pedido de Oração", {
            "fields": (
                "friend",
                "title",
                "details",
                "tags",
            )
        }),
        ("Resposta", {
            "fields": (
                "is_answered",
                "answered_notes",
            )
        }),
        ("Auditoria", {
            "classes": ("collapse",),
            "fields": ("created_at", "updated_at"),
        }),
    )


# -----------------------------
# Reading Plan
# -----------------------------

@admin.register(ReadingPlan)
class ReadingPlanAdmin(admin.ModelAdmin):
    list_display = (
        "name",
        "friend",
        "days",
        "is_active",
        "created_at",
    )
    list_filter = ("is_active", "days")
    search_fields = (
        "name",
        "description",
        "friend__name",
        "friend__owner__email",
    )
    readonly_fields = ("created_at", "updated_at")

    fieldsets = (
        ("Plano de Leitura", {
            "fields": (
                "friend",
                "name",
                "description",
                "days",
                "is_active",
            )
        }),
        ("Estrutura do Plano", {
            "classes": ("collapse",),
            "fields": ("plan",)
        }),
        ("Auditoria", {
            "classes": ("collapse",),
            "fields": ("created_at", "updated_at"),
        }),
    )