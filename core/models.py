from __future__ import annotations

import uuid
from django.conf import settings
from django.db import models
from django.utils import timezone

from core.constants import ConversationMode, Gender


class TimeStampedModel(models.Model):
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
    )
    created_at = models.DateTimeField(default=timezone.now, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class UserSpiritualProfile(TimeStampedModel):
    """
    Perfil espiritual do usuário, preferências e contexto que ajudam o Amigo.
    """
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="spiritual_profile")
    gender = models.CharField(max_length=10, choices=Gender.choices, null=True, blank=False)
    preferred_translation = models.CharField(max_length=32, default="NVI")  # ARA, NVI, NVT etc
    doctrine_profile = models.CharField(
        max_length=32,
        default="generic",
        choices=[
            ("generic", "Cristão genérico"),
            ("evangelical", "Evangélico"),
            ("catholic", "Católico"),
            ("custom", "Custom"),
        ],
    )
    onboarding_answers = models.JSONField(default=dict, blank=True)  # { "faith_level": "...", "church": "...", ... }
    allowed_topics = models.JSONField(default=list, blank=True)      # opcional: whitelist
    blocked_topics = models.JSONField(default=list, blank=True)      # opcional: blacklist
    extracted_profile = models.JSONField(default=dict)


class VirtualFriend(TimeStampedModel):
    """
    Um usuário pode ter vários amigos bíblicos, cada um com personalidade e parâmetros.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="virtual_friends")
    gender = models.CharField(max_length=10, choices=Gender.choices, null=True, blank=False)

    name = models.CharField(max_length=48, default="Amigo Bíblico")
    persona = models.TextField(default="Amigo bíblico, acolhedor, pastoral, baseado em Escrituras.")
    tone = models.CharField(max_length=32, default="carinhoso")  # carinhoso, direto, reflexivo, etc

    # Parametros que você pode ir completando com o tempo (igual você queria)
    age = models.PositiveSmallIntegerField(null=True, blank=True)
    background = models.JSONField(default=dict, blank=True)  # { "where_studied": "", "where_works": "", ... }

    is_active = models.BooleanField(default=True)

    class Meta:
        indexes = [
            models.Index(fields=["owner", "is_active", "created_at"]),
        ]

    def __str__(self):
        return f"{self.name} ({self.owner})"


class Conversation(TimeStampedModel):
    """
    Sessão de conversa, pode ser diária, tema, estudo bíblico etc.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    friend = models.ForeignKey(VirtualFriend, on_delete=models.CASCADE, related_name="conversations")
    title = models.CharField(max_length=120, blank=True, default="")
    context = models.JSONField(default=dict, blank=True)  # { "channel": "app", "topic": "...", ... }
    is_closed = models.BooleanField(default=False)
    current_mode = models.CharField(
        max_length=32,
        choices=[(m.value, m.value) for m in ConversationMode],
        default=ConversationMode.LISTENING.value,
    )

    def is_onboarding_phase(self) -> int:
        """
        Retorna quantas mensagens do ASSISTANT já existem na conversa.
        """
        return self.messages.filter(role=Message.Role.ASSISTANT).count()

    class Meta:
        indexes = [
            models.Index(fields=["friend", "created_at"]),
            models.Index(fields=["friend", "is_closed"]),
        ]


class Message(TimeStampedModel):
    class Role(models.TextChoices):
        USER = "user", "User"
        ASSISTANT = "assistant", "Assistant"
        SYSTEM = "system", "System"
        TOOL = "tool", "Tool"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name="messages")

    role = models.CharField(max_length=16, choices=Role.choices, db_index=True)
    content = models.TextField()
    metadata = models.JSONField(default=dict, blank=True)  # { "tokens": 123, "model": "...", "verses": [...] }

    class Meta:
        indexes = [
            models.Index(fields=["conversation", "created_at"]),
            models.Index(fields=["conversation", "role", "created_at"]),
        ]


class FriendMemory(TimeStampedModel):
    """
    Memória episódica e semântica do amigo, por usuário/amigo.
    Ex: "Usuário lida com ansiedade antes de dormir", "Verso favorito: Salmo 23".
    """
    class Kind(models.TextChoices):
        EPISODIC = "episodic", "Episodic"
        SEMANTIC = "semantic", "Semantic"
        PRAYER = "prayer", "Prayer"
        VERSE = "verse", "Verse"
        PLAN = "plan", "Plan"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    friend = models.ForeignKey(VirtualFriend, on_delete=models.CASCADE, related_name="memories")

    kind = models.CharField(max_length=16, choices=Kind.choices, db_index=True)
    key = models.CharField(max_length=128, db_index=True)  # "anxiety_bedtime", "favorite_verse", ...
    value = models.TextField()
    confidence = models.DecimalField(max_digits=3, decimal_places=2, default=0.80)  # 0.00 a 1.00
    source = models.JSONField(default=dict, blank=True)  # { "message_id": "...", "reason": "...", ... }

    is_active = models.BooleanField(default=True)

    class Meta:
        unique_together = [("friend", "kind", "key")]
        indexes = [
            models.Index(fields=["friend", "kind", "is_active"]),
            models.Index(fields=["friend", "key"]),
        ]


class PrayerRequest(TimeStampedModel):
    """
    Pedidos de oração, para o amigo retomar depois.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    friend = models.ForeignKey(VirtualFriend, on_delete=models.CASCADE, related_name="prayer_requests")

    title = models.CharField(max_length=120)
    details = models.TextField(blank=True, default="")
    is_answered = models.BooleanField(default=False)
    answered_notes = models.TextField(blank=True, default="")
    tags = models.JSONField(default=list, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=["friend", "is_answered", "created_at"]),
        ]


class ReadingPlan(TimeStampedModel):
    """
    Plano de leitura bíblica guiado pelo amigo.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    friend = models.ForeignKey(VirtualFriend, on_delete=models.CASCADE, related_name="reading_plans")

    name = models.CharField(max_length=120)
    description = models.TextField(blank=True, default="")
    days = models.PositiveSmallIntegerField(default=7)
    plan = models.JSONField(default=list, blank=True)
    # exemplo de plan:
    # [
    #   {"day": 1, "passages": ["Salmos 23"], "focus": "Confiança"},
    #   {"day": 2, "passages": ["Mateus 6:25-34"], "focus": "Ansiedade"},
    # ]

    is_active = models.BooleanField(default=True)

    class Meta:
        indexes = [
            models.Index(fields=["friend", "is_active", "created_at"]),
        ]