from django.db import models
from pgvector.django import VectorField


class Profile(models.Model):
    """
    User profile for storing user information across multiple channels.

    This model represents the long-term identity of a user interacting
    with the bot through various channels (Telegram, WhatsApp, etc.).
    """

    telegram_user_id = models.CharField(
        max_length=100,
        blank=True,
        null=True,
        unique=True,  # Keep unique constraint when not null
        db_index=True,
        help_text="Telegram user ID (unique identifier from Telegram)",
    )
    name = models.CharField(
        max_length=255, help_text="User's name from Telegram (first name or full name)"
    )
    phone_number = models.CharField(
        max_length=50,
        blank=True,
        null=True,
        help_text="User's phone number (if shared)",
    )
    inferred_gender = models.CharField(
        max_length=20,
        blank=True,
        null=True,
        help_text="Gender inferred from name (male/female/unknown)",
    )
    prompt_theme = models.CharField(
        max_length=100,
        blank=True,
        null=True,
        help_text="Active thematic prompt id to apply to this user's conversation",
    )
    pending_reset_confirmation = models.BooleanField(
        default=False,
        help_text="True if user has initiated /reset and is awaiting confirmation",
    )
    reset_confirmation_timestamp = models.DateTimeField(
        blank=True,
        null=True,
        help_text="Timestamp when /reset was initiated (expires after timeout)",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.name} ({self.telegram_user_id})"


class Message(models.Model):
    """
    Conversation message linked to a user profile.

    Stores messages from both the system/assistant and the user
    to maintain conversation context.
    """

    ROLE_CHOICES = [
        ("system", "System"),
        ("assistant", "Assistant"),
        ("user", "User"),
    ]

    CHANNEL_CHOICES = [
        ("telegram", "Telegram"),
        ("whatsapp", "WhatsApp"),
        ("other", "Other"),
    ]

    profile = models.ForeignKey(
        Profile,
        on_delete=models.CASCADE,
        related_name="messages",
        help_text="User profile this message belongs to",
    )
    role = models.CharField(
        max_length=20, choices=ROLE_CHOICES, help_text="Role of the message sender"
    )
    content = models.TextField(help_text="Message text content")
    channel = models.CharField(
        max_length=20,
        choices=CHANNEL_CHOICES,
        default="telegram",
        help_text="Channel through which the message was sent",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"{self.role}: {self.content[:50]}..."


class RagChunk(models.Model):
    """
    RAG (Retrieval-Augmented Generation) chunk storage.

    Stores text chunks extracted from PDFs with their embeddings
    for semantic search during response generation.
    """

    TYPE_CHOICES = [
        ("behavior", "Behavior"),
        ("content", "Content"),
    ]

    id = models.CharField(max_length=255, primary_key=True)
    source = models.CharField(
        max_length=255,
        db_index=True,
        help_text="Source document identifier (e.g., filename without extension)",
    )
    page = models.IntegerField(help_text="Page number in the source document")
    chunk_index = models.IntegerField(
        help_text="Index of the chunk within the page"
    )
    text = models.TextField(help_text="The actual text content of the chunk")
    embedding = VectorField(
        dimensions=768,
        help_text="Vector embedding of the text (768-dimensional for nomic-embed-text)",
    )
    type = models.CharField(
        max_length=20,
        choices=TYPE_CHOICES,
        default="content",
        help_text="Type of chunk: behavior (guidance/posture) or content (informational)",
    )

    class Meta:
        indexes = [
            models.Index(fields=["source", "page"], name="rag_source_page_idx"),
        ]
        ordering = ["source", "page", "chunk_index"]

    def __str__(self):
        return f"{self.source}:p{self.page}:c{self.chunk_index} ({self.type})"
