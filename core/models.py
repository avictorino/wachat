from django.db import models
from pgvector.django import VectorField


class ThemeRoleChoices(models.TextChoices):
    PERSON_SIMULATOR = "PERSON_SIMULATOR"
    BOT_SIMULATOR = "BOT_SIMULATOR"
    PERSON = "PERSON"
    BOT = "BOT"


class Theme(models.Model):
    name = models.CharField(max_length=100, primary_key=True)
    role = models.CharField(
        choices=ThemeRoleChoices.choices,
        default=ThemeRoleChoices.PERSON_SIMULATOR,
        max_length=255,
    )
    prompt = models.TextField(
        help_text="Thematic prompt text to apply to conversations"
    )


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
    theme = models.ForeignKey(
        Theme,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        help_text="Assigned conversation theme (temporary)",
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
        return f"{self.name} ({self.inferred_gender})"


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
        ("analysis", "Analysis"),
    ]

    CHANNEL_CHOICES = [
        ("telegram", "Telegram"),
        ("whatsapp", "WhatsApp"),
        ("simulation", "Simulation"),
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
    ollama_prompt = models.JSONField(
        null=True,
        blank=True,
        help_text="Full Ollama prompt payload sent to LLM (for observability)",
    )
    ollama_prompt_temperature = models.FloatField(
        null=True,
        blank=True,
        help_text="Temperature setting used in Ollama prompt (for observability)",
    )
    exclude_from_context = models.BooleanField(
        default=False,
        help_text="If True, exclude this message from RAG and memory context",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"{self.role}: {self.content[:50]}..."


class RagChunk(models.Model):
    """
    RAG (Retrieval-Augmented Generation) conversational chunk storage.

    Stores conversational chunks derived from source documents,
    optimized for human-like semantic retrieval.
    """

    TYPE_CHOICES = [
        ("conversation", "Conversation"),
        ("behavior", "Behavior"),
        ("content", "Content"),
    ]

    # Stable deterministic ID: <source>:p<page>:c<chunk_index>
    id = models.CharField(max_length=255, primary_key=True)

    # Source metadata
    source = models.CharField(
        max_length=255,
        db_index=True,
        help_text="Source document identifier (e.g., filename without extension)",
    )
    page = models.IntegerField(help_text="Page number in the source document")
    chunk_index = models.IntegerField(help_text="Index of the chunk within the page")

    # Original extracted text (raw book content)
    raw_text = models.TextField(
        help_text="Original text extracted from the source document"
    )

    # Structured conversational representation (RAG-first)
    conversations = models.JSONField(
        help_text="Structured conversation derived from the raw text.",
        default=list,
    )

    # Flattened conversational text (fallback, debugging, simple search)
    text = models.TextField(
        help_text="Flattened conversational text for fallback usage"
    )

    # Vector embedding based on conversational text
    embedding = VectorField(
        dimensions=768,
        help_text="Vector embedding derived from the conversational text",
    )

    # Chunk semantic type
    type = models.CharField(
        max_length=20,
        choices=TYPE_CHOICES,
        default="conversation",
        help_text="Semantic type of the chunk",
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["source", "page"], name="rag_source_page_idx"),
        ]
        ordering = ["source", "page", "chunk_index"]

    def __str__(self):
        return f"{self.source}:p{self.page}:c{self.chunk_index} ({self.type})"
