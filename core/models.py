import json

from django.db import models
from pgvector.django import VectorField
from pgvector.django.indexes import HnswIndex


class ThemeV2(models.Model):
    id = models.CharField(max_length=50, primary_key=True)
    name = models.CharField(max_length=100)
    prompt = models.TextField(
        help_text="Thematic prompt text to apply to conversations"
    )

    class Meta:
        db_table = "theme_v2"
        ordering = ["name"]

    def __str__(self):
        return f"{self.name} ({self.id})"


class MessageManager(models.Manager):
    """Custom manager for Message model with context filtering."""

    def for_context(self):
        """
        Return messages that should be included in conversation context.

        Excludes:
        - System messages (role="system"): Internal/technical messages
        - Analysis messages (role="analysis"): Retrospective analysis reports
        - Messages with exclude_from_context=True: Any message explicitly marked for exclusion

        Note: Analysis messages are filtered both by role and by the exclude_from_context flag
        as a defense-in-depth measure. The flag allows for future flexibility if other message
        types need to be excluded from context without changing the role-based logic.
        """
        return (
            self.exclude(role="system")
            .exclude(role="analysis")
            .exclude(exclude_from_context=True)
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
        ThemeV2,
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

    welcome_message_sent = models.BooleanField(
        default=False,
        help_text="True if this message is a generated welcome message (for analytics)",
    )
    conversation_mode = models.CharField(
        max_length=20,
        choices=[
            ("WELCOME", "Welcome"),
            ("ACOLHIMENTO", "Acolhimento"),
            ("EXPLORACAO", "Exploracao"),
            ("AMBIVALENCIA", "Ambivalencia"),
            ("DEFENSIVO", "Defensivo"),
            ("CULPA", "Culpa"),
            ("ORIENTACAO", "Orientacao"),
        ],
        default="WELCOME",
        help_text="Current conversation mode used by response runtime state machine",
    )
    loop_detected_count = models.PositiveIntegerField(
        default=0,
        help_text="Total number of loop detections for this profile conversation",
    )
    regeneration_count = models.PositiveIntegerField(
        default=0,
        help_text="Total number of response regenerations for this profile conversation",
    )
    current_topic = models.CharField(
        max_length=120,
        blank=True,
        null=True,
        help_text="Currently active high-level conversation topic",
    )
    primary_topics = models.JSONField(
        default=list,
        blank=True,
        help_text="Persisted topic memory with score and recency metadata",
    )
    topic_last_updated = models.DateTimeField(
        blank=True,
        null=True,
        help_text="Timestamp of the latest topic memory update",
    )

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
        ("chat", "Chat Interface"),
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
    exclude_from_context = models.BooleanField(
        default=False,
        help_text="If True, exclude this message from RAG and memory context",
    )
    generated_by_simulator = models.BooleanField(
        default=False,
        help_text="True when this user message was generated by simulation runtime",
    )
    theme = models.ForeignKey(
        ThemeV2,
        on_delete=models.PROTECT,
        default="outros",
        db_index=True,
        db_column="theme",
        help_text="Classified predominant theme for this message",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    objects = MessageManager()

    class Meta:
        ordering = ["created_at"]

    @property
    def ollama_prompt_pretty_json(self) -> str:
        """
        Return ollama_prompt as a human-readable JSON string for UI rendering.

        Keeps backward compatibility if legacy rows still contain plain text.
        """
        if self.ollama_prompt is None:
            return ""
        if isinstance(self.ollama_prompt, (dict, list)):
            return json.dumps(self.ollama_prompt, indent=2, ensure_ascii=False)
        return str(self.ollama_prompt)

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
    theme = models.ForeignKey(
        ThemeV2,
        on_delete=models.PROTECT,
        default="outros",
        db_index=True,
        db_column="theme",
        help_text="Theme used to constrain RAG retrieval",
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["source", "page"], name="rag_source_page_idx"),
        ]
        ordering = ["source", "page", "chunk_index"]

    def __str__(self):
        return f"{self.source}:p{self.page}:c{self.chunk_index} ({self.type})"


class BibleTextFlat(models.Model):
    id = models.AutoField(primary_key=True)
    translation = models.CharField(max_length=50)
    testament = models.CharField(max_length=2)
    book = models.CharField(max_length=100)
    book_order = models.IntegerField()
    chapter = models.IntegerField()
    verse = models.IntegerField()
    reference = models.CharField(max_length=30)
    text = models.TextField()
    embedding = VectorField(dimensions=768)
    theme = models.ForeignKey(
        ThemeV2,
        on_delete=models.PROTECT,
        db_index=True,
        db_column="theme",
    )

    class Meta:
        indexes = [
            models.Index(fields=["book", "chapter", "verse"], name="bible_bcv_idx"),
            HnswIndex(
                name="bible_embedding_hnsw_idx",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_cosine_ops"],
            ),
        ]
