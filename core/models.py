import json

from django.db import models


class Theme(models.Model):
    id = models.AutoField(primary_key=True)
    slug = models.CharField(
        max_length=80,
        blank=True,
        null=True,
        unique=True,
        db_index=True,
    )
    name = models.CharField(max_length=100)
    meta_prompt = models.TextField(
        help_text="Thematic prompt text to apply to conversations"
    )
    score = models.FloatField(null=True, blank=True)
    improvement = models.TextField(null=True, blank=True)

    class Meta:
        db_table = "theme"
        ordering = ["name"]

    def __str__(self):
        if self.slug:
            return f"{self.name} ({self.slug})"
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
    last_simulation_report = models.TextField(
        blank=True,
        null=True,
        help_text="Last full conversation simulation report generated for this profile",
    )
    simulated_behavior = models.JSONField(
        blank=True,
        null=True,
        help_text="Latest simulated user behavior controls and generation metadata",
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
    block_root = models.ForeignKey(
        "self",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="block_messages",
        help_text="First message of the regeneration block this message belongs to",
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
        help_text="If True, exclude this message from memory context",
    )
    generated_by_simulator = models.BooleanField(
        default=False,
        help_text="True when this user message was generated by simulation runtime",
    )
    score = models.FloatField(
        null=True,
        blank=True,
        db_index=True,
        help_text="Evaluation score for assistant messages (0-10)",
    )
    bot_mode = models.CharField(
        max_length=40,
        blank=True,
        null=True,
        db_index=True,
        help_text="Conversation runtime mode used to generate assistant message",
    )
    theme = models.ForeignKey(
        Theme,
        on_delete=models.PROTECT,
        null=True,
        blank=True,
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


class SocialMediaExport(models.Model):
    STATUS_PENDING = "pending"
    STATUS_LIKED = "liked"
    STATUS_APPROVED = "approved"
    STATUS_DOESNT_LIKE = "doesnt_like"

    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_LIKED, "Liked"),
        (STATUS_APPROVED, "Approved"),
        (STATUS_DOESNT_LIKE, "Doesn't like"),
    ]

    original_message = models.OneToOneField(
        Message,
        on_delete=models.CASCADE,
        related_name="social_media_export",
        help_text="Assistant message used as source for this social media export",
    )
    original_text = models.TextField(
        help_text="Original source excerpt (user question + assistant answer)"
    )
    adapted_text = models.TextField(
        help_text="Adapted social media text generated by LLM"
    )
    image_summary = models.TextField(
        help_text="Short summary intended to be fixed on top of the image"
    )
    religous_reference = models.TextField(
        blank=True,
        null=True,
        help_text="Related biblical passage or famous sermon excerpt",
    )
    generated_image = models.ImageField(
        upload_to="social_media_exports/",
        blank=True,
        null=True,
        help_text="Generated social media image file",
    )
    image_generation_usage = models.JSONField(
        blank=True,
        null=True,
        help_text="OpenAI image generation usage payload with estimated USD cost",
    )
    score = models.FloatField(
        null=True,
        blank=True,
        help_text="Relevance score for social media usage (0-10)",
    )
    is_religious = models.BooleanField(
        default=False,
        help_text="True when excerpt has explicit religious context",
    )
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_PENDING,
        db_index=True,
    )

    class Meta:
        db_table = "socialmediaexport"
        ordering = ["-id"]

    def __str__(self):
        return f"Export {self.id} | message={self.original_message_id} | {self.status}"
