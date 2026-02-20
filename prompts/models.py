from django.db import models
from django.db.models import Q

from core.models import Theme


class PromptComponent(models.Model):
    COMPONENT_SCOPE_CHOICES = [
        ("global", "Global"),
        ("mode", "Mode"),
        ("theme", "Theme"),
        ("custom", "Custom"),
    ]

    COMPONENT_TYPE_CHOICES = [
        ("system", "System"),
        ("runtime", "Runtime"),
        ("theme_meta", "Theme Meta"),
        ("evaluation", "Evaluation"),
        ("welcome", "Welcome"),
        ("topic", "Topic"),
        ("other", "Other"),
    ]

    key = models.CharField(
        max_length=120,
        unique=True,
        db_index=True,
        help_text="Stable unique component key, ex: runtime.mode.WELCOME",
    )
    component_type = models.CharField(
        max_length=20,
        choices=COMPONENT_TYPE_CHOICES,
        db_index=True,
    )
    scope = models.CharField(max_length=20, choices=COMPONENT_SCOPE_CHOICES)
    mode = models.CharField(
        max_length=40,
        blank=True,
        null=True,
        db_index=True,
        help_text="Conversation mode when scope=mode",
    )
    theme = models.ForeignKey(
        Theme,
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="prompt_components",
        help_text="Theme when scope=theme",
    )
    name = models.CharField(max_length=140)
    description = models.TextField(
        blank=True,
        help_text="Short summary describing when/why this component is used",
    )
    active_version = models.PositiveIntegerField(
        blank=True,
        null=True,
        help_text="Fast pointer to currently active version number",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "prompt_component"
        ordering = ["key"]

    def __str__(self):
        return self.key


class PromptComponentVersion(models.Model):
    STATUS_CHOICES = [
        ("draft", "Draft"),
        ("suggested", "Suggested"),
        ("approved", "Approved"),
        ("active", "Active"),
        ("rejected", "Rejected"),
        ("archived", "Archived"),
    ]

    component = models.ForeignKey(
        PromptComponent,
        on_delete=models.CASCADE,
        related_name="versions",
    )
    version = models.PositiveIntegerField()
    content = models.TextField(help_text="Prompt content")
    description = models.TextField(
        help_text="Summary for what this prompt version is for"
    )
    score = models.FloatField(
        blank=True,
        null=True,
        db_index=True,
        help_text="Observed quality score for this version (0-10)",
    )
    improvement = models.TextField(
        blank=True,
        help_text="Short recommendation to improve next generation round",
    )
    score_details = models.JSONField(
        default=dict,
        blank=True,
        help_text="Optional score breakdown payload",
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, db_index=True)
    change_summary = models.TextField(blank=True)
    created_by = models.CharField(max_length=60, default="system")
    parent = models.ForeignKey(
        "self",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="children",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "prompt_component_version"
        ordering = ["component_id", "version"]
        unique_together = [("component", "version")]
        constraints = [
            models.UniqueConstraint(
                fields=["component"],
                condition=Q(status="active"),
                name="uniq_active_prompt_component_version",
            ),
        ]

    def __str__(self):
        return f"{self.component.key}@v{self.version}"
