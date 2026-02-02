from django.db import models


class Profile(models.Model):
    """
    User profile for storing Telegram user information.
    
    This model represents the long-term identity of a user interacting
    with the bot through Telegram.
    """
    telegram_user_id = models.CharField(
        max_length=100,
        unique=True,
        db_index=True,
        help_text="Telegram user ID (unique identifier from Telegram)"
    )
    name = models.CharField(
        max_length=255,
        help_text="User's name from Telegram (first name or full name)"
    )
    phone_number = models.CharField(
        max_length=50,
        blank=True,
        null=True,
        help_text="User's phone number (if shared)"
    )
    inferred_gender = models.CharField(
        max_length=20,
        blank=True,
        null=True,
        help_text="Gender inferred from name (male/female/unknown)"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} ({self.telegram_user_id})"


class Message(models.Model):
    """
    Conversation message linked to a user profile.
    
    Stores messages from both the system/assistant and the user
    to maintain conversation context.
    """
    ROLE_CHOICES = [
        ('system', 'System'),
        ('assistant', 'Assistant'),
        ('user', 'User'),
    ]
    
    profile = models.ForeignKey(
        Profile,
        on_delete=models.CASCADE,
        related_name='messages',
        help_text="User profile this message belongs to"
    )
    role = models.CharField(
        max_length=20,
        choices=ROLE_CHOICES,
        help_text="Role of the message sender"
    )
    content = models.TextField(
        help_text="Message text content"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"{self.role}: {self.content[:50]}..."
