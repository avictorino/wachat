from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0011_update_profile_conversation_mode_states"),
    ]

    operations = [
        migrations.AddField(
            model_name="profile",
            name="current_topic",
            field=models.CharField(
                blank=True,
                help_text="Currently active high-level conversation topic",
                max_length=120,
                null=True,
            ),
        ),
        migrations.AddField(
            model_name="profile",
            name="primary_topics",
            field=models.JSONField(
                blank=True,
                default=list,
                help_text="Persisted topic memory with score and recency metadata",
            ),
        ),
        migrations.AddField(
            model_name="profile",
            name="topic_last_updated",
            field=models.DateTimeField(
                blank=True,
                help_text="Timestamp of the latest topic memory update",
                null=True,
            ),
        ),
    ]
