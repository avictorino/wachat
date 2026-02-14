from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0014_profile_conversation_mode_add_defensivo_culpa"),
    ]

    operations = [
        migrations.AlterField(
            model_name="message",
            name="ollama_prompt",
            field=models.JSONField(
                null=True,
                blank=True,
                help_text="Full Ollama prompt payload sent to LLM (for observability)",
            ),
        ),
    ]
