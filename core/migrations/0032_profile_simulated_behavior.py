from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0031_message_bot_mode"),
    ]

    operations = [
        migrations.AddField(
            model_name="profile",
            name="simulated_behavior",
            field=models.JSONField(
                blank=True,
                help_text="Latest simulated user behavior controls and generation metadata",
                null=True,
            ),
        ),
    ]
