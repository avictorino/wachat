from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0009_merge_20260212_0145"),
    ]

    operations = [
        migrations.AddField(
            model_name="profile",
            name="conversation_mode",
            field=models.CharField(
                choices=[
                    ("acolhimento", "Acolhimento"),
                    ("exploração", "Exploração"),
                    ("orientação", "Orientação"),
                ],
                default="acolhimento",
                help_text="Current conversation mode used by response runtime state machine",
                max_length=20,
            ),
        ),
        migrations.AddField(
            model_name="profile",
            name="loop_detected_count",
            field=models.PositiveIntegerField(
                default=0,
                help_text="Total number of loop detections for this profile conversation",
            ),
        ),
        migrations.AddField(
            model_name="profile",
            name="regeneration_count",
            field=models.PositiveIntegerField(
                default=0,
                help_text="Total number of response regenerations for this profile conversation",
            ),
        ),
    ]
