from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0013_message_generated_by_simulator"),
    ]

    operations = [
        migrations.AlterField(
            model_name="profile",
            name="conversation_mode",
            field=models.CharField(
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
                max_length=20,
            ),
        ),
    ]
