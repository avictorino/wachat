# Generated migration to remove detected_intent field

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0006_profile_prompt_theme"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="profile",
            name="detected_intent",
        ),
    ]
