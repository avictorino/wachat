from django.db import migrations, models


def _forward_map_modes(apps, schema_editor):
    Profile = apps.get_model("core", "Profile")
    mode_map = {
        "acolhimento": "ACOLHIMENTO",
        "exploração": "EXPLORACAO",
        "orientação": "ORIENTACAO",
    }
    for profile in Profile.objects.all().only("id", "conversation_mode"):
        new_mode = mode_map.get(profile.conversation_mode, profile.conversation_mode)
        if new_mode not in {
            "WELCOME",
            "ACOLHIMENTO",
            "EXPLORACAO",
            "AMBIVALENCIA",
            "ORIENTACAO",
        }:
            new_mode = "WELCOME"
        if profile.conversation_mode != new_mode:
            profile.conversation_mode = new_mode
            profile.save(update_fields=["conversation_mode"])


def _noop_reverse(apps, schema_editor):
    pass


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0010_profile_conversation_runtime_state"),
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
                    ("ORIENTACAO", "Orientacao"),
                ],
                default="WELCOME",
                help_text="Current conversation mode used by response runtime state machine",
                max_length=20,
            ),
        ),
        migrations.RunPython(_forward_map_modes, _noop_reverse),
    ]
