from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0026_message_score"),
    ]

    operations = [
        migrations.AddField(
            model_name="profile",
            name="last_simulation_report",
            field=models.TextField(
                blank=True,
                help_text="Last full conversation simulation report generated for this profile",
                null=True,
            ),
        ),
    ]
