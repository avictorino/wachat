from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("prompts", "0001_initial"),
        ("core", "0029_promptcomponent_promptcomponentversion_promptrelease_and_more"),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            database_operations=[],
            state_operations=[
                migrations.DeleteModel(name="PromptReleaseItem"),
                migrations.DeleteModel(name="PromptVersionProposal"),
                migrations.DeleteModel(name="PromptRelease"),
                migrations.DeleteModel(name="PromptComponentVersion"),
                migrations.DeleteModel(name="PromptComponent"),
            ],
        )
    ]
