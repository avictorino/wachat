import django.db.models.deletion
from django.db import migrations, models

SQL_FORWARD = """
DO $$
DECLARE
    missing_count BIGINT;
BEGIN
    ALTER TABLE theme_v2 ADD COLUMN slug VARCHAR(80);
    UPDATE theme_v2 SET slug = id;
    UPDATE theme_v2
    SET slug = 'dinheiro_e_dividas'
    WHERE id = 'financeiro'
      AND NOT EXISTS (SELECT 1 FROM theme_v2 t2 WHERE t2.id = 'dinheiro_e_dividas');
    UPDATE theme_v2
    SET slug = 'vicios_e_recaidas'
    WHERE id = 'vicios'
      AND NOT EXISTS (SELECT 1 FROM theme_v2 t2 WHERE t2.id = 'vicios_e_recaidas');
    UPDATE theme_v2
    SET slug = 'saude_e_cansaco'
    WHERE id = 'saude'
      AND NOT EXISTS (SELECT 1 FROM theme_v2 t2 WHERE t2.id = 'saude_e_cansaco');
    UPDATE theme_v2
    SET slug = 'luto_e_perda'
    WHERE id = 'luto_perda'
      AND NOT EXISTS (SELECT 1 FROM theme_v2 t2 WHERE t2.id = 'luto_e_perda');
    UPDATE theme_v2
    SET slug = 'trabalho_e_pressao'
    WHERE id = 'trabalho'
      AND NOT EXISTS (SELECT 1 FROM theme_v2 t2 WHERE t2.id = 'trabalho_e_pressao');
    UPDATE theme_v2
    SET slug = 'espiritualidade'
    WHERE id = 'espiritual'
      AND NOT EXISTS (SELECT 1 FROM theme_v2 t2 WHERE t2.id = 'espiritualidade');

    ALTER TABLE theme_v2 ADD COLUMN int_id INTEGER;
    UPDATE theme_v2
    SET int_id = CASE slug
        WHEN 'relacionamento' THEN 1
        WHEN 'dinheiro_e_dividas' THEN 2
        WHEN 'vicios_e_recaidas' THEN 3
        WHEN 'saude_e_cansaco' THEN 4
        WHEN 'luto_e_perda' THEN 5
        WHEN 'trabalho_e_pressao' THEN 6
        WHEN 'solidao' THEN 7
        WHEN 'espiritualidade' THEN 8
        WHEN 'ansiedade' THEN 9
        WHEN 'outros' THEN 10
        ELSE NULL
    END;

    CREATE SEQUENCE IF NOT EXISTS theme_v2_int_fallback_seq START WITH 1000 INCREMENT BY 1;
    UPDATE theme_v2
    SET int_id = nextval('theme_v2_int_fallback_seq')
    WHERE int_id IS NULL;
    ALTER TABLE theme_v2 ALTER COLUMN int_id SET NOT NULL;

    SELECT COUNT(*)
    INTO missing_count
    FROM core_message m
    LEFT JOIN theme_v2 t ON m.theme = t.id
    WHERE m.theme IS NOT NULL
      AND t.id IS NULL;
    IF missing_count > 0 THEN
        RAISE EXCEPTION 'core_message has % rows with unknown theme ids', missing_count;
    END IF;

    SELECT COUNT(*)
    INTO missing_count
    FROM core_ragchunk r
    LEFT JOIN theme_v2 t ON r.theme = t.id
    WHERE r.theme IS NOT NULL
      AND t.id IS NULL;
    IF missing_count > 0 THEN
        RAISE EXCEPTION 'core_ragchunk has % rows with unknown theme ids', missing_count;
    END IF;

    SELECT COUNT(*)
    INTO missing_count
    FROM core_bibletextflat b
    LEFT JOIN theme_v2 t ON b.theme = t.id
    WHERE b.theme IS NOT NULL
      AND t.id IS NULL;
    IF missing_count > 0 THEN
        RAISE EXCEPTION 'core_bibletextflat has % rows with unknown theme ids', missing_count;
    END IF;

    SELECT COUNT(*)
    INTO missing_count
    FROM core_profile p
    LEFT JOIN theme_v2 t ON p.theme_id = t.id
    WHERE p.theme_id IS NOT NULL
      AND t.id IS NULL;
    IF missing_count > 0 THEN
        RAISE EXCEPTION 'core_profile has % rows with unknown theme ids', missing_count;
    END IF;
END $$;

ALTER TABLE core_profile ADD COLUMN theme_id_int INTEGER;
UPDATE core_profile p
SET theme_id_int = t.int_id
FROM theme_v2 t
WHERE p.theme_id = t.id;

ALTER TABLE core_message ADD COLUMN theme_int INTEGER;
UPDATE core_message m
SET theme_int = t.int_id
FROM theme_v2 t
WHERE m.theme = t.id;

ALTER TABLE core_ragchunk ADD COLUMN theme_int INTEGER;
UPDATE core_ragchunk r
SET theme_int = t.int_id
FROM theme_v2 t
WHERE r.theme = t.id;
ALTER TABLE core_ragchunk ALTER COLUMN theme_int SET NOT NULL;

ALTER TABLE core_bibletextflat ADD COLUMN theme_int INTEGER;
UPDATE core_bibletextflat b
SET theme_int = t.int_id
FROM theme_v2 t
WHERE b.theme = t.id;
ALTER TABLE core_bibletextflat ALTER COLUMN theme_int SET NOT NULL;

ALTER TABLE core_profile DROP COLUMN theme_id;
ALTER TABLE core_message DROP COLUMN theme;
ALTER TABLE core_ragchunk DROP COLUMN theme;
ALTER TABLE core_bibletextflat DROP COLUMN theme;

ALTER TABLE core_profile RENAME COLUMN theme_id_int TO theme_id;
ALTER TABLE core_message RENAME COLUMN theme_int TO theme;
ALTER TABLE core_ragchunk RENAME COLUMN theme_int TO theme;
ALTER TABLE core_bibletextflat RENAME COLUMN theme_int TO theme;

ALTER TABLE theme_v2 DROP CONSTRAINT theme_v2_pkey;
ALTER TABLE theme_v2 DROP COLUMN id;
ALTER TABLE theme_v2 RENAME COLUMN int_id TO id;
ALTER TABLE theme_v2 ADD CONSTRAINT theme_v2_pkey PRIMARY KEY (id);
ALTER TABLE theme_v2 ALTER COLUMN slug DROP NOT NULL;

CREATE SEQUENCE IF NOT EXISTS theme_id_seq START WITH 1 INCREMENT BY 1;
SELECT setval('theme_id_seq', COALESCE((SELECT MAX(id) FROM theme_v2), 1), true);
ALTER TABLE theme_v2 ALTER COLUMN id SET DEFAULT nextval('theme_id_seq');
ALTER SEQUENCE theme_id_seq OWNED BY theme_v2.id;

ALTER TABLE theme_v2 RENAME TO theme;
ALTER TABLE theme ADD CONSTRAINT theme_slug_key UNIQUE (slug);

ALTER TABLE core_profile
    ADD CONSTRAINT core_profile_theme_id_fkey
    FOREIGN KEY (theme_id) REFERENCES theme(id) DEFERRABLE INITIALLY DEFERRED;
ALTER TABLE core_message
    ADD CONSTRAINT core_message_theme_fkey
    FOREIGN KEY (theme) REFERENCES theme(id) DEFERRABLE INITIALLY DEFERRED;
ALTER TABLE core_ragchunk
    ADD CONSTRAINT core_ragchunk_theme_fkey
    FOREIGN KEY (theme) REFERENCES theme(id) DEFERRABLE INITIALLY DEFERRED;
ALTER TABLE core_bibletextflat
    ADD CONSTRAINT core_bibletextflat_theme_fkey
    FOREIGN KEY (theme) REFERENCES theme(id) DEFERRABLE INITIALLY DEFERRED;

CREATE INDEX core_profile_theme_id_idx ON core_profile(theme_id);
CREATE INDEX core_message_theme_idx ON core_message(theme);
CREATE INDEX core_ragchunk_theme_idx ON core_ragchunk(theme);
CREATE INDEX core_bibletextflat_theme_idx ON core_bibletextflat(theme);
"""


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0027_profile_last_simulation_report"),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            database_operations=[
                migrations.RunSQL(SQL_FORWARD, migrations.RunSQL.noop),
            ],
            state_operations=[
                migrations.RenameModel(
                    old_name="ThemeV2",
                    new_name="Theme",
                ),
                migrations.AlterModelTable(
                    name="theme",
                    table="theme",
                ),
                migrations.AlterField(
                    model_name="theme",
                    name="id",
                    field=models.AutoField(primary_key=True, serialize=False),
                ),
                migrations.AddField(
                    model_name="theme",
                    name="slug",
                    field=models.CharField(
                        blank=True,
                        db_index=True,
                        max_length=80,
                        null=True,
                        unique=True,
                    ),
                ),
                migrations.AlterField(
                    model_name="profile",
                    name="theme",
                    field=models.ForeignKey(
                        blank=True,
                        help_text="Assigned conversation theme (temporary)",
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        to="core.theme",
                    ),
                ),
                migrations.AlterField(
                    model_name="message",
                    name="theme",
                    field=models.ForeignKey(
                        blank=True,
                        db_column="theme",
                        db_index=True,
                        help_text="Classified predominant theme for this message",
                        null=True,
                        on_delete=django.db.models.deletion.PROTECT,
                        to="core.theme",
                    ),
                ),
                migrations.AlterField(
                    model_name="ragchunk",
                    name="theme",
                    field=models.ForeignKey(
                        db_column="theme",
                        db_index=True,
                        help_text="Theme used to constrain RAG retrieval",
                        on_delete=django.db.models.deletion.PROTECT,
                        to="core.theme",
                    ),
                ),
                migrations.AlterField(
                    model_name="bibletextflat",
                    name="theme",
                    field=models.ForeignKey(
                        db_column="theme",
                        db_index=True,
                        on_delete=django.db.models.deletion.PROTECT,
                        to="core.theme",
                    ),
                ),
            ],
        ),
    ]
