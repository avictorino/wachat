THEME_CHOICES = [
    (1, "relacionamento", "Relacionamento e família"),
    (2, "dinheiro_e_dividas", "Dinheiro e dívidas"),
    (3, "vicios_e_recaidas", "Vícios e recaídas"),
    (4, "saude_e_cansaco", "Saúde e cansaço"),
    (5, "luto_e_perda", "Luto e perda"),
    (6, "trabalho_e_pressao", "Trabalho e pressão"),
    (7, "solidao", "Solidão"),
    (8, "espiritualidade", "Espiritualidade"),
    (9, "ansiedade", "Ansiedade"),
    (10, "outros", "Outro problema"),
]

THEME_IDS = [theme[0] for theme in THEME_CHOICES]
THEME_SLUGS = [theme[1] for theme in THEME_CHOICES]
THEME_SLUG_TO_ID = {slug: theme_id for theme_id, slug, _ in THEME_CHOICES}
THEME_ID_TO_SLUG = {theme_id: slug for theme_id, slug, _ in THEME_CHOICES}
