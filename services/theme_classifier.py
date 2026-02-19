from core.models import Theme
from services.openai_service import OpenAIService

THEME_CLASSIFIER_MODEL = "gpt-4o-mini"
THEME_CLASSIFIER_TEMPERATURE = 0.1
THEME_CLASSIFIER_MAX_COMPLETION_TOKENS = 10
THEME_CLASSIFIER_TIMEOUT_SECONDS = 60


class ThemeClassifier:
    def __init__(self):
        self._llm_service = OpenAIService()

    def classify(self, text: str) -> int:
        if not text or not text.strip():
            raise ValueError("Text is required for theme classification.")

        client = getattr(self._llm_service, "client", None)
        if client is None:
            raise RuntimeError("OpenAI client is not available for theme classifier.")

        allowed_themes = list(
            Theme.objects.all().order_by("id").values("id", "name", "slug")
        )
        if not allowed_themes:
            raise RuntimeError("No themes found in database for classification.")
        allowed_theme_ids = [theme["id"] for theme in allowed_themes]
        allowed_theme_catalog_lines = []
        for theme in allowed_themes:
            allowed_theme_catalog_lines.append(
                f"{theme['id']} | nome={theme.get('name') or ''} | slug={theme.get('slug') or ''}"
            )
        allowed_theme_catalog = "\n".join(allowed_theme_catalog_lines)

        response = client.chat.completions.create(
            model=THEME_CLASSIFIER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você é um classificador estrito.\n"
                        "Retorne APENAS um tema da lista permitida.\n\n"
                        "Classifique pelo núcleo emocional predominante, não por contexto incidental.\n"
                        "Quando houver ambiguidade, use esta prioridade de desempate:\n"
                        "1) Emoção/sofrimento nomeado explicitamente\n"
                        "2) Estado interno persistente\n"
                        "3) Contexto externo (trabalho, dinheiro, relacionamentos)\n"
                        "Se houver termos como 'ansioso/ansiosa/ansiedade/pânico', prefira tema de Ansiedade.\n"
                        "Se houver termos de gasto, dívida, boleto, conta, cartão ou compulsão financeira, prefira Dinheiro e dívidas.\n"
                        "Use Luto e perda apenas quando houver evidência explícita de luto/perda/morte/saudade de alguém.\n"
                        "Não explique sua escolha.\n\n"
                        f"Temas permitidos (id | nome | slug):\n{allowed_theme_catalog}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Classifique o tema emocional ou de vida predominante desta mensagem.\n"
                        "Foque no que está causando a maior carga emocional agora.\n\n"
                        f'Mensagem:\n"{text}"\n\n'
                        "Retorne apenas o id numérico do tema."
                    ),
                },
            ],
            temperature=THEME_CLASSIFIER_TEMPERATURE,
            max_completion_tokens=THEME_CLASSIFIER_MAX_COMPLETION_TOKENS,
            timeout=THEME_CLASSIFIER_TIMEOUT_SECONDS,
        )

        choices = getattr(response, "choices", None) or []
        if not choices:
            raise RuntimeError("Theme classifier returned no choices.")

        message = getattr(choices[0], "message", None)
        if not message:
            raise RuntimeError("Theme classifier returned empty message.")

        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Theme classifier returned empty content.")

        try:
            theme = int(content.strip())
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid non-integer theme returned by classifier: '{content.strip()}'"
            ) from exc

        if theme not in allowed_theme_ids:
            raise RuntimeError(f"Invalid theme returned by classifier: '{theme}'")

        return theme
