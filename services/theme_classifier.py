from core.models import ThemeV2
from services.openai_service import OpenAIService

THEME_CLASSIFIER_MODEL = "gpt-4o-mini"
THEME_CLASSIFIER_TEMPERATURE = 0.1
THEME_CLASSIFIER_MAX_COMPLETION_TOKENS = 10
THEME_CLASSIFIER_TIMEOUT_SECONDS = 60


class ThemeClassifier:
    def __init__(self):
        self._llm_service = OpenAIService()

    def classify(self, text: str) -> str:
        if not text or not text.strip():
            raise ValueError("Text is required for theme classification.")

        client = getattr(self._llm_service, "client", None)
        if client is None:
            raise RuntimeError("OpenAI client is not available for theme classifier.")

        allowed_theme_ids = list(
            ThemeV2.objects.values_list("id", flat=True).order_by("id")
        )
        if not allowed_theme_ids:
            raise RuntimeError("No themes found in database for classification.")

        response = client.chat.completions.create(
            model=THEME_CLASSIFIER_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict classifier.\n"
                        "Return ONLY one theme from the allowed list.\n\n"
                        f"Allowed themes:\n{allowed_theme_ids}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Classify the predominant emotional or life theme "
                        f'of this message:\n"{text}"\n\n'
                        "Return only the theme keyword."
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

        theme = content.strip().lower()
        if theme not in allowed_theme_ids:
            raise RuntimeError(f"Invalid theme returned by classifier: '{theme}'")

        return theme
