import base64
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

from openai import OpenAI

from config import settings
from core.services.prompt_builder import image_generation_base_prompt

logger = logging.getLogger(__name__)


def maybe_generate_image(
    image_analysis: "ImageAnalysis",  # noqa: F821
) -> Optional[str]:
    """
    Generates a contemplative biblical image using OpenAI Images
    if the analysis indicates it should be generated.

    Returns a public media URL or None.
    """

    if not image_analysis or not image_analysis.should_generate_image:
        return None

    # ---- Build prompt ----
    base_prompt = image_generation_base_prompt(image_analysis.image_type)
    elements = ", ".join(image_analysis.visual_elements or [])

    prompt = (
        f"{base_prompt} "
        f"{image_analysis.visual_description}. "
        f"Visual elements: {elements}. "
        f"Emotional tone: {image_analysis.emotional_tone}."
    )

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        # ---- Generate image ----
        result = client.images.generate(
            model=os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1"),
            prompt=prompt,
            size="1024x1024",
        )

        # ---- Decode image ----
        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)

        # ---- Save image ----
        image_dir = Path(settings.MEDIA_ROOT) / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{uuid.uuid4()}.png"
        path = image_dir / filename
        path.write_bytes(image_bytes)

        return f"{settings.SITE_URL}/media/images/{filename}"

    except Exception as ex:
        logger.error(f"Image generation failed: {ex}")
        return None
