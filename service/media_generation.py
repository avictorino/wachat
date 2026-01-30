"""
Media generation services for text-to-speech, speech-to-text, and text-to-image.

This module provides functionality for converting between different media formats
and generating contemplative biblical images.
"""
import base64
import io
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import requests
from django.conf import settings
from elevenlabs.client import ElevenLabs
from elevenlabs.types.voice_settings import VoiceSettings
from groq import Groq
from openai import OpenAI
from requests.auth import HTTPBasicAuth

from core.constants import ConversationMode
from service.prompts import image_generation_base_prompt

logger = logging.getLogger(__name__)


# ============================================================================
# Text-to-Speech Service
# ============================================================================


class TextToSpeechService:
    """
    Minimal TTS service using ElevenLabs SDK.
    """

    def __init__(self):
        self._audio_dir = Path(settings.MEDIA_ROOT) / "audio"
        self._audio_dir.mkdir(parents=True, exist_ok=True)

        self._client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

    def speak_and_store(
        self, text: str, conversation_mode: ConversationMode, previous_text: str = None
    ) -> str:
        filename = f"{uuid.uuid4()}.mp3"
        path = self._audio_dir / filename

        if conversation_mode == ConversationMode.LISTENING:
            voice_settings = VoiceSettings(
                stability=0.72,  # voz bem estável, transmite segurança
                similarity_boost=0.60,  # mantém identidade sem rigidez
                style=0.22,  # pouca teatralidade
                use_speaker_boost=True,  # presença suave
            )
        elif conversation_mode == ConversationMode.REFLECTIVE:
            voice_settings = VoiceSettings(
                stability=0.68,  # estável, mas não rígido
                similarity_boost=0.58,  # menos "voz padrão"
                style=0.35,  # emoção presente
                use_speaker_boost=False,  # menos projeção, mais intimidade
            )
        elif conversation_mode in (
            ConversationMode.SPIRITUAL_AWARENESS,
            ConversationMode.BIBLICAL,
        ):
            voice_settings = VoiceSettings(
                stability=0.55,  # menos estabilidade → mais energia
                similarity_boost=0.65,  # voz mais presente
                style=0.45,  # mais expressão emocional
                use_speaker_boost=True,  # autoridade
            )
        else:
            logger.error("Unknown conversation mode for TTS: %s", conversation_mode)
            voice_settings = VoiceSettings(
                stability=0.72,  # voz bem estável, transmite segurança
                similarity_boost=0.60,  # mantém identidade sem rigidez
                style=0.22,  # pouca teatralidade
                use_speaker_boost=True,  # presença suave
            )

        audio_generator = self._client.text_to_speech.convert(
            text=text,
            previous_text=previous_text,
            voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
            model_id=os.getenv("ELEVENLABS_MODEL_ID"),
            output_format="mp3_44100_128",
            voice_settings=voice_settings,
            apply_text_normalization="on",
            language_code="pt",
            seed=42,
        )

        with open(path, "wb") as f:
            for chunk in audio_generator:
                f.write(chunk)

        self.cleanup_old_audio()

        return f"{settings.SITE_URL}/media/audio/{filename}"

    def cleanup_old_audio(self, hours: int = 2) -> None:
        cutoff = time.time() - (hours * 3600)

        for file in self._audio_dir.glob("*.mp3"):
            try:
                if file.stat().st_mtime < cutoff:
                    file.unlink()
            except FileNotFoundError:
                continue
            except Exception:
                logger.exception("Failed to delete audio file: %s", file)


# ============================================================================
# Speech-to-Text Service
# ============================================================================


class GroqWhisperSTT:
    """
    Speech-to-Text service using Groq + Whisper.
    Receives raw audio bytes and returns transcribed text.
    """

    def __init__(self, model: str = "whisper-large-v3"):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model

    @staticmethod
    def download_audio_from_twilio(media_url: str) -> bytes:
        """
        Download audio media from Twilio using Basic Auth.
        Twilio media URLs are protected and require Account SID + Auth Token.
        """
        if not media_url:
            raise ValueError("media_url is required to download audio from Twilio")

        response = requests.get(
            media_url,
            auth=HTTPBasicAuth(
                os.getenv("TWILIO_ACCOUNT_SID"),
                os.getenv("TWILIO_AUTH_TOKEN"),
            ),
            timeout=15,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to download media from Twilio "
                f"(status={response.status_code})"
            )

        return response.content

    def transcribe_media_url(
        self,
        media_url: str,
        language: Optional[str] = "pt",
    ) -> str:
        """
        Transcribe raw audio bytes using Groq Whisper.

        Supported formats depend on Whisper:
        ogg, mp3, wav, m4a, webm, etc.

        :param audio_bytes: Raw audio bytes
        :param language: Optional language hint (default: pt)
        :return: Transcribed text
        """

        audio_bytes = self.download_audio_from_twilio(media_url)

        if not audio_bytes:
            return ""

        try:
            # Whisper requires a file-like object with a name
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "audio.ogg"

            response = self.client.audio.transcriptions.create(
                file=audio_file,
                model=self.model,
                language=language,
                response_format="verbose_json",
            )

            # Groq may return dict or object depending on SDK version
            if isinstance(response, dict):
                text = response.get("text")
            else:
                text = getattr(response, "text", None)

            return (text or "").strip()

        except Exception:
            logger.exception("Groq Whisper transcription failed")
            return ""


# ============================================================================
# Text-to-Image Generation
# ============================================================================


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
