import logging
import os
import time
import uuid
from pathlib import Path

from django.conf import settings
from elevenlabs.client import ElevenLabs
from elevenlabs.types.voice_settings import VoiceSettings

from core.constants import ConversationMode

logger = logging.getLogger(__name__)


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

        if conversation_mode.LISTENING:
            voice_settings = VoiceSettings(
                stability=0.72,  # voz bem estável, transmite segurança
                similarity_boost=0.60,  # mantém identidade sem rigidez
                style=0.22,  # pouca teatralidade
                use_speaker_boost=True,  # presença suave
            )
        elif conversation_mode.REFLECTIVE:
            voice_settings = VoiceSettings(
                stability=0.68,  # estável, mas não rígido
                similarity_boost=0.58,  # menos “voz padrão”
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
