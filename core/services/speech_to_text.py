from __future__ import annotations

import io
import logging
import os
from typing import Optional

import requests
from groq import Groq
from requests.auth import HTTPBasicAuth

logger = logging.getLogger(__name__)


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