import json
import os
import random
from base64 import b64decode
from typing import Any, Dict, Optional
from uuid import uuid4

from django.core.files.base import ContentFile
from openai import OpenAI

from core.models import Message, Profile, SocialMediaExport

REQUEST_TIMEOUT_SECONDS = 120
MAX_COMPLETION_TOKENS = 1400
MAX_EXPORTS_PER_PROFILE = 10
MIN_ASSISTANT_SCORE = 6.0
IMAGE_MODEL = "gpt-image-1"
IMAGE_SIZE = "512x768"
IMAGE_QUALITY_LOW = "low"
IMAGE_QUALITY_MEDIUM = "medium"
IMAGE_QUALITY_HIGH = "high"
IMAGE_QUALITY_AUTO = "auto"
IMAGE_PRICING_USD = {
    IMAGE_QUALITY_LOW: 0.016,
    IMAGE_QUALITY_MEDIUM: 0.063,
    IMAGE_QUALITY_HIGH: 0.25,
}
PORTRAIT_VARIANTS = [
    "rosto em 3/4 virado para a esquerda",
    "rosto em 3/4 virado para a direita",
    "perfil lateral esquerdo",
    "perfil lateral direito",
    "frontal com leve inclinação de cabeça",
]
BACKGROUND_VARIANTS = [
    "pessoas em cenário cotidiano com profundidade de campo suave",
    "litoral com mar ao fundo, céu aberto e luz natural",
    "paisagem natural ampla com montanhas, vegetação e horizonte",
    "ambiente urbano histórico com arquitetura antiga e ruas de pedra",
    "interior contemplativo com atmosfera espiritual discreta (sem símbolos explícitos)",
    "monumentos da antiguidade e ruínas históricas, sem estátuas e sem cruzes",
    "arquitetura antiga do mediterrâneo com tons quentes e textura de pedra",
    "cenário externo ao entardecer com sensação de esperança e recolhimento",
]


def _get_openai_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Variável OPENAI_API_KEY é obrigatória.")
    return api_key


def _get_openai_model() -> str:
    model = os.environ.get("OPENAI_MODEL")
    if not model:
        raise RuntimeError("Variável OPENAI_MODEL é obrigatória.")
    return model


class SocialMediaExportService:
    def __init__(self):
        self.client = OpenAI(api_key=_get_openai_api_key())
        self.model = _get_openai_model()

    def export_profile_messages(self, profile: Profile) -> int:
        created_count = 0
        candidates = self._candidate_assistant_messages(profile=profile)

        for assistant_message in candidates:
            user_message = self._previous_user_message(
                assistant_message=assistant_message
            )
            if user_message is None:
                continue

            payload = self._adapt_pair_for_social_media(
                user_question=user_message.content,
                assistant_answer=assistant_message.content,
            )

            original_text = (
                "Pergunta do usuário:\n"
                + user_message.content.strip()
                + "\n\n"
                + "Resposta original do assistente:\n"
                + assistant_message.content.strip()
            )

            SocialMediaExport.objects.create(
                original_message=assistant_message,
                original_text=original_text,
                adapted_text=payload["adapted_text"],
                image_summary=payload["image_summary"],
                religous_reference=payload["religous_reference"],
                score=payload["score"],
                is_religious=payload["is_religious"],
            )
            created_count += 1

        return created_count

    def generate_image_for_export(self, export_item: SocialMediaExport) -> None:
        image_prompt = self._build_image_prompt(export_item=export_item)
        response = self.client.images.generate(
            model=IMAGE_MODEL,
            prompt=image_prompt,
            size=IMAGE_SIZE,
            quality="low",
        )

        data = getattr(response, "data", None) or []
        if not data:
            raise RuntimeError("OpenAI returned no image data for social media export.")

        first_item = data[0]
        image_b64 = getattr(first_item, "b64_json", None)
        if not image_b64 and isinstance(first_item, dict):
            image_b64 = first_item.get("b64_json")
        if not image_b64:
            raise RuntimeError(
                "OpenAI returned image payload without b64_json for social media export."
            )

        usage_payload = self._extract_usage_payload(response=response)
        quality_used = self._extract_quality(response=response, first_item=first_item)
        pricing_payload = self._build_pricing_payload(quality=quality_used)

        image_bytes = b64decode(image_b64)
        filename = f"social-export-{export_item.id}-{uuid4().hex}.png"
        export_item.generated_image.save(filename, ContentFile(image_bytes), save=False)
        export_item.image_generation_usage = {
            "provider": "openai",
            "model": IMAGE_MODEL,
            "size": IMAGE_SIZE,
            "quality": quality_used,
            "usage": usage_payload,
            "pricing": pricing_payload,
        }
        export_item.save(update_fields=["generated_image", "image_generation_usage"])

    def _candidate_assistant_messages(self, profile: Profile):
        return (
            Message.objects.filter(profile=profile, role="assistant")
            .filter(score__gte=MIN_ASSISTANT_SCORE)
            .filter(social_media_export__isnull=True)
            .order_by("-score", "-created_at")[:MAX_EXPORTS_PER_PROFILE]
        )

    def _previous_user_message(self, assistant_message: Message) -> Optional[Message]:
        return (
            Message.objects.filter(
                profile=assistant_message.profile,
                role="user",
                created_at__lt=assistant_message.created_at,
            )
            .order_by("-created_at")
            .first()
        )

    def _adapt_pair_for_social_media(
        self, user_question: str, assistant_answer: str
    ) -> Dict[str, object]:
        prompt = (
            "Converta o trecho abaixo em formato para Instagram.\n"
            "Entrada:\n"
            f"Pergunta do usuário: {user_question}\n"
            f"Resposta do assistente: {assistant_answer}\n\n"
            "Objetivo:\n"
            "- Extrair a essência mais significativa da conversa.\n"
            "- Produzir texto adaptado com linguagem natural, midiática e clara.\n"
            "- O texto deve funcionar como descrição de post.\n"
            "- Produzir resumo curto para sobrepor na imagem.\n"
            "- Classificar se o trecho tem conteúdo religioso explícito.\n"
            "- Definir score de relevância para redes (0 a 10).\n\n"
            "Formato de saída obrigatório em JSON:\n"
            "{\n"
            '  "adapted_text": "texto final para descrição do post no Instagram",\n'
            '  "image_summary": "resumo curto (até 140 caracteres) para fixar na imagem",\n'
            '  "religous_reference": "passagem bíblica relacionada (com referência) OU trecho curto de sermão famoso",\n'
            '  "is_religious": true,\n'
            '  "score": 8.4\n'
            "}\n\n"
            "Regras:\n"
            "- adapted_text deve preservar sentido central, mas pode reescrever.\n"
            "- adapted_text deve incluir: gancho inicial, contexto e fechamento breve.\n"
            "- image_summary deve ser direto e memorável.\n"
            "- religous_reference deve trazer uma citação curta e relevante.\n"
            "- score precisa ser número entre 0 e 10.\n"
            "- Responda apenas JSON válido."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Você adapta trechos de conversa para social media. "
                        "Responda somente JSON válido."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            reasoning_effort="low",
            timeout=REQUEST_TIMEOUT_SECONDS,
            response_format={"type": "json_object"},
        )

        choices = getattr(response, "choices", None) or []
        if not choices:
            raise RuntimeError("OpenAI returned no choices for social media export.")

        message = getattr(choices[0], "message", None)
        if message is None:
            raise RuntimeError("OpenAI returned empty message for social media export.")

        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("OpenAI returned empty content for social media export.")

        payload = json.loads(content.strip())
        if not isinstance(payload, dict):
            raise RuntimeError("Social media export payload must be a JSON object.")

        adapted_text = payload.get("adapted_text")
        image_summary = payload.get("image_summary")
        religous_reference = payload.get("religous_reference")
        is_religious = payload.get("is_religious")
        score = payload.get("score")

        if not isinstance(adapted_text, str) or not adapted_text.strip():
            raise RuntimeError("Campo 'adapted_text' inválido na exportação social.")
        if not isinstance(image_summary, str) or not image_summary.strip():
            raise RuntimeError("Campo 'image_summary' inválido na exportação social.")
        if not isinstance(religous_reference, str) or not religous_reference.strip():
            raise RuntimeError(
                "Campo 'religous_reference' inválido na exportação social."
            )
        if not isinstance(is_religious, bool):
            raise RuntimeError("Campo 'is_religious' inválido na exportação social.")
        if not isinstance(score, (int, float)):
            raise RuntimeError("Campo 'score' inválido na exportação social.")

        normalized_score = float(score)
        if normalized_score < 0 or normalized_score > 10:
            raise RuntimeError("Campo 'score' deve estar entre 0 e 10.")

        return {
            "adapted_text": adapted_text.strip(),
            "image_summary": image_summary.strip(),
            "religous_reference": religous_reference.strip(),
            "is_religious": is_religious,
            "score": normalized_score,
        }

    def _build_image_prompt(self, export_item: SocialMediaExport) -> str:
        if not export_item.religous_reference:
            raise RuntimeError(
                "Campo 'religous_reference' é obrigatório para gerar imagem."
            )
        if not export_item.adapted_text:
            raise RuntimeError("Campo 'adapted_text' é obrigatório para gerar imagem.")
        if not export_item.image_summary:
            raise RuntimeError("Campo 'image_summary' é obrigatório para gerar imagem.")

        portrait_variant = random.choice(PORTRAIT_VARIANTS)
        background_variant = random.choice(BACKGROUND_VARIANTS)

        return (
            "Crie uma imagem vertical para Instagram com foco em legibilidade mobile.\n"
            "Regras de composição:\n"
            "- 60% da direção criativa vem da referência religiosa.\n"
            "- 40% da direção criativa vem do texto adaptado.\n"
            "- Visual editorial moderno, forte, humano e sem aparência genérica.\n"
            "- Sem logos e sem marca d'água.\n"
            "- Renderize o texto diretamente na arte (sem caixa/tarja extra pós-processada).\n"
            "- Não repetir blocos de texto.\n"
            "- Não incluir texto adicional fora dos blocos obrigatórios.\n"
            "- Ortografia e acentuação em português devem estar corretas.\n\n"
            "Variação visual obrigatória:\n"
            f"- Enquadramento da pessoa: {portrait_variant}.\n"
            f"- Background: {background_variant}.\n"
            "- Não usar estátuas.\n"
            "- Não usar cruzes.\n"
            "- Evite repetir composição padrão centralizada em todos os casos.\n\n"
            "Textos obrigatórios na arte (use EXATAMENTE estes textos):\n"
            "1) Headline principal (maior destaque):\n"
            f"{export_item.image_summary.strip()}\n\n"
            "2) Referência religiosa (menor destaque, em uma linha de citação):\n"
            f"{export_item.religous_reference.strip()}\n\n"
            "Layout obrigatório de texto:\n"
            "- Headline no terço superior ou central.\n"
            "- Referência religiosa no terço inferior.\n"
            "- Garantir contraste e leitura clara dos dois blocos.\n\n"
            "- Evitar colocar texto muito no topo.\n"
            "- Evitar colocar texto muito no rodapé.\n"
            "- Instagram sobrepõe UI; manter textos dentro da área segura central.\n\n"
            "Referência religiosa (peso 60%):\n"
            f"{export_item.religous_reference.strip()}\n\n"
            "Texto adaptado (peso 40%):\n"
            f"{export_item.adapted_text.strip()}"
        )

    def _extract_usage_payload(self, response: Any) -> Dict[str, Any]:
        usage = getattr(response, "usage", None)
        if usage is None:
            raise RuntimeError(
                "OpenAI image generation response did not include usage payload."
            )

        if hasattr(usage, "model_dump"):
            payload = usage.model_dump()
            if not isinstance(payload, dict):
                raise RuntimeError("OpenAI usage payload format is invalid.")
            return payload

        if isinstance(usage, dict):
            return usage

        raise RuntimeError("OpenAI usage payload type is unsupported.")

    def _extract_quality(self, response: Any, first_item: Any) -> str:
        quality = getattr(response, "quality", None)
        if not quality and isinstance(first_item, dict):
            quality = first_item.get("quality")
        if not quality and hasattr(first_item, "quality"):
            quality = getattr(first_item, "quality")
        if not quality:
            return IMAGE_QUALITY_AUTO
        return str(quality).strip().lower()

    def _build_pricing_payload(self, quality: str) -> Dict[str, Any]:
        if quality in IMAGE_PRICING_USD:
            return {
                "currency": "USD",
                "estimated_cost_usd": IMAGE_PRICING_USD[quality],
                "quality_source": quality,
                "reference": "gpt-image-1 pricing table",
            }

        return {
            "currency": "USD",
            "estimated_cost_usd": None,
            "estimated_range_usd": {
                "min": IMAGE_PRICING_USD[IMAGE_QUALITY_LOW],
                "max": IMAGE_PRICING_USD[IMAGE_QUALITY_HIGH],
            },
            "quality_source": quality,
            "reference": "gpt-image-1 pricing table (quality auto/unknown)",
        }
