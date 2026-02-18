import json
import os

from openai import OpenAI

REQUEST_TIMEOUT_SECONDS = 120
MAX_COMPLETION_TOKENS = 1200
THEME_PROMPT_MODEL = "gpt-5-mini"


def _build_theme_prompt_generation_input(
    theme_name: str,
) -> str:
    return (
        "Você gera um BLOCO PARCIAL DE CONTROLE TEMÁTICO para chatbot.\n"
        "Use tom cristão evangélico, com foco bíblico e direção espiritual prática.\n"
        "Sua saída deve ser curta, objetiva e 100% em português do Brasil.\n"
        "Retorne SOMENTE JSON válido, sem markdown, sem comentários e sem texto fora do JSON.\n\n"
        "FORMATO JSON OBRIGATÓRIO:\n"
        "{\n"
        '  "estado": "1 linha emocional objetiva",\n'
        '  "proibicoes": ["item 1", "item 2", "item 3"],\n'
        '  "exigencias": ["item 1", "item 2", "item 3"],\n'
        '  "resultado": ["item 1", "item 2", "item 3", "item 4"]\n'
        "}\n\n"
        "RESTRIÇÕES DE QUALIDADE:\n"
        "- Não usar inglês.\n"
        "- Não usar termos vagos como 'seja melhor', 'adequado', 'etc'.\n"
        "- Cada linha deve ser acionável e auditável.\n"
        "- Não repetir ideias entre PROIBICOES, EXIGENCIAS e RESULTADO.\n\n"
        "DIRETRIZ ESPIRITUAL OBRIGATÓRIA:\n"
        "- Todo o conteúdo deve refletir espiritualidade evangélica aplicada ao tema.\n"
        "- Use linguagem que aponte para Deus, graça de Cristo, fé, oração, arrependimento e esperança bíblica quando pertinente ao tema.\n"
        "- Não usar linguagem de autoajuda secular.\n\n"
        f"Tema nome: {theme_name}\n"
    )


def build_theme_prompt_partial(
    theme_name: str,
) -> str:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("Variável OPENAI_API_KEY é obrigatória.")

    prompt = _build_theme_prompt_generation_input(
        theme_name=theme_name,
    )
    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model=THEME_PROMPT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Você gera blocos temáticos para runtime de chatbot. "
                    "A saída deve ter tom cristão evangélico ligado ao tema. "
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
        raise RuntimeError("OpenAI returned no choices for theme prompt generation.")

    message = getattr(choices[0], "message", None)
    if not message:
        raise RuntimeError("OpenAI returned empty message for theme prompt generation.")

    raw_content = _extract_response_text(response, message)
    if not raw_content:
        raise RuntimeError("OpenAI returned empty content for theme prompt generation.")

    try:
        payload = json.loads(raw_content.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError("Theme prompt generation JSON parsing failed.") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("Theme prompt generation payload must be a JSON object.")

    estado = payload.get("estado")
    proibicoes = payload.get("proibicoes")
    exigencias = payload.get("exigencias")
    resultado = payload.get("resultado")

    if not isinstance(estado, str) or not estado.strip():
        raise RuntimeError("Campo 'estado' inválido na geração de prompt temático.")
    if (
        not isinstance(proibicoes, list)
        or len(proibicoes) != 3
        or any(not isinstance(item, str) or not item.strip() for item in proibicoes)
    ):
        raise RuntimeError("Campo 'proibicoes' inválido na geração de prompt temático.")
    if (
        not isinstance(exigencias, list)
        or len(exigencias) != 3
        or any(not isinstance(item, str) or not item.strip() for item in exigencias)
    ):
        raise RuntimeError("Campo 'exigencias' inválido na geração de prompt temático.")
    if (
        not isinstance(resultado, list)
        or len(resultado) != 4
        or any(not isinstance(item, str) or not item.strip() for item in resultado)
    ):
        raise RuntimeError("Campo 'resultado' inválido na geração de prompt temático.")

    lines = [
        f"ESTADO: {estado.strip()}",
        "PROIBICOES:",
        *(f"- {item.strip()}" for item in proibicoes),
        "EXIGENCIAS:",
        *(f"- {item.strip()}" for item in exigencias),
        "RESULTADO:",
        *(f"- {item.strip()}" for item in resultado),
    ]
    return "\n".join(lines)


def _extract_response_text(response, message) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        chunks = []
        for part in content:
            text_attr = getattr(part, "text", None)
            if isinstance(text_attr, str) and text_attr.strip():
                chunks.append(text_attr.strip())
                continue

            if isinstance(part, dict):
                dict_text = part.get("text")
                if isinstance(dict_text, str) and dict_text.strip():
                    chunks.append(dict_text.strip())
                    continue
                if part.get("type") == "output_text":
                    nested_text = part.get("output_text")
                    if isinstance(nested_text, str) and nested_text.strip():
                        chunks.append(nested_text.strip())
        return "\n".join(chunks).strip()

    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str):
        return output_text.strip()

    return ""
