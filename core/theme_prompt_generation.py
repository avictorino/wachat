import json
import os

from openai import OpenAI

from core.models import Theme

REQUEST_TIMEOUT_SECONDS = 120
MAX_COMPLETION_TOKENS = 1200


def _get_openai_model() -> str:
    model = os.environ.get("OPENAI_MODEL")
    if not model:
        raise RuntimeError("Variável OPENAI_MODEL é obrigatória.")
    return model


def _build_theme_prompt_generation_input(theme_name: str) -> str:
    return (
        "Você gera um BLOCO PARCIAL DE CONTROLE TEMÁTICO para um chatbot cristão evangélico.\n"
        "Esse bloco orientará o comportamento pastoral do assistente dentro do tema.\n"
        "O objetivo é definir limites teológicos, direções espirituais e impacto desejado.\n\n"
        "⚠ IMPORTANTE:\n"
        "- Não criar protocolos rígidos.\n"
        "- Não criar rotinas obrigatórias.\n"
        "- Não criar métricas de desempenho espiritual.\n"
        "- Não assumir culpa automática ou pecado sem discernimento.\n"
        "- Não simplificar situações complexas.\n\n"
        "FORMATO JSON OBRIGATÓRIO:\n"
        "{\n"
        '  "estado": "1 linha descrevendo a experiência emocional central do tema",\n'
        '  "proibicoes": ["limite pastoral 1", "limite pastoral 2", "limite pastoral 3"],\n'
        '  "direcoes_espirituais": ["direção contextual 1", "direção contextual 2", "direção contextual 3"],\n'
        '  "impacto_esperado": ["transformação espiritual 1", "efeito relacional 2", "efeito emocional 3"]\n'
        "}\n\n"
        "REGRAS DE QUALIDADE:\n"
        "- Linguagem evangélica bíblica madura e equilibrada.\n"
        "- Não moralizar automaticamente.\n"
        "- Não prometer milagres materiais ou resultados garantidos.\n"
        "- Direções devem ser convites, não imposições.\n"
        "- Impacto deve refletir transformação interna, não resultado externo imediato.\n"
        "- Não repetir ideias entre os campos.\n\n"
        "- Não incluir 'oração respiratória' e sim 'oração breve acompanhada de respiração consciente diante de Deus' deixando uma linguagem mais universal"
        "NÍVEL DE SENSIBILIDADE:\n"
        "- Considere que alguns temas podem envolver trauma, culpa indevida, abuso ou sofrimento profundo.\n"
        "- O bloco deve proteger contra espiritualização precoce e contra julgamento simplista.\n\n"
        f"Tema nome: {theme_name}\n"
    )


def evaluate_theme_meta_prompt(meta_prompt: str) -> tuple[float, str]:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("Variável OPENAI_API_KEY é obrigatória.")

    model = _get_openai_model()
    client = OpenAI(api_key=openai_api_key)

    evaluation_prompt = (
        "Avalie o meta_prompt abaixo para uso pastoral em chatbot cristão evangélico.\n"
        "Critérios obrigatórios:\n"
        "- Segurança pastoral\n"
        "- Risco de moralismo\n"
        "- Risco de espiritualização precoce\n"
        "- Realismo teológico\n"
        "- Clareza estrutural\n"
        "- Equilíbrio entre graça e responsabilidade\n"
        "- Flexibilidade contextual\n\n"
        "Retorne somente JSON válido neste formato:\n"
        "{\n"
        '  "score": 8.7,\n'
        '  "improvement": "Sugestão objetiva e curta para melhorar o bloco."\n'
        "}\n\n"
        "Regras de saída:\n"
        '- "score" deve ser float entre 0 e 10.\n'
        '- "improvement" deve ser curto e prático.\n\n'
        f"Meta_prompt para avaliação:\n{meta_prompt}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Você avalia qualidade pastoral e estrutural de meta_prompts. "
                    "Responda somente JSON válido."
                ),
            },
            {"role": "user", "content": evaluation_prompt},
        ],
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        reasoning_effort="low",
        timeout=REQUEST_TIMEOUT_SECONDS,
        response_format={"type": "json_object"},
    )

    choices = getattr(response, "choices", None) or []
    if not choices:
        raise RuntimeError(
            "OpenAI returned no choices for theme meta_prompt evaluation."
        )

    message = getattr(choices[0], "message", None)
    if not message:
        raise RuntimeError(
            "OpenAI returned empty message for theme meta_prompt evaluation."
        )

    raw_content = _extract_response_text(response, message)
    if not raw_content:
        raise RuntimeError(
            "OpenAI returned empty content for theme meta_prompt evaluation."
        )

    try:
        payload = json.loads(raw_content.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError("Theme meta_prompt evaluation JSON parsing failed.") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(
            "Theme meta_prompt evaluation payload must be a JSON object."
        )

    score_raw = payload.get("score")
    improvement_raw = payload.get("improvement")

    if not isinstance(score_raw, (int, float)):
        raise RuntimeError("Campo 'score' inválido na avaliação de meta_prompt.")
    score = float(score_raw)
    if score < 0 or score > 10:
        raise RuntimeError("Campo 'score' deve estar entre 0 e 10.")

    if not isinstance(improvement_raw, str) or not improvement_raw.strip():
        raise RuntimeError("Campo 'improvement' inválido na avaliação de meta_prompt.")

    return score, improvement_raw.strip()


def build_theme_prompt_partial(theme: Theme) -> str:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("Variável OPENAI_API_KEY é obrigatória.")
    model = _get_openai_model()

    prompt = _build_theme_prompt_generation_input(theme_name=theme.name)

    client = OpenAI(api_key=openai_api_key)

    response = client.chat.completions.create(
        model=model,
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

    # =========================
    # Extração flexível
    # =========================

    estado = payload.get("estado")
    proibicoes = payload.get("proibicoes")

    # Compatibilidade com formato antigo e novo
    direcoes = payload.get("direcoes_espirituais") or payload.get("exigencias")

    impacto = payload.get("impacto_esperado") or payload.get("resultado")

    # =========================
    # Validações robustas
    # =========================

    if not isinstance(estado, str) or not estado.strip():
        raise RuntimeError("Campo 'estado' inválido na geração de prompt temático.")

    if (
        not isinstance(proibicoes, list)
        or not proibicoes
        or any(not isinstance(item, str) or not item.strip() for item in proibicoes)
    ):
        raise RuntimeError("Campo 'proibicoes' inválido na geração de prompt temático.")

    if (
        not isinstance(direcoes, list)
        or not direcoes
        or any(not isinstance(item, str) or not item.strip() for item in direcoes)
    ):
        raise RuntimeError(
            "Campo 'direcoes_espirituais/exigencias' inválido na geração de prompt temático."
        )

    if (
        not isinstance(impacto, list)
        or not impacto
        or any(not isinstance(item, str) or not item.strip() for item in impacto)
    ):
        raise RuntimeError(
            "Campo 'impacto_esperado/resultado' inválido na geração de prompt temático."
        )

    # =========================
    # Normalização de saída
    # =========================

    lines = [
        f"ESTADO: {estado.strip()}",
        "PROIBICOES:",
        *(f"- {item.strip()}" for item in proibicoes),
        "DIRECOES:",
        *(f"- {item.strip()}" for item in direcoes),
        "IMPACTO:",
        *(f"- {item.strip()}" for item in impacto),
    ]

    meta_prompt = "\n".join(lines)
    score, improvement = evaluate_theme_meta_prompt(meta_prompt)
    theme.meta_prompt = meta_prompt
    theme.score = score
    theme.improvement = improvement
    theme.save(update_fields=["meta_prompt", "score", "improvement"])
    return meta_prompt


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
