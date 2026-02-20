import json
import os

from openai import OpenAI

from prompts.models import PromptComponent, PromptComponentVersion

REQUEST_TIMEOUT_SECONDS = 120
MAX_COMPLETION_TOKENS = 1200


def _get_openai_model() -> str:
    model = os.environ.get("OPENAI_MODEL")
    if not model:
        raise RuntimeError("Variável OPENAI_MODEL é obrigatória.")
    return model


def _get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Variável OPENAI_API_KEY é obrigatória.")
    return OpenAI(api_key=api_key)


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


def _to_command_description(description: str, component_key: str) -> str:
    normalized = (description or "").strip()
    if not normalized:
        return f"Defina as instruções operacionais do componente {component_key}."
    if normalized.endswith("."):
        return normalized
    return f"{normalized}."


def _get_active_version(component: PromptComponent) -> PromptComponentVersion:
    if component.active_version is None:
        raise RuntimeError(
            f"Componente '{component.key}' não possui versão ativa para regenerar."
        )
    active = PromptComponentVersion.objects.filter(
        component=component,
        version=component.active_version,
        status="active",
    ).first()
    if not active:
        raise RuntimeError(
            f"Versão ativa do componente '{component.key}' não foi encontrada."
        )
    return active


def regenerate_prompt_content(component: PromptComponent) -> tuple[str, str]:
    model = _get_openai_model()
    client = _get_openai_client()
    active = _get_active_version(component)
    command_description = _to_command_description(component.description, component.key)

    generation_prompt = (
        "Regere um prompt com melhoria incremental mantendo o mesmo propósito.\n"
        "O campo descrição deve expressar a intenção como comando operacional.\n"
        "Retorne SOMENTE JSON válido no formato:\n"
        "{\n"
        '  "description_command": "frase em tom de comando",\n'
        '  "prompt": "texto completo do novo prompt"\n'
        "}\n\n"
        "Regras:\n"
        "- Não remover restrições de segurança do prompt base.\n"
        "- Tornar instruções mais diretas e executáveis.\n"
        "- Evitar repetição e redundância.\n"
        "- Manter consistência com tipo/escopo/mode do componente.\n\n"
        f"COMPONENTE: {component.key}\n"
        f"TIPO: {component.component_type}\n"
        f"ESCOPO: {component.scope}\n"
        f"MODO: {component.mode or 'N/A'}\n"
        f"DESCRIÇÃO-ALVO (COMANDO): {command_description}\n\n"
        f"PROMPT ATIVO ATUAL:\n{active.content}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Você é um especialista em engenharia de prompts. "
                    "Responda somente JSON válido."
                ),
            },
            {"role": "user", "content": generation_prompt},
        ],
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        reasoning_effort="low",
        timeout=REQUEST_TIMEOUT_SECONDS,
        response_format={"type": "json_object"},
    )
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise RuntimeError("OpenAI não retornou opções para regeneração de prompt.")
    message = getattr(choices[0], "message", None)
    if not message:
        raise RuntimeError("OpenAI retornou mensagem vazia na regeneração de prompt.")
    raw_content = _extract_response_text(response, message)
    if not raw_content:
        raise RuntimeError("OpenAI retornou conteúdo vazio na regeneração de prompt.")

    try:
        payload = json.loads(raw_content.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError("JSON inválido na regeneração de prompt.") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("Payload de regeneração deve ser objeto JSON.")

    description_command = payload.get("description_command")
    prompt = payload.get("prompt")
    if not isinstance(description_command, str) or not description_command.strip():
        raise RuntimeError("Campo 'description_command' inválido na regeneração.")
    if not isinstance(prompt, str) or not prompt.strip():
        raise RuntimeError("Campo 'prompt' inválido na regeneração.")

    return description_command.strip(), prompt.strip()


def evaluate_prompt_content(
    *,
    component: PromptComponent,
    description_command: str,
    prompt_content: str,
) -> tuple[float, str, str]:
    model = _get_openai_model()
    client = _get_openai_client()

    evaluation_prompt = (
        "Avalie a qualidade de um prompt para operação em produção.\n"
        "Retorne SOMENTE JSON válido no formato:\n"
        "{\n"
        '  "score": 0.0,\n'
        '  "analysis": "análise curta",\n'
        '  "improvement": "melhoria curta para próxima rodada"\n'
        "}\n\n"
        "Regras:\n"
        "- score deve ser float de 0 a 10.\n"
        "- improvement deve ser objetivo e acionável.\n"
        "- Penalize ambiguidade, redundância e falta de executabilidade.\n"
        "- Penalize se a descrição não estiver em tom de comando.\n\n"
        f"COMPONENTE: {component.key}\n"
        f"DESCRIÇÃO (COMANDO): {description_command}\n"
        f"PROMPT AVALIADO:\n{prompt_content}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Você avalia qualidade de prompts. " "Responda somente JSON válido."
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
        raise RuntimeError("OpenAI não retornou opções para avaliação de prompt.")
    message = getattr(choices[0], "message", None)
    if not message:
        raise RuntimeError("OpenAI retornou mensagem vazia na avaliação de prompt.")
    raw_content = _extract_response_text(response, message)
    if not raw_content:
        raise RuntimeError("OpenAI retornou conteúdo vazio na avaliação de prompt.")

    try:
        payload = json.loads(raw_content.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError("JSON inválido na avaliação de prompt.") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("Payload de avaliação deve ser objeto JSON.")

    score_raw = payload.get("score")
    analysis_raw = payload.get("analysis")
    improvement_raw = payload.get("improvement")

    if not isinstance(score_raw, (int, float)):
        raise RuntimeError("Campo 'score' inválido na avaliação.")
    score = float(score_raw)
    if score < 0 or score > 10:
        raise RuntimeError("Campo 'score' deve estar entre 0 e 10.")

    if not isinstance(analysis_raw, str) or not analysis_raw.strip():
        raise RuntimeError("Campo 'analysis' inválido na avaliação.")
    if not isinstance(improvement_raw, str) or not improvement_raw.strip():
        raise RuntimeError("Campo 'improvement' inválido na avaliação.")

    return score, analysis_raw.strip(), improvement_raw.strip()
