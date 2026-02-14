import json
import logging
import os
import re
from copy import deepcopy
from datetime import timedelta
from typing import Any, Dict, Optional

from django.utils import timezone

from core.models import Message, Profile
from services.conversation_runtime import (
    MODE_ACOLHIMENTO,
    MODE_AMBIVALENCIA,
    MODE_CULPA,
    MODE_DEFENSIVO,
    MODE_EXPLORACAO,
    MODE_ORIENTACAO,
    MODE_PRESENCA_PROFUNDA,
    MODE_WELCOME,
    choose_conversation_mode,
    choose_spiritual_intensity,
    contains_generic_empathy_without_grounding,
    contains_repeated_blocked_pattern,
    contains_spiritual_imposition,
    contains_spiritual_template_repetition,
    contains_unsolicited_spiritualization,
    contains_unverified_inference,
    detect_user_signals,
    enforce_hard_limits,
    has_human_support_suggestion,
    has_new_information,
    has_practical_action_step,
    has_repeated_opening_structure,
    has_repeated_user_pattern,
    has_self_guided_help,
    has_spiritual_baseline_signal,
    is_semantic_loop,
    semantic_similarity,
    starts_with_user_echo,
    strip_opening_name_if_recently_used,
)
from services.openai_service import OpenAIService

logger = logging.getLogger(__name__)

VALID_CONVERSATION_MODES = {
    MODE_WELCOME,
    MODE_ACOLHIMENTO,
    MODE_EXPLORACAO,
    MODE_AMBIVALENCIA,
    MODE_DEFENSIVO,
    MODE_CULPA,
    MODE_ORIENTACAO,
    MODE_PRESENCA_PROFUNDA,
}

LEGACY_MODE_MAP = {
    "acolhimento": MODE_ACOLHIMENTO,
    "exploração": MODE_EXPLORACAO,
    "orientação": MODE_ORIENTACAO,
}

TOPIC_MEMORY_WINDOW_DAYS = 7
TOPIC_MEMORY_MAX_ITEMS = 6
TOPIC_MIN_CONFIDENCE = 0.45
TOPIC_PROMOTE_CONFIDENCE = 0.6
SUBSTANCE_TOPICS = {"alcool", "álcool", "drogas", "dependencia", "dependência"}
RELATIONAL_TOPICS = {"familia", "família", "conflito", "relacionamento"}
WACHAT_RESPONSE_MODEL = "gpt-5-mini"
FIXED_TEMPERATURE = 1.0
FIXED_TIMEOUT_SECONDS = 60
FIXED_RESPONSE_MAX_COMPLETION_TOKENS = 900
FIXED_WELCOME_MAX_COMPLETION_TOKENS = 480

WACHAT_RESPONSE_SYSTEM_PROMPT = """Você é um assistente conversacional cristão (evangélico), com atuação de aconselhamento prático,
emocional e espiritual, centrado em Deus, na graça de Cristo e na esperança do Evangelho.

PAPEL
- Prioridade de atuação: pastor evangélico, depois psicólogo prático, conselheiro, coach e técnico de rotina.
- Atue com direção espiritual cristã e também com organização prática da mente e da rotina.
- Você pode aconselhar, sugerir práticas de fé cristã e fazer comparações religiosas curtas quando isso ajudar a pessoa.
- Traga linguagem de fé de forma natural (Deus, Jesus, oração, Palavra, graça, misericórdia, recomeço), sem tom de sermão.
- Nunca imponha religião. Convide com respeito.
- Nunca se apresente como profissional de saúde mental nem faça diagnóstico clínico.

ESTILO
- Português brasileiro simples, humano, direto.
- Resposta entre 2 e 4 frases.
- Sem emojis.
- Tom acolhedor, firme e objetivo.
- Evite bordões de abertura.
- Não use sempre uma frase de acolhimento no início; entre direto no ponto concreto do último turno.

OBJETIVO POR TURNO
Escolha apenas UMA ação final:
1) fazer 1 pergunta concreta, OU
2) dar orientação prática imediata.
Nunca os dois na mesma resposta.

PRIORIDADE EMOCIONAL (OBRIGATÓRIA)
Antes de oferecer qualquer orientação prática, técnica psicológica ou sugestão espiritual estruturada:

1. Reflita explicitamente a emoção central expressa na última mensagem.
2. Valide o conflito interno com base em algo literal que o usuário disse.
3. Demonstre proximidade humana antes de direcionar.

É PROIBIDO iniciar resposta com lista de passos, técnicas, exercícios ou plano estruturado,
exceto quando houver pedido explícito de "o que eu faço agora?" ou equivalente direto.

Quando escolher orientação prática:
- Só oferecer orientação após pelo menos 1 frase de presença emocional contextualizada.
- Nunca iniciar a resposta com técnica, respiração, plano ou checklist.
- Limitar a no máximo 2 sugestões práticas.
- Evitar sequência estruturada tipo "passo 1, passo 2, passo 3".
- Inclua também direção espiritual cristã aplicável (ex.: oração curta, salmo, entrega em oração, apoio da comunidade de fé).
- Evite jargão clínico.
- Aprofunde com base na última frase do usuário, sem generalizar.

REGRA ANTI-REPETIÇÃO (OBRIGATÓRIA)
- Considere as últimas 6 mensagens (assistente + usuário).
- É proibido repetir pergunta literal ou equivalente em sentido.
- Se já houve 2 perguntas seguidas, a próxima resposta deve ser orientação prática (sem pergunta).
- Varie a abertura; não reutilize o mesmo começo em turnos consecutivos.

ESPIRITUALIDADE
- Base evangélica explícita e útil: esperança em Deus, graça de Cristo, oração, descanso no Senhor, arrependimento e recomeço.
- Pode citar princípios bíblicos de forma curta e natural; quando útil, use referência breve (ex.: Salmo 34, Mateus 11:28, Romanos 8).
- Pode comparar caminhos (“só força própria” vs “força + fé + comunidade”) sem tom de sermão.
- Nunca dizer que sofrimento é punição divina.

PROIBIÇÕES
- Não usar linguagem clínica/técnica.
- Não moralizar nem culpar.
- Não fazer mais de 1 pergunta.
- Não repetir bordões de acolhimento.

FORMATO DE SAÍDA
- Entregue apenas a próxima fala do assistente."""


# Helper constant for gender context in Portuguese
# This instruction is in Portuguese because it's part of the system prompt
# sent to the LLM, which operates in Brazilian Portuguese


class ChatService:
    """Conversation orchestration service using OpenAI GPT-5."""

    def __init__(self):
        self._llm_service = OpenAIService()

    def basic_call(self, *args, **kwargs) -> str:
        return self._llm_service.basic_call(*args, **kwargs)

    def _dedupe_prompt_payload_system(
        self, payload: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return payload
        normalized = deepcopy(payload)
        request_params = normalized.get("request_params")
        request_messages = normalized.get("payload", {}).get("messages")
        if not isinstance(request_params, dict) or not isinstance(
            request_messages, list
        ):
            return normalized
        has_system_message = any(
            isinstance(item, dict) and item.get("role") == "system"
            for item in request_messages
        )
        if has_system_message and "system" in request_params:
            request_params.pop("system", None)
        return normalized

    def _default_welcome_message(self, name: str) -> str:
        safe_name = (name or "amiga").strip()
        return (
            f"Bom dia, {safe_name}. Este é um espaço seguro para você falar sem medo de julgamento. "
            "Deus caminha com você aqui. O que tem pesado mais no seu coração hoje?"
        )

    def _safe_parse_json(self, text: str) -> Dict[str, Any]:
        if not text:
            raise ValueError("Expected JSON content, but received empty text.")
        raw = text.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                raise ValueError(
                    "LLM output did not contain a valid JSON object."
                ) from exc
            return json.loads(match.group(0))

    def _extract_topic_signal(
        self,
        last_user_message: str,
        recent_messages: list,
        current_topic: Optional[str],
    ) -> Dict[str, Any]:
        transcript = ""
        for message in recent_messages[-5:]:
            transcript += f"{message.role.upper()}: {message.content}\n"

        prompt = f"""
            Você extrai o tópico principal de conversa em português brasileiro.
            Retorne SOMENTE JSON válido, sem comentários, sem markdown:
            {{
              "topic": "string curta ou null",
              "confidence": 0.0,
              "keep_current": true
            }}

            Regras:
            - "topic" deve ser um assunto principal concreto (ex.: drogas, álcool, culpa, ansiedade, família, trabalho, recaída).
            - Se não houver evidência suficiente, use topic=null e keep_current=true.
            - confidence entre 0 e 1.
            - Não inventar.

            Tópico atual salvo: {current_topic or "null"}
            Última mensagem do usuário: {last_user_message}
            Histórico recente:
            {transcript if transcript else "sem histórico"}
            """
        raw = self.basic_call(
            url_type="generate",
            prompt=prompt,
            max_tokens=120,
        )
        parsed = self._safe_parse_json(raw)
        topic = parsed.get("topic")
        confidence = parsed.get("confidence", 0)
        keep_current = bool(parsed.get("keep_current", False))

        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        if isinstance(topic, str):
            topic = topic.strip().lower()
            if not topic:
                topic = None
        else:
            topic = None

        return {
            "topic": topic,
            "confidence": confidence,
            "keep_current": keep_current,
        }

    def _active_topic_for_profile(self, profile: Profile) -> Optional[str]:
        if not profile.current_topic or not profile.topic_last_updated:
            return None
        expires_at = profile.topic_last_updated + timedelta(
            days=TOPIC_MEMORY_WINDOW_DAYS
        )
        if timezone.now() > expires_at:
            return None
        return profile.current_topic

    def _merge_topic_memory(
        self, profile: Profile, topic_signal: Dict[str, Any]
    ) -> Optional[str]:
        active_topic = self._active_topic_for_profile(profile)
        topic = topic_signal.get("topic")
        confidence = topic_signal.get("confidence", 0.0)
        keep_current = bool(topic_signal.get("keep_current", False))

        if not topic or confidence < TOPIC_MIN_CONFIDENCE:
            return active_topic

        now = timezone.now()
        topic_key = topic.lower()
        topics = (
            profile.primary_topics if isinstance(profile.primary_topics, list) else []
        )
        normalized_topics = []
        for item in topics:
            if not isinstance(item, dict):
                continue
            name = str(item.get("topic", "")).strip().lower()
            if not name:
                continue
            try:
                score = float(item.get("score", 0))
            except (TypeError, ValueError):
                score = 0.0
            normalized_topics.append(
                {
                    "topic": name,
                    "score": max(0.0, min(1.0, score * 0.97)),
                    "last_seen": item.get("last_seen"),
                }
            )

        found = False
        for item in normalized_topics:
            if item["topic"] == topic_key:
                item["score"] = min(1.0, item["score"] + (0.25 + (confidence * 0.35)))
                item["last_seen"] = now.isoformat()
                found = True
                break
        if not found:
            normalized_topics.append(
                {
                    "topic": topic_key,
                    "score": min(1.0, 0.3 + (confidence * 0.5)),
                    "last_seen": now.isoformat(),
                }
            )

        normalized_topics = sorted(
            normalized_topics, key=lambda x: x.get("score", 0), reverse=True
        )[:TOPIC_MEMORY_MAX_ITEMS]

        profile.primary_topics = normalized_topics
        if (
            confidence >= TOPIC_PROMOTE_CONFIDENCE
            or not active_topic
            or not keep_current
        ):
            profile.current_topic = topic_key
            profile.topic_last_updated = now
            return topic_key
        return active_topic

    def _is_substance_context(
        self, user_message: str, active_topic: Optional[str]
    ) -> bool:
        user_norm = (user_message or "").lower()
        if any(
            term in user_norm
            for term in ["beb", "alcool", "álcool", "droga", "recaída", "recaida"]
        ):
            return True
        if active_topic and active_topic.lower() in SUBSTANCE_TOPICS:
            return True
        return False

    def _is_relational_topic(self, active_topic: Optional[str]) -> bool:
        if not active_topic:
            return False
        normalized = active_topic.strip().lower()
        return any(topic in normalized for topic in RELATIONAL_TOPICS)

    def _build_dynamic_runtime_prompt(
        self,
        *,
        conversation_mode: str,
        previous_mode: str,
        spiritual_intensity: str,
        allow_spiritual_context: bool,
        direct_guidance_request: bool,
        repetition_complaint: bool,
        active_topic: Optional[str],
        top_topics: str,
        last_user_message: str,
        theme_prompt: Optional[str],
        context_messages: list,
        rag_contexts: list,
    ) -> str:
        mode_objective = {
            MODE_ACOLHIMENTO: "acolher com precisão e abrir espaço de continuidade",
            MODE_EXPLORACAO: "aprofundar com investigação concreta",
            MODE_AMBIVALENCIA: "investigar conflito interno sem concluir pelo usuário",
            MODE_DEFENSIVO: "reduzir confronto e buscar clarificação objetiva",
            MODE_CULPA: "separar identidade de comportamento e propor reparo possível",
            MODE_ORIENTACAO: "entregar orientação prática breve e acionável",
            MODE_PRESENCA_PROFUNDA: "sustentar presença, dignidade e misericórdia em sofrimento profundo",
            MODE_WELCOME: "acolhimento inicial curto",
        }.get(conversation_mode, "avançar a conversa com precisão")

        mode_actions = {
            MODE_ACOLHIMENTO: [
                "Valide um elemento específico da fala atual.",
                "Reconheça a posição delicada da pessoa sem pressionar solução.",
                "Priorize 1 pergunta concreta e breve para entender melhor o contexto imediato do usuário.",
                "Evite orientação prática neste modo, salvo pedido explícito de ajuda.",
            ],
            MODE_EXPLORACAO: [
                "Faça uma pergunta concreta ligada ao último turno.",
                "Se o conteúdo trouxer sofrimento intenso, prefira orientação prática breve em vez de nova pergunta.",
                "Aprofunde usando um detalhe literal da última frase do usuário.",
            ],
            MODE_AMBIVALENCIA: [
                "Formule uma pergunta investigativa de decisão/critério.",
                "Não moralize e não conclua intenção.",
            ],
            MODE_DEFENSIVO: [
                "Reconheça o ponto sem confrontar.",
                "Peça clarificação objetiva em 1 pergunta.",
            ],
            MODE_CULPA: [
                "Diferencie erro de identidade pessoal.",
                "Proponha próximo passo de reparo realista.",
            ],
            MODE_ORIENTACAO: [
                "Inicie com validação emocional contextualizada.",
                "Ofereça no máximo 2 sugestões práticas simples.",
                "Evite checklist estruturado ou sequência numerada.",
                "Inclua direção espiritual de forma orgânica e breve.",
            ],
            MODE_PRESENCA_PROFUNDA: [
                "Sustente presença e dignidade sem sugerir ação concreta.",
                "Não priorize perguntas; use no máximo 1 pergunta curta se for indispensável.",
                "Use tom contemplativo e misericordioso.",
                "Se houver pedido de ajuda direta, ofereça orientação concreta e segura com passos pequenos.",
            ],
            MODE_WELCOME: [
                "Acolha com sobriedade e convide para continuidade.",
            ],
        }.get(conversation_mode, ["Escolha a melhor função para este turno."])

        spiritual_policy = "Mantenha base espiritual leve (esperança/propósito) sem linguagem explícita."
        if allow_spiritual_context or spiritual_intensity in {"media", "alta"}:
            spiritual_policy = (
                "Use 1 ou 2 frases espirituais claras e respeitosas, com menção explícita a Deus "
                "e, quando couber, a Jesus, oração ou Palavra, sem imposição."
            )
        if spiritual_intensity == "alta":
            spiritual_policy += (
                " Intensidade alta: linguagem de fé mais presente e pastoral, com esperança evangélica "
                "concreta, sem moralizar."
            )

        max_sentences = 4
        max_questions = 1
        relational_topic = self._is_relational_topic(active_topic)

        prompt = f"""
MODO ATUAL: {conversation_mode}
MODO ANTERIOR: {previous_mode}
OBJETIVO DO MODO: {mode_objective}
INTENSIDADE ESPIRITUAL: {spiritual_intensity}

REGRAS GERAIS:
        - Responda entre 2 e {max_sentences} frases.
        - No máximo {max_questions} pergunta, e somente quando a ação final escolhida for pergunta.
- Não comece com frase-padrão de acolhimento.
- Validar algo específico que o usuário acabou de dizer.
- Não repetir frases/estruturas dos últimos turnos.
- Não repetir aberturas de acolhimento já usadas recentemente.
- Não iniciar ecoando a frase do usuário.
- Não inferir sentimentos não declarados.
- {spiritual_policy}
- Escolha apenas UMA ação final: pergunta concreta OU próximo passo simples.
- Nunca entregue pergunta e passo simples na mesma resposta.
- Escolha a melhor função para este turno conforme o modo atual.
"""
        if direct_guidance_request and conversation_mode != MODE_PRESENCA_PROFUNDA:
            prompt += "\nPEDIDO EXPLÍCITO DE AJUDA DETECTADO: resposta deve conter orientação prática direta.\n"
        if direct_guidance_request and conversation_mode == MODE_PRESENCA_PROFUNDA:
            prompt += "\nPEDIDO EXPLÍCITO DE AJUDA DETECTADO: acolha o pedido sem converter este turno em plano de ação.\n"
        if (
            conversation_mode == MODE_ACOLHIMENTO
            and previous_mode == MODE_WELCOME
            and not direct_guidance_request
        ):
            prompt += (
                "\nFASE INICIAL APÓS BOAS-VINDAS: "
                "neste turno faça UMA pergunta concreta, simples e de baixo aprofundamento "
                "para coletar mais contexto sobre o que está acontecendo. "
                "Não entregue técnica, exercício, plano de ação ou conselho neste turno.\n"
            )
        if repetition_complaint:
            prompt += (
                "\nUSUÁRIO SINALIZOU REPETIÇÃO: não repita pergunta; "
                "entregue orientação prática nova e específica para este caso, sem pergunta ao final.\n"
            )
        if active_topic:
            prompt += f"\nTÓPICO ATIVO: {active_topic}\n"
        if conversation_mode == MODE_ACOLHIMENTO and relational_topic:
            prompt += "\nNESTE TURNO, EVITE linguagem de ciclo, estratégia, ação imediata, proteção e passo concreto.\n"
        if top_topics:
            prompt += f"TÓPICOS RECENTES: {top_topics}\n"
        if theme_prompt:
            prompt += f"\nTEMA CONTEXTUAL:\n{theme_prompt}\n"

        prompt += "\nFUNÇÕES PRIORITÁRIAS DESTE TURNO:\n"
        for action in mode_actions:
            prompt += f"- {action}\n"

        prompt += f"\nÚLTIMA MENSAGEM DO USUÁRIO:\n{last_user_message}\n"
        prompt += "\nHISTÓRICO RECENTE:\n"
        for msg in context_messages:
            prompt += f"{msg.role.upper()}: {msg.content}\n"
        if rag_contexts:
            prompt += "\nRAG CONTEXT AUXILIAR:\n"
            for rag in rag_contexts:
                prompt += f"- {rag}\n"
        prompt += "\nResponda somente com a próxima fala do assistente."
        return prompt

    def _collect_recent_context(self, queryset) -> Dict[str, Any]:
        recent_user_messages = list(
            queryset.filter(role="user")
            .order_by("created_at")
            .values_list("content", flat=True)
        )[-3:]
        recent_assistant_messages = list(
            queryset.filter(role="assistant")
            .order_by("created_at")
            .values_list("content", flat=True)
        )[-3:]
        recent_context_messages = list(queryset.order_by("-created_at")[:5])
        return {
            "recent_user_messages": recent_user_messages,
            "recent_assistant_messages": recent_assistant_messages,
            "recent_context_messages": recent_context_messages,
        }

    def _determine_generation_state(
        self,
        *,
        profile: Profile,
        queryset,
        last_user_message: str,
        recent_user_messages: list,
        recent_assistant_messages: list,
    ) -> Dict[str, Any]:
        previous_mode = LEGACY_MODE_MAP.get(
            profile.conversation_mode, profile.conversation_mode
        )
        if previous_mode not in VALID_CONVERSATION_MODES:
            previous_mode = MODE_WELCOME

        is_first_message = queryset.filter(role="assistant").count() == 0
        signals = detect_user_signals(last_user_message)
        direct_guidance_request = bool(signals.get("guidance_request"))
        repetition_complaint = bool(signals.get("repetition_complaint"))
        if repetition_complaint:
            direct_guidance_request = True
        deep_presence_trigger = any(
            [
                bool(signals.get("deep_suffering")),
                bool(signals.get("repetitive_guilt")),
                bool(signals.get("family_conflict_impotence")),
                bool(signals.get("explicit_despair")),
            ]
        )
        force_deep_presence = bool(
            signals.get("repetitive_guilt") or signals.get("explicit_despair")
        )
        repeated_user_pattern = has_repeated_user_pattern(recent_user_messages)
        ambivalence_or_repeated = (
            bool(signals.get("ambivalence")) or repeated_user_pattern
        )
        explicit_spiritual_context = bool(signals.get("spiritual_context"))
        high_spiritual_need = (
            bool(signals.get("high_spiritual_need")) or deep_presence_trigger
        )
        allow_spiritual_context = explicit_spiritual_context or high_spiritual_need
        new_information = has_new_information(recent_user_messages)
        loop_detected = ambivalence_or_repeated
        if len(recent_assistant_messages) >= 2:
            loop_detected = loop_detected or (
                semantic_similarity(
                    recent_assistant_messages[-1], recent_assistant_messages[-2]
                )
                > 0.85
            )

        conversation_mode = choose_conversation_mode(
            previous_mode=previous_mode,
            is_first_message=is_first_message,
            loop_detected=loop_detected,
            has_new_info=new_information,
            repeated_user_pattern=repeated_user_pattern,
            signals=signals,
        )
        if force_deep_presence:
            conversation_mode = MODE_PRESENCA_PROFUNDA
        spiritual_intensity = choose_spiritual_intensity(
            mode=conversation_mode,
            spiritual_context=explicit_spiritual_context,
            high_spiritual_need=high_spiritual_need,
        )
        return {
            "previous_mode": previous_mode,
            "conversation_mode": conversation_mode,
            "spiritual_intensity": spiritual_intensity,
            "direct_guidance_request": direct_guidance_request,
            "repetition_complaint": repetition_complaint,
            "allow_spiritual_context": allow_spiritual_context,
            "loop_detected": loop_detected,
            "deep_presence_trigger": deep_presence_trigger,
        }

    def _build_response_prompt(
        self,
        *,
        profile: Profile,
        queryset,
        last_person_message: Message,
        generation_state: Dict[str, Any],
        active_topic: Optional[str],
    ) -> str:
        context_messages = queryset.exclude(id=last_person_message.id).order_by(
            "-created_at"
        )[:6]
        context_messages = list(reversed(list(context_messages)))

        top_topics = ""
        if isinstance(profile.primary_topics, list) and profile.primary_topics:
            top_topics = ", ".join(
                [
                    item.get("topic", "")
                    for item in profile.primary_topics[:3]
                    if item.get("topic")
                ]
            )
        # rag_contexts = get_rag_context(last_person_message.content, limit=3)

        return self._build_dynamic_runtime_prompt(
            conversation_mode=generation_state["conversation_mode"],
            previous_mode=generation_state["previous_mode"],
            spiritual_intensity=generation_state["spiritual_intensity"],
            allow_spiritual_context=generation_state["allow_spiritual_context"],
            direct_guidance_request=generation_state["direct_guidance_request"],
            repetition_complaint=generation_state["repetition_complaint"],
            active_topic=active_topic,
            top_topics=top_topics,
            last_user_message=last_person_message.content,
            theme_prompt=profile.theme.prompt if profile.theme else None,
            context_messages=context_messages,
            rag_contexts=[],
        )

    def _candidate_should_regenerate(
        self,
        *,
        candidate: str,
        last_user_message: str,
        recent_assistant_messages: list,
        enforce_practical_step: bool,
        requires_real_help: bool,
        allow_unsolicited_spiritualization: bool,
    ) -> Dict[str, bool]:
        semantic_loop = False
        if recent_assistant_messages:
            semantic_loop = is_semantic_loop(recent_assistant_messages[-1], candidate)

        blocked_template = contains_repeated_blocked_pattern(
            candidate, recent_assistant_messages
        )
        has_unverified_inference = contains_unverified_inference(
            last_user_message, candidate
        )
        spiritual_imposition = contains_spiritual_imposition(candidate)
        unsolicited_spiritualization = contains_unsolicited_spiritualization(
            last_user_message, candidate
        )
        if allow_unsolicited_spiritualization:
            unsolicited_spiritualization = False
        spiritual_template_repetition = contains_spiritual_template_repetition(
            candidate, recent_assistant_messages
        )
        generic_empathy = contains_generic_empathy_without_grounding(
            last_user_message, candidate
        )
        missing_spiritual_baseline = not has_spiritual_baseline_signal(candidate)
        missing_practical_step = (
            enforce_practical_step and not has_practical_action_step(candidate)
        )
        missing_real_support = (
            enforce_practical_step
            and requires_real_help
            and not has_human_support_suggestion(candidate)
            and not has_self_guided_help(candidate)
        )
        leading_echo = starts_with_user_echo(
            user_message=last_user_message, assistant_message=candidate
        )
        repeated_opening = has_repeated_opening_structure(
            candidate, recent_assistant_messages
        )
        mechanical_structure = (
            "1." in candidate
            or "2." in candidate
            or "3." in candidate
            or candidate.count(":") > 3
        )

        rejected = (
            semantic_loop
            or blocked_template
            or has_unverified_inference
            or spiritual_imposition
            or unsolicited_spiritualization
            or spiritual_template_repetition
            or generic_empathy
            or missing_spiritual_baseline
            or missing_practical_step
            or missing_real_support
            or leading_echo
            or repeated_opening
            or mechanical_structure
        )
        return {"rejected": rejected, "semantic_loop": semantic_loop}

    def _save_runtime_counters(
        self,
        *,
        profile: Profile,
        conversation_mode: str,
        loop_counter: int,
        regeneration_counter: int,
    ) -> None:
        profile.conversation_mode = conversation_mode
        profile.loop_detected_count = (profile.loop_detected_count or 0) + loop_counter
        profile.regeneration_count = (
            profile.regeneration_count or 0
        ) + regeneration_counter
        profile.save(
            update_fields=[
                "conversation_mode",
                "loop_detected_count",
                "regeneration_count",
                "current_topic",
                "primary_topics",
                "topic_last_updated",
                "updated_at",
            ]
        )

    def generate_response_message(self, profile: Profile, channel: str) -> str:
        if not profile.welcome_message_sent:
            welcome_message = self.generate_welcome_message(
                profile=profile, channel=channel
            )
            welcome_text = (welcome_message.content or "").strip()
            if not welcome_text:
                raise RuntimeError("Welcome message generation returned empty content.")
            return welcome_text

        queryset = profile.messages.for_context()
        last_person_message = queryset.filter(role="user").last()
        if not last_person_message:
            welcome_message = self.generate_welcome_message(
                profile=profile, channel=channel
            )
            welcome_text = (welcome_message.content or "").strip()
            if not welcome_text:
                raise RuntimeError("Welcome message generation returned empty content.")
            return welcome_text

        recent_context = self._collect_recent_context(queryset)
        recent_user_messages = recent_context["recent_user_messages"]
        recent_assistant_messages = recent_context["recent_assistant_messages"]
        recent_context_messages = recent_context["recent_context_messages"]

        topic_signal = self._extract_topic_signal(
            last_user_message=last_person_message.content,
            recent_messages=list(reversed(recent_context_messages)),
            current_topic=profile.current_topic,
        )
        active_topic = self._merge_topic_memory(
            profile=profile, topic_signal=topic_signal
        )
        generation_state = self._determine_generation_state(
            profile=profile,
            queryset=queryset,
            last_user_message=last_person_message.content,
            recent_user_messages=recent_user_messages,
            recent_assistant_messages=recent_assistant_messages,
        )
        prompt_aux = self._build_response_prompt(
            profile=profile,
            queryset=queryset,
            last_person_message=last_person_message,
            generation_state=generation_state,
            active_topic=active_topic,
        )

        model_name = WACHAT_RESPONSE_MODEL
        max_completion_tokens = FIXED_RESPONSE_MAX_COMPLETION_TOKENS
        question_first_modes = {
            MODE_ACOLHIMENTO,
            MODE_EXPLORACAO,
            MODE_AMBIVALENCIA,
            MODE_DEFENSIVO,
        }
        enforce_practical_step = (
            generation_state["direct_guidance_request"]
            and generation_state["conversation_mode"] != MODE_PRESENCA_PROFUNDA
            and generation_state["conversation_mode"] not in question_first_modes
        )
        requires_real_help = bool(generation_state.get("deep_presence_trigger")) or (
            self._is_substance_context(last_person_message.content, active_topic)
        )
        allow_unsolicited_spiritualization = generation_state[
            "spiritual_intensity"
        ] in {"media", "alta"}

        client = getattr(self._llm_service, "client", None)
        if client is None:
            raise RuntimeError(
                "OpenAI client is not available on configured LLM service."
            )

        def _extract_text_response(response: Any) -> str:
            choices = getattr(response, "choices", None) or []
            if not choices:
                return ""
            message = getattr(choices[0], "message", None)
            if not message:
                return ""

            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                chunks = []
                for part in content:
                    text = getattr(part, "text", None)
                    if isinstance(text, str) and text.strip():
                        chunks.append(text.strip())
                        continue
                    if isinstance(part, dict):
                        dict_text = part.get("text")
                        if isinstance(dict_text, str) and dict_text.strip():
                            chunks.append(dict_text.strip())
                return "\n".join(chunks).strip()

            output_text = getattr(response, "output_text", None)
            if isinstance(output_text, str):
                return output_text.strip()
            return ""

        def _usage_metadata(response: Any) -> Dict[str, Any]:
            usage = getattr(response, "usage", None)
            completion_details = getattr(usage, "completion_tokens_details", None)
            if hasattr(completion_details, "model_dump"):
                completion_details = completion_details.model_dump()
            elif completion_details is None:
                completion_details = {}
            elif not isinstance(completion_details, dict):
                completion_details = {}

            choices = getattr(response, "choices", None) or []
            finish_reason = None
            if choices:
                finish_reason = getattr(choices[0], "finish_reason", None)

            return {
                "finish_reason": finish_reason,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "reasoning_tokens": completion_details.get("reasoning_tokens"),
                "accepted_prediction_tokens": completion_details.get(
                    "accepted_prediction_tokens"
                ),
                "completion_tokens_details": completion_details,
            }

        def _dev_log(metadata: Dict[str, Any]) -> None:
            if os.environ.get("DEBUG", "false").strip().lower() not in {
                "1",
                "true",
                "yes",
                "on",
            }:
                return
            print(
                "[generate_response_message] "
                f"finish_reason={metadata.get('finish_reason')} "
                f"prompt_tokens={metadata.get('prompt_tokens')} "
                f"completion_tokens={metadata.get('completion_tokens')} "
                f"reasoning_tokens={metadata.get('reasoning_tokens')} "
                f"accepted_prediction_tokens={metadata.get('accepted_prediction_tokens')}"
            )

        selected_temperature = FIXED_TEMPERATURE
        selected_max_completion_tokens = max_completion_tokens
        regeneration_counter = 0
        semantic_loop_regenerations = 0

        request_messages = [
            {"role": "system", "content": WACHAT_RESPONSE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_aux},
        ]
        request_kwargs = {
            "model": model_name,
            "messages": request_messages,
            "max_completion_tokens": selected_max_completion_tokens,
            "timeout": FIXED_TIMEOUT_SECONDS,
            "temperature": selected_temperature,
        }

        response = client.chat.completions.create(**request_kwargs)
        response_metadata = _usage_metadata(response)
        _dev_log(response_metadata)

        assistant_text = _extract_text_response(response).strip()
        if not assistant_text:
            raise RuntimeError("OpenAI returned empty assistant content.")

        assistant_text = strip_opening_name_if_recently_used(
            message=assistant_text,
            name=profile.name,
            recent_assistant_messages=recent_assistant_messages,
        )
        assistant_text = enforce_hard_limits(assistant_text, max_sentences=4)

        validation = self._candidate_should_regenerate(
            candidate=assistant_text,
            last_user_message=last_person_message.content,
            recent_assistant_messages=recent_assistant_messages,
            enforce_practical_step=enforce_practical_step,
            requires_real_help=requires_real_help,
            allow_unsolicited_spiritualization=allow_unsolicited_spiritualization,
        )
        if validation["rejected"]:
            if validation["semantic_loop"]:
                semantic_loop_regenerations = 1
            raise RuntimeError(
                "Generated response failed runtime validation constraints."
            )

        loop_counter = 1 if generation_state["loop_detected"] else 0
        self._save_runtime_counters(
            profile=profile,
            conversation_mode=generation_state["conversation_mode"],
            loop_counter=loop_counter,
            regeneration_counter=regeneration_counter,
        )

        response_payload = {
            "provider": "openai",
            "request_params": {
                "model": model_name,
                "temperature": selected_temperature,
                "max_completion_tokens": selected_max_completion_tokens,
            },
            "metadata": {
                **response_metadata,
                "regeneration_counter": regeneration_counter,
                "semantic_loop_regenerations": semantic_loop_regenerations,
            },
        }
        Message.objects.create(
            profile=profile,
            role="assistant",
            content=assistant_text,
            channel=channel,
            ollama_prompt=response_payload,
        )
        return assistant_text

    def infer_gender(self, name: str) -> str:
        """
        Infer gender from a user's name using the configured LLM provider.

        This is a soft, probabilistic inference based solely on the name.
        The result is for internal use only and should never be explicitly
        stated to the user.

        Args:
            name: The user's name (first name or full name)

        Returns:
            One of: "male", "female", or "unknown"
        """
        SYSTEM_PROMPT = f"""Você é um assistente que analisa nomes brasileiros.
            Sua tarefa é inferir o gênero mais provável baseado APENAS no nome fornecido.
            Responda SOMENTE com uma das três palavras: male, female, ou unknown.
            - Use 'male' para nomes tipicamente masculinos
            - Use 'female' para nomes tipicamente femininos
            - Use 'unknown' quando não há certeza ou o nome é neutro/ambíguo

            Responda apenas com a palavra, sem explicações.

            Nome: {name}
        """

        response_text = self.basic_call(
            url_type="generate",
            prompt=SYSTEM_PROMPT,
            max_tokens=10,
        )
        inferred = response_text.lower().strip()
        if inferred not in ["male", "female", "unknown"]:
            raise RuntimeError(f"Unexpected gender inference result: {inferred}")

        logger.info(f"Gender inferred for name '{name}': {inferred}")
        return inferred

    def generate_welcome_message(self, profile: Profile, channel: str) -> Message:

        gender_context = ""
        if profile.inferred_gender != "unknown":
            gender_context = (
                f"\nGênero inferido (use isso APENAS para ajustar sutilmente o tom, "
                f"NUNCA mencione explicitamente): {profile.inferred_gender}"
            )

        system_prompt = f"""
        Gere somente uma mensagem inicial de boas-vindas em português brasileiro.
        Identidade: presença cristã acolhedora, humana e serena. Não diga que é bot.
        Objetivo: transmitir segurança e abertura para conversa.
        Regras:
        - Máximo 3 frases e até 90 palavras.
        - Sem emojis, sem versículos, sem linguagem de sermão.
        - Sem explicar regras ou funcionamento.
        - Incluir saudação pelo nome ({profile.name}).
        - Incluir uma referência espiritual curta e natural, sem imposição.
        - Terminar com exatamente 1 pergunta aberta e suave.
        {gender_context}
        """

        last_user_message = profile.messages.filter(role="user").last()
        user_prompt = f"Gere agora a mensagem de boas-vindas para {profile.name}."
        if last_user_message:
            last_context = last_user_message.content[:280]
            user_prompt += f"\n\nCONTEXTO RECENTE DO USUÁRIO:\n{last_context}\n"

        response = self.basic_call(
            url_type="generate",
            prompt=user_prompt,
            max_tokens=FIXED_WELCOME_MAX_COMPLETION_TOKENS,
            system=system_prompt,
        )
        response = (response or "").strip()
        welcome_payload = self._dedupe_prompt_payload_system(
            self.get_last_prompt_payload()
        )

        profile.welcome_message_sent = True
        profile.conversation_mode = MODE_WELCOME
        profile.save(update_fields=["welcome_message_sent", "conversation_mode"])

        return Message.objects.create(
            profile=profile,
            role="assistant",
            content=response,
            channel=channel,
            ollama_prompt=welcome_payload,
        )

    def build_theme_prompt(self, theme_name: str) -> str:

        if not theme_name:
            raise ValueError("theme_name must be provided to build theme prompt")

        PROMPT = f"""Você é um GERADOR DE RESTRIÇÕES OPERACIONAIS DE CONVERSA.

            Sua tarefa é gerar um BLOCO DE CONTROLE DE COMPORTAMENTO
            que será ANEXADO ao prompt principal de um chatbot
            quando um TEMA for identificado na conversa.

            ⚠️ IMPORTANTE:
            Você NÃO deve gerar textos explicativos, guias, manuais ou conselhos.
            Você NÃO deve ensinar empatia.
            Você NÃO deve listar “atitudes a adotar” ou “atitudes a evitar”.

            O BLOCO GERADO DEVE:
            - Restringir comportamentos do chatbot
            - Proibir padrões que causam loop
            - Forçar avanço conversacional
            - Ser curto, direto e operacional

            =====================================
            OBJETIVO DO BLOCO GERADO
            =====================================

            Evitar:
            - repetição de acolhimento
            - reaplicação de templates narrativos
            - verbosidade
            - perguntas genéricas ou abstratas
            - over-interpretação

            Forçar:
            - respostas curtas
            - mudança de função após resistência
            - incorporação explícita do último turno do usuário
            - progressão da conversa

            =====================================
            FORMATO DE SAÍDA OBRIGATÓRIO
            =====================================

            Retorne SOMENTE um bloco no formato abaixo,
            sem introdução, sem explicações, sem listas didáticas:

            -------------------------------------
            CONTROLE TEMÁTICO — {theme_name}
            -------------------------------------

            ESTADO DO TEMA:
            [Descreva o estado emocional EM UMA FRASE curta, sem explicar.]

            É PROIBIDO AO ASSISTENTE:
            - [3 a 6 proibições claras e específicas]

            A PRÓXIMA RESPOSTA DEVE:
            - [3 a 5 exigências comportamentais objetivas]

            REGRAS DURAS:
            - Máximo de 2 frases
            - No máximo 1 pergunta, somente se destravar a conversa
            - É proibido repetir frases, perguntas ou funções já usadas

            Se violar qualquer regra acima, a resposta é inválida.

            -------------------------------------

            ⚠️ NÃO inclua:
            - “Atitudes a adotar”
            - “Atitudes a evitar”
            - Linguagem didática
            - Linguagem terapêutica
            - Conselhos
            - Explicações religiosas

            RETORNE APENAS O BLOCO.
            """

        logger.info(f"Generated theme prompt for '{theme_name}'")

        result = self.basic_call(
            url_type="generate",
            prompt=PROMPT,
            max_tokens=250,
        )

        return result

    def get_last_prompt_payload(self) -> Optional[Dict[str, Any]]:
        """Return the last provider payload sent to the underlying LLM client."""
        return self._llm_service.get_last_prompt_payload()

    def analyze_conversation_emotions(self, profile: Profile) -> str:

        transcript_text = ""
        for message in profile.messages.for_context():
            transcript_text += f"{message.role}: {message.content}\n\n"

        SYSTEM_PROMPT = f"""Você é um AUDITOR TÉCNICO DE QUALIDADE CONVERSACIONAL HUMANO–IA.

            Seu papel é produzir uma ANÁLISE CRÍTICA, OPERACIONAL, IMPARCIAL e RIGOROSA
            da interação entre USUÁRIO e BOT.

            Você NÃO é terapeuta, moderador, conselheiro ou participante da conversa.
            Você atua como um engenheiro de qualidade conversacional.

            ==================================================
            ESCOPO DA ANÁLISE
            ==================================================

            Analise EXCLUSIVAMENTE as mensagens do BOT.
            Mensagens do USUÁRIO servem apenas como contexto factual.

            Seu foco é detectar e explicar falhas estruturais como:
            - loops
            - repetição literal ou semântica
            - reaplicação de templates narrativos
            - falhas de progressão
            - falhas de estágio conversacional
            - over-interpretação
            - imposição narrativa ou moral
            - verbosidade excessiva
            - quebra de contexto ou identidade
            - inconsistência de perguntas
            - julgamento implícito

            ==================================================
            REGRAS ABSOLUTAS
            ==================================================

            - NÃO faça terapia
            - NÃO console o usuário
            - NÃO seja gentil por educação
            - NÃO interprete intenções além do texto
            - NÃO invente contexto
            - NÃO normalize falhas do BOT

            Baseie-se SOMENTE no que está explicitamente presente na transcrição.
            Seja técnico, direto e específico.

            Quando necessário, cite trechos curtos (máx. 12 palavras).

            ==================================================
            OBJETIVO PRINCIPAL
            ==================================================

            Explicar POR QUE uma conversa que poderia evoluir
            entra em LOOP, TRAVA ou REGRESSÃO,
            identificando FALHAS ESTRUTURAIS do BOT
            e propondo correções concretas em PROMPT e RUNTIME.

            ==================================================
            DEFINIÇÕES OBRIGATÓRIAS (USE COMO CRITÉRIO)
            ==================================================

            A) LOOP (FALHA CRÍTICA)
            Ocorre quando, por 2 ou mais turnos, o BOT:
            - repete frases ou variações mínimas
            - reaplica o mesmo template estrutural
            - faz a mesma pergunta (mesmo significado)
            - ignora informação nova trazida pelo usuário

            B) TEMPLATE DOMINANTE (FALHA)
            Uso repetido do mesmo molde estrutural,
            independente do conteúdo do usuário, por exemplo:
            - acolhimento genérico
            - “espaço seguro”
            - fé abstrata
            - moralização suave
            - perguntas genéricas de reflexão

            C) OVER-INTERPRETAÇÃO (FALHA)
            O BOT atribui:
            - intenções
            - desejos
            - estágios emocionais
            - valores morais
            que o usuário NÃO declarou explicitamente.

            D) IMPOSIÇÃO NARRATIVA (FALHA CRÍTICA)
            O BOT:
            - define a história do usuário por ele
            - atribui culpa, preço, erro ou mérito
            - fecha possibilidades com julgamentos implícitos
            Ex.: “você está pagando um preço alto”, “isso trouxe mais dor”.

            E) VERBOSIDADE (FALHA)
            O BOT:
            - escreve demais para entradas curtas
            - mistura múltiplas ideias
            - usa abstrações desnecessárias
            especialmente em temas sensíveis.

            F) FALHA DE ESTÁGIO CONVERSACIONAL (FALHA CRÍTICA)
            O usuário muda claramente de estágio (ex.: ambivalência, resistência),
            mas o BOT:
            - não muda de estratégia
            - mantém o mesmo modo de resposta

            G) QUEBRA DE CONTEXTO / IDENTIDADE (FALHA CRÍTICA)
            O BOT:
            - erra o nome do usuário
            - alterna identidades
            - contradiz fatos básicos já estabelecidos

            H) PROGRESSÃO (SUCESSO)
            O BOT progride quando:
            - incorpora informação nova do usuário
            - muda de estratégia após resistência
            - faz UMA pergunta concreta e destravadora
            - oferece um próximo passo pequeno e realista

            ==================================================
            PLACAR OBRIGATÓRIO (0–10)
            ==================================================

            Avalie APENAS mensagens do BOT.

            Para cada TURNO DO BOT, atribua:

            1) NOTA DA RESPOSTA (0–10)

            0–2  → falha crítica (loop, julgamento, quebra de contexto)
            3–4  → template dominante, repetição, sem avanço
            5–6  → parcialmente relevante, mas fraca ou genérica
            7–8  → clara, contida, avança
            9–10 → excelente, destrava a conversa

            REGRA DURA:
            Se houver LOOP, TEMPLATE DOMINANTE,
            IMPOSIÇÃO NARRATIVA ou QUEBRA DE CONTEXTO,
            a nota NÃO pode ser maior que 4.

            2) NOTA DA PERGUNTA (0–10), se houver

            0–2  → repetida, moralizante, abstrata
            3–4  → pouco conectada ao último turno
            5–6  → aceitável, mas ampla
            7–8  → curta, específica e conectada
            9–10 → simples, concreta e destravadora

            Se NÃO houver pergunta: escreva “Pergunta: N/A”.

            ==================================================
            ESTRUTURA DE SAÍDA (FORMATO RÍGIDO)
            ==================================================

            Retorne EXATAMENTE nas seções abaixo, nesta ordem:

            1) Diagnóstico rápido (3 bullets)
            - Cada bullet deve apontar UMA CAUSA RAIZ estrutural

            2) Placar turno a turno (tabela textual)
            Inclua APENAS turnos do BOT.

            TURNO | RESUMO (≤12 palavras) | RESPOSTA (0–10) | PERGUNTA (0–10 ou N/A) | FALHA PRINCIPAL | COMO CORRIGIR (1 frase)

            FALHA PRINCIPAL deve ser UMA destas:
            - LOOP
            - TEMPLATE DOMINANTE
            - OVER-INTERPRETAÇÃO
            - IMPOSIÇÃO NARRATIVA
            - VERBOSIDADE
            - FALHA DE ESTÁGIO
            - QUEBRA DE CONTEXTO
            - PERGUNTA RUIM
            - BOM

            3) Evidências do loop
            - Liste 2–5 frases repetidas ou quase repetidas
            - Explique por que isso trava a conversa

            4) Falhas estruturais identificadas
            - Descreva PADRÕES recorrentes
            - Não descreva eventos isolados

            5) Recomendações de PROMPT (máx. 8 bullets)
            Inclua obrigatoriamente:
            - regra anti-repetição literal
            - regra anti-template dominante
            - regra anti-imposição narrativa
            - mudança obrigatória de estratégia após ambivalência
            - ação obrigatória após 2 turnos sem avanço
            - limite duro de tamanho
            - resposta direta a pedidos explícitos

            6) Recomendações de RUNTIME (máx. 6 bullets)
            Sugestões técnicas, como:
            - detecção de similaridade semântica
            - bloqueio de frases/julgamentos
            - validação de identidade (nome)
            - cache do modo conversacional
            - state machine explícita
            - forçar “modo orientação” após loop

            ==================================================
            ENTRADA
            ==================================================

            TRANSCRIÇÃO:
            {transcript_text}

            Responda APENAS com a análise estruturada acima.
            Não adicione introdução, conclusão ou comentários extras.
        """

        response_text = self.basic_call(
            url_type="generate",
            prompt=SYSTEM_PROMPT,
            max_tokens=2000,
        )

        analysis = response_text
        logger.info("Generated critical analysis of simulated conversation")
        return analysis
