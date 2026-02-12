"""
TEMPERATURE        | COMPORTAMENTO
-------------------|---------------------------------------------
0.0 – 0.2          | Quase determinístico, frio, repetitivo
0.3 – 0.5          | Controlado, humano, consistente
0.6 – 0.8          | Natural, mais espontâneo
0.9 – 1.2          | Criativo, imprevisível
> 1.2              | Caótico, quebra regras fácil

NUM_PREDICT                      | RECOMENDADO              | OBSERVAÇÃO
---------------------------------|--------------------------|-------------------------------
Resposta ultra curta (1 frase)   | 30                       | Muito rígido
1–2 frases humanas               | 50–70                    | Ideal para simulação
Até 3 frases (limite duro)       | 80–100                   | Mais seguro
Resposta explicativa curta       | 150                      | Pode escapar
Texto médio                      | 250–400                  | Já não é conversa
Texto longo                      | 500+                     | Risco alto de quebrar regras
"""
import json
import logging
import os
import re
from datetime import timedelta
from typing import Any, Dict, Literal, Optional, Union
from urllib.parse import urljoin

import requests
from django.utils import timezone

from core.models import Message, Profile
from services.conversation_runtime import (
    MODE_ACOLHIMENTO,
    MODE_AMBIVALENCIA,
    MODE_CULPA,
    MODE_DEFENSIVO,
    MODE_EXPLORACAO,
    MODE_ORIENTACAO,
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
    has_repeated_user_pattern,
    has_self_guided_help,
    has_spiritual_baseline_signal,
    is_semantic_loop,
    semantic_similarity,
    should_force_progress_fallback,
    starts_with_user_echo,
    strip_opening_name_if_recently_used,
)
from services.rag_service import get_rag_context

logger = logging.getLogger(__name__)

VALID_CONVERSATION_MODES = {
    MODE_WELCOME,
    MODE_ACOLHIMENTO,
    MODE_EXPLORACAO,
    MODE_AMBIVALENCIA,
    MODE_DEFENSIVO,
    MODE_CULPA,
    MODE_ORIENTACAO,
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


# Helper constant for gender context in Portuguese
# This instruction is in Portuguese because it's part of the system prompt
# sent to the LLM, which operates in Brazilian Portuguese


class OllamaService:
    """Service class for interacting with local Ollama LLM API."""

    def __init__(self):
        """Initialize Ollama client with configuration from environment."""
        self.base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.api_url_base = f"{self.base_url}/api/"
        self._last_prompt_payload = None  # Store last payload for observability

    def basic_call(
        self,
        prompt: Union[str, list],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 100,
        url_type: str = Literal["chat", "generate"],
        timeout: int = 60,
        top_p: float = None,
        repeat_penalty: float = None,
    ) -> str:

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if top_p:
            payload["options"]["top_p"] = top_p
        if repeat_penalty:
            payload["options"]["repeat_penalty"] = repeat_penalty

        response = requests.post(
            urljoin(self.api_url_base, url_type),
            json=payload,
            timeout=timeout,
        )

        response.raise_for_status()
        response_data = response.json()

        if url_type == "chat":
            return response_data.get("message", {}).get("content", "").strip()
        else:
            return response_data.get("response", "").strip()

    def _safe_parse_json(self, text: str) -> Dict[str, Any]:
        if not text:
            return {}
        raw = text.strip()
        try:
            return json.loads(raw)
        except Exception:
            pass

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}

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
            model="llama3:8b",
            temperature=0.2,
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

    def _build_guided_fallback_response(
        self,
        *,
        user_message: str,
        recent_assistant_messages: list,
        direct_guidance_request: bool,
        requires_real_help: bool,
        allow_spiritual_context: bool,
    ) -> str:
        if direct_guidance_request:
            candidate = (
                "Vamos começar por um passo simples hoje: retire a bebida de perto e avise uma pessoa de confiança que você precisa de apoio agora. "
                "Depois disso, me diga qual horário você vai fazer esse passo."
            )
        elif requires_real_help:
            candidate = (
                "O que você está vivendo é sério, e pedir apoio agora é um passo de coragem. "
                "Hoje, procure uma pessoa de confiança ou um grupo como AA/CAPS AD e compartilhe exatamente o que você me disse."
            )
        else:
            candidate = (
                "Obrigado por abrir isso com sinceridade. "
                "Diante do que você trouxe, qual pequeno passo concreto você consegue dar ainda hoje para se proteger desse ciclo? "
                "Se quiser, eu te ajudo a escolher esse passo agora."
            )

        if (
            recent_assistant_messages
            and semantic_similarity(recent_assistant_messages[-1], candidate) > 0.85
        ):
            candidate = (
                "Vamos tornar isso prático agora: escolha uma ação de proteção para as próximas 2 horas e me diga qual será. "
                "Se puder, peça apoio de alguém de confiança hoje mesmo."
            )
        if allow_spiritual_context:
            candidate += " Se você quiser, agora mesmo faça uma oração curta pedindo perdão, força e direção a Deus."
        return enforce_hard_limits(candidate)

    def _build_dynamic_runtime_prompt(
        self,
        *,
        conversation_mode: str,
        previous_mode: str,
        spiritual_intensity: str,
        allow_spiritual_context: bool,
        direct_guidance_request: bool,
        force_progress_fallback: bool,
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
            MODE_WELCOME: "acolhimento inicial curto",
        }.get(conversation_mode, "avançar a conversa com precisão")

        mode_actions = {
            MODE_ACOLHIMENTO: [
                "Valide um elemento específico da fala atual.",
                "Use no máximo 1 pergunta breve.",
            ],
            MODE_EXPLORACAO: [
                "Faça uma pergunta aberta concreta ligada ao último turno.",
                "Evite explicações longas.",
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
                "Ofereça 1 ação prática para hoje.",
                "Ofereça apoio humano concreto ou ação individual imediata.",
            ],
            MODE_WELCOME: [
                "Acolha com sobriedade e convide para continuidade.",
            ],
        }.get(conversation_mode, ["Avance um passo concreto."])

        spiritual_policy = "Mantenha base espiritual leve (esperança/propósito) sem linguagem explícita."
        if allow_spiritual_context:
            spiritual_policy = "Pode usar linguagem cristã explícita curta e respeitosa, sem imposição."

        prompt = f"""
MODO ATUAL: {conversation_mode}
MODO ANTERIOR: {previous_mode}
OBJETIVO DO MODO: {mode_objective}
INTENSIDADE ESPIRITUAL: {spiritual_intensity}

REGRAS GERAIS:
- Máximo 3 frases e 1 pergunta.
- Validar algo específico que o usuário acabou de dizer.
- Não repetir frases/estruturas dos últimos turnos.
- Não iniciar ecoando a frase do usuário.
- Não inferir sentimentos não declarados.
- {spiritual_policy}
- Estrutura: validação específica -> direção espiritual compatível -> próximo passo.
"""
        if direct_guidance_request:
            prompt += "\nPEDIDO EXPLÍCITO DE AJUDA DETECTADO: resposta deve conter orientação prática direta.\n"
        if force_progress_fallback:
            prompt += "\nESTAGNAÇÃO DETECTADA: evitar pergunta padrão repetida; destravar com ação concreta.\n"
        if active_topic:
            prompt += f"\nTÓPICO ATIVO: {active_topic}\n"
        if top_topics:
            prompt += f"TÓPICOS RECENTES: {top_topics}\n"
        if theme_prompt:
            prompt += f"\nTEMA CONTEXTUAL:\n{theme_prompt}\n"

        prompt += "\nAÇÕES OBRIGATÓRIAS DESTE TURNO:\n"
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

    def generate_response_message(self, profile: Profile, channel: str) -> Message:

        if not profile.welcome_message_sent:
            return self.generate_welcome_message(profile=profile, channel=channel)

        queryset = profile.messages.for_context()
        last_person_message = queryset.filter(role="user").last()
        if not last_person_message:
            return self.generate_welcome_message(profile=profile, channel=channel)

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

        topic_signal = self._extract_topic_signal(
            last_user_message=last_person_message.content,
            recent_messages=list(reversed(recent_context_messages)),
            current_topic=profile.current_topic,
        )
        active_topic = self._merge_topic_memory(
            profile=profile, topic_signal=topic_signal
        )

        previous_mode = LEGACY_MODE_MAP.get(
            profile.conversation_mode, profile.conversation_mode
        )
        if previous_mode not in VALID_CONVERSATION_MODES:
            previous_mode = MODE_WELCOME

        is_first_message = queryset.filter(role="assistant").count() == 0
        signals = detect_user_signals(last_person_message.content)
        direct_guidance_request = bool(signals.get("guidance_request"))
        repeated_user_pattern = has_repeated_user_pattern(recent_user_messages)
        ambivalence_or_repeated = (
            bool(signals.get("ambivalence")) or repeated_user_pattern
        )
        explicit_spiritual_context = bool(signals.get("spiritual_context"))
        high_spiritual_need = bool(signals.get("high_spiritual_need"))
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

        spiritual_intensity = choose_spiritual_intensity(
            mode=conversation_mode,
            spiritual_context=explicit_spiritual_context,
            high_spiritual_need=high_spiritual_need,
        )

        force_progress_fallback = (
            should_force_progress_fallback(
                recent_user_messages, recent_assistant_messages
            )
            and not new_information
        )

        substance_context = self._is_substance_context(
            user_message=last_person_message.content, active_topic=active_topic
        )
        requires_real_help = direct_guidance_request or substance_context
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
        rag_contexts = get_rag_context(last_person_message.content, limit=3)
        prompt_aux = self._build_dynamic_runtime_prompt(
            conversation_mode=conversation_mode,
            previous_mode=previous_mode,
            spiritual_intensity=spiritual_intensity,
            allow_spiritual_context=allow_spiritual_context,
            direct_guidance_request=direct_guidance_request,
            force_progress_fallback=force_progress_fallback,
            active_topic=active_topic,
            top_topics=top_topics,
            last_user_message=last_person_message.content,
            theme_prompt=profile.theme.prompt if profile.theme else None,
            context_messages=context_messages,
            rag_contexts=rag_contexts,
        )

        max_regenerations = 3
        regeneration_counter = 0
        loop_counter = 1 if loop_detected else 0

        base_temperature = 0.6
        if conversation_mode in {MODE_ORIENTACAO, MODE_AMBIVALENCIA}:
            base_temperature = 0.45
        selected_temperature = base_temperature
        approved_content = ""

        for attempt in range(max_regenerations):
            if approved_content:
                break

            selected_temperature = max(0.2, base_temperature - (attempt * 0.15))
            candidate = self.basic_call(
                url_type="generate",
                model="wachat-v9",
                prompt=prompt_aux,
                temperature=selected_temperature,
                max_tokens=120,
            )

            candidate = strip_opening_name_if_recently_used(
                message=candidate,
                name=profile.name,
                recent_assistant_messages=recent_assistant_messages,
            )
            candidate = enforce_hard_limits(candidate)

            semantic_loop = False
            if recent_assistant_messages:
                semantic_loop = is_semantic_loop(
                    recent_assistant_messages[-1], candidate
                )
            blocked_template = contains_repeated_blocked_pattern(
                candidate, recent_assistant_messages
            )
            has_unverified_inference = contains_unverified_inference(
                last_person_message.content, candidate
            )
            spiritual_imposition = contains_spiritual_imposition(candidate)
            unsolicited_spiritualization = contains_unsolicited_spiritualization(
                last_person_message.content, candidate
            )
            spiritual_template_repetition = contains_spiritual_template_repetition(
                candidate, recent_assistant_messages
            )
            generic_empathy = contains_generic_empathy_without_grounding(
                last_person_message.content, candidate
            )
            missing_spiritual_baseline = not has_spiritual_baseline_signal(candidate)
            missing_practical_step = (
                direct_guidance_request and not has_practical_action_step(candidate)
            )
            missing_real_support = (
                requires_real_help
                and not has_human_support_suggestion(candidate)
                and not has_self_guided_help(candidate)
            )
            leading_echo = starts_with_user_echo(
                user_message=last_person_message.content, assistant_message=candidate
            )

            if (
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
            ):
                regeneration_counter += 1
                if semantic_loop:
                    loop_counter += 1
                continue

            approved_content = candidate

        if not approved_content:
            loop_counter += 1
            approved_content = self._build_guided_fallback_response(
                user_message=last_person_message.content,
                recent_assistant_messages=recent_assistant_messages,
                direct_guidance_request=direct_guidance_request,
                requires_real_help=requires_real_help,
                allow_spiritual_context=allow_spiritual_context,
            )

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

        return Message.objects.create(
            profile=profile,
            role="assistant",
            content=approved_content,
            channel=channel,
            ollama_prompt=prompt_aux,
            ollama_prompt_temperature=selected_temperature,
        )

    def infer_gender(self, name: str) -> str:
        """
        Infer gender from a user's name using Ollama LLM.

        This is a soft, probabilistic inference based solely on the name.
        The result is for internal use only and should never be explicitly
        stated to the user.

        Args:
            name: The user's name (first name or full name)

        Returns:
            One of: "male", "female", or "unknown"
        """
        try:
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
                model="llama3:8b",
                temperature=0.3,
                max_tokens=10,
            )

            inferred = response_text.lower()

            # Validate response
            if inferred not in ["male", "female", "unknown"]:
                logger.warning(f"Unexpected gender inference result: {inferred}")
                return "unknown"

            logger.info(f"Gender inferred for name '{name}': {inferred}")
            return inferred

        except Exception as e:
            logger.error(f"Error inferring gender: {str(e)}", exc_info=True)
            return "unknown"

    def generate_welcome_message(self, profile: Profile, channel: str) -> Message:

        gender_context = ""
        if profile.inferred_gender != "unknown":
            gender_context = (
                f"\nGênero inferido (use isso APENAS para ajustar sutilmente o tom, "
                f"NUNCA mencione explicitamente): {profile.inferred_gender}"
            )

        PROMPT = f"""
        Você está gerando APENAS a primeira mensagem de boas-vindas para um novo usuário.

        IDENTIDADE:
        Você é uma presença espiritual cristã, humana, acolhedora e reflexiva.
        Você não é um pastor.
        Você não é um pregador.
        Você não é um coach.
        Você não é um bot.

        Seu papel é ser companhia espiritual serena.

        {gender_context}

        OBJETIVO:
        Criar uma mensagem inicial curta que gere:
        - Segurança
        - Presença
        - Abertura para diálogo

        TOM:
        - Calmo
        - Humano
        - Respeitoso
        - Natural em português brasileiro
        - Sem linguagem rebuscada

        PROIBIDO:
        - Emojis
        - Jargões religiosos
        - Versículos explícitos
        - Chamadas à conversão
        - Explicar funcionalidades
        - Falar sobre religião institucional
        - Usar Deus como autoridade
        - Dizer “Deus quer”, “Deus exige”, “Deus manda”
        - Mencionar gênero explicitamente

        PERMITIDO:
        - Apresentar Deus como presença que acompanha
        - Linguagem simples e próxima
        - Sensação de escuta genuína

        ESTRUTURA OBRIGATÓRIA (máximo 3 frases):
        1. Saudação usando o nome: {profile.name}
        2. Apresentação do espaço como lugar seguro, sem julgamento
        3. UMA única pergunta aberta e suave

        EXEMPLOS DE PERGUNTA (escolha apenas uma, ou crie variação semelhante):
        - O que te trouxe até aqui hoje?
        - O que tem pesado mais dentro de você ultimamente?
        - Em que parte da sua caminhada você sente que precisa de companhia agora?

        IMPORTANTE:
        - Não explique o que está fazendo.
        - Não escreva nada fora da mensagem.
        - Não adicione títulos.
        - Não quebre a estrutura.
        - Seja breve.
        - Máximo 90 palavras.

        Se existir última mensagem do usuário, adapte levemente o tom para ela.
        """

        last_user_message = profile.messages.filter(role="user").last()
        if last_user_message:
            PROMPT += f"\n\nCONTEXTO DO USUÁRIO:\n{last_user_message.content}\n"

        temperature = 0.6
        response = self.basic_call(
            url_type="generate",
            prompt=PROMPT,
            model="llama3:8b",
            temperature=temperature,
            max_tokens=120,
        )

        profile.welcome_message_sent = True
        profile.conversation_mode = MODE_WELCOME
        profile.save(update_fields=["welcome_message_sent", "conversation_mode"])

        return Message.objects.create(
            profile=profile,
            role="assistant",
            content=response,
            channel=channel,
            ollama_prompt=PROMPT,
            ollama_prompt_temperature=temperature,
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
            model="llama3:8b",
            temperature=0.7,
            max_tokens=250,
        )

        return result

    def get_last_prompt_payload(self) -> Optional[Dict[str, Any]]:
        """
        Get the last Ollama prompt payload sent for observability.

        Returns:
            The last payload dict sent to Ollama, or None if no request was made yet
        """
        return self._last_prompt_payload

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
            - fallback após 2 turnos sem avanço
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
            model="llama3:8b",
            temperature=0.45,
            max_tokens=2000,
        )

        analysis = response_text
        logger.info("Generated critical analysis of simulated conversation")
        return analysis
