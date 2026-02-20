import json
import logging
import re
from copy import deepcopy
from datetime import timedelta
from string import Formatter
from typing import Any, Dict, List, Optional

from django.utils import timezone

from core.models import Message, Profile, Theme
from prompts.prompt_defaults import DEFAULT_WACHAT_SYSTEM_PROMPT
from prompts.prompt_registry import PromptRegistry
from services.conversation_runtime import (
    MODE_ACOLHIMENTO,
    MODE_AMBIVALENCIA,
    MODE_CULPA,
    MODE_DEFENSIVO,
    MODE_EXPLORACAO,
    MODE_ORIENTACAO,
    MODE_PASTOR_INSTITUCIONAL,
    MODE_PRESENCA_PROFUNDA,
    MODE_VULNERABILIDADE_INICIAL,
    MODE_WELCOME,
    choose_conversation_mode,
    choose_spiritual_intensity,
    detect_user_signals,
    has_new_information,
    has_repeated_user_pattern,
    semantic_similarity,
)
from services.openai_service import OpenAIService
from services.rag_service import RAGService
from services.theme_classifier import ThemeClassifier

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
    MODE_PASTOR_INSTITUCIONAL,
    MODE_VULNERABILIDADE_INICIAL,
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
RELATIONAL_TOPICS = {"familia", "família", "conflito", "relacionamento"}
WACHAT_RESPONSE_MODEL = "gpt-5-mini"
FIXED_TEMPERATURE = 1.0
FIXED_TIMEOUT_SECONDS = 60
FIXED_RESPONSE_MAX_COMPLETION_TOKENS = 1800
FIXED_WELCOME_MAX_COMPLETION_TOKENS = 1200
FIXED_TOPIC_SIGNAL_MAX_COMPLETION_TOKENS = 1000
FIXED_GENDER_INFERENCE_MAX_COMPLETION_TOKENS = 400
FIXED_THEME_PROMPT_MAX_COMPLETION_TOKENS = 1200
FIXED_EVALUATION_MAX_COMPLETION_TOKENS = 500
EVALUATION_EMPTY_RETRY_ATTEMPTS = 2
FIXED_SIMULATION_ANALYSIS_MAX_COMPLETION_TOKENS = 3200
EVALUATION_MODEL = "gpt-5-mini"
MULTI_MESSAGE_MIN_PARTS = 3
MULTI_MESSAGE_MAX_PARTS = 4
LOW_SCORE_REFINEMENT_THRESHOLD = 5.0
TARGET_RESPONSE_SCORE = 8.0
MAX_SCORE_REFINEMENT_ROUNDS = 3
LOOP_SIMILARITY_THRESHOLD = 0.85
OPENING_SIMILARITY_BLOCK_THRESHOLD = 0.9
ASSISTANT_NGRAM_SIZE = 4
LOOP_PRACTICAL_COOLDOWN_TURNS = 3
PRAYER_COOLDOWN_TURNS = 2
MAX_INFERENCE_REGEN_PER_ROUND = 1
STALL_TURNS_FORCE_ACTION = 2
MAX_EMPATHY_SENTENCES_PER_RESPONSE = 1
MAX_EMPATHY_SENTENCE_WORDS = 18
EMPATHY_MARKERS = [
    "sinto muito",
    "lamento",
    "imagino como",
    "faz sentido",
    "entendo que",
    "isso dói",
    "isso doi",
    "deve estar pesado",
]
PRAYER_LANGUAGE_MARKERS = [
    "deus",
    "jesus",
    "oração",
    "oracao",
    "orar",
    "oro por",
    "senhor,",
    "amém",
    "amen",
]
STRONG_INFERENCE_MARKERS = [
    "isso mostra que",
    "isso prova que",
    "a causa é",
    "a causa disso é",
    "claramente você",
    "com certeza você",
    "o problema é que você",
    "isso aconteceu porque você",
]
USER_CITATION_MARKERS = [
    "você disse",
    "voce disse",
    "você falou",
    "voce falou",
    "você mencionou",
    "voce mencionou",
    "você contou",
    "voce contou",
]
PROGRESS_STATE_COLETA = "COLETA"
PROGRESS_STATE_PROPOSTA = "PROPOSTA"
PROGRESS_STATE_EXECUCAO = "EXECUCAO"
PROGRESS_STATE_CONFIRMACAO = "CONFIRMACAO"
PROGRESS_STATE_FECHAMENTO = "FECHAMENTO"
VALID_PROGRESS_STATES = {
    PROGRESS_STATE_COLETA,
    PROGRESS_STATE_PROPOSTA,
    PROGRESS_STATE_EXECUCAO,
    PROGRESS_STATE_CONFIRMACAO,
    PROGRESS_STATE_FECHAMENTO,
}
PRAYER_REQUEST_MARKERS = (
    "ore por mim",
    "ora por mim",
    "ore comigo",
    "ora comigo",
    "reze por mim",
    "reze comigo",
    "preciso de oração",
    "pode orar",
    "oração por mim",
)
LIVE_SUPPORT_REQUEST_MARKERS = (
    "posso te ligar",
    "você pode me ligar",
    "voce pode me ligar",
    "me liga",
    "me ligue",
    "me manda mensagem",
    "me mande mensagem",
    "fique comigo",
    "fica comigo",
)
INTENT_DEFAULT = "DEFAULT"
INTENT_COMPANHIA = "COMPANHIA"
INTENT_TEXTO = "TEXTO"
INTENT_PASSO_PRATICO = "PASSO_PRATICO"
INTENT_ORACAO = "ORACAO"

STRATEGY_SUPPORT_COMPANHIA = "SUPPORT_COMPANHIA"
STRATEGY_OFFER_TEXT_ALTERNATIVES = "OFFER_TEXT_ALTERNATIVES"
STRATEGY_EXECUTE_PRACTICAL_STEP = "EXECUTE_PRACTICAL_STEP"
STRATEGY_PRAYER_BRIEF = "PRAYER_BRIEF"
STRATEGY_CONTEXT_EXPLORATION = "CONTEXT_EXPLORATION"
STRATEGY_CONFIRM_AND_EXECUTE = "CONFIRM_AND_EXECUTE"
STRATEGY_CLARIFY_BLOCKER = "CLARIFY_BLOCKER"

WACHAT_RESPONSE_SYSTEM_PROMPT = DEFAULT_WACHAT_SYSTEM_PROMPT


# Helper constant for gender context in Portuguese
# This instruction is in Portuguese because it's part of the system prompt
# sent to the LLM, which operates in Brazilian Portuguese


class ChatService:
    """Conversation orchestration service using OpenAI GPT-5."""

    def __init__(self):
        self._llm_service = OpenAIService()
        self._rag_service = RAGService()
        self._theme_classifier = ThemeClassifier()
        self._prompt_registry = PromptRegistry()

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

    def _split_sentences(self, text: str) -> List[str]:
        normalized = re.sub(r"\s+", " ", (text or "").strip())
        if not normalized:
            return []
        sentences = [
            item.strip()
            for item in re.split(r"(?<=[.!?])\s+", normalized)
            if item.strip()
        ]
        return sentences

    def _tokenize_for_ngram(self, text: str) -> List[str]:
        normalized = (text or "").lower()
        return re.findall(r"[a-zà-ÿ0-9]+", normalized)

    def _extract_ngrams(self, text: str, n: int) -> set:
        tokens = self._tokenize_for_ngram(text)
        if len(tokens) < n:
            return set()
        ngrams = set()
        for index in range(0, len(tokens) - n + 1):
            ngrams.add(" ".join(tokens[index : index + n]))  # noqa: E203
        return ngrams

    def _build_recent_assistant_ngram_ban(
        self, recent_assistant_messages: List[str]
    ) -> set:
        banned = set()
        for text in recent_assistant_messages[-2:]:
            banned.update(self._extract_ngrams(text, ASSISTANT_NGRAM_SIZE))
        return banned

    def _candidate_has_banned_ngram(self, candidate: str, banned_ngrams: set) -> bool:
        if not banned_ngrams:
            return False
        candidate_ngrams = self._extract_ngrams(candidate, ASSISTANT_NGRAM_SIZE)
        if not candidate_ngrams:
            return False
        return any(ngram in banned_ngrams for ngram in candidate_ngrams)

    def _candidate_opening_similarity(
        self, candidate: str, recent_assistant_messages: List[str]
    ) -> float:
        candidate_sentences = self._split_sentences(candidate)
        if not candidate_sentences:
            return 0.0
        candidate_opening = candidate_sentences[0]
        best_similarity = 0.0
        for text in recent_assistant_messages[-2:]:
            reference_sentences = self._split_sentences(text)
            if not reference_sentences:
                continue
            similarity = semantic_similarity(candidate_opening, reference_sentences[0])
            if similarity > best_similarity:
                best_similarity = similarity
        return best_similarity

    def _candidate_has_required_new_element(self, candidate: str) -> bool:
        normalized = (candidate or "").lower()
        has_question = "?" in normalized
        action_markers = [
            "faça",
            "faca",
            "vamos",
            "tente",
            "comece",
            "agora",
            "passo",
            "escolha",
            "envie",
            "respire",
        ]
        summary_markers = [
            "resumindo",
            "em resumo",
            "então",
            "pelo que você disse",
            "pelo que voce disse",
        ]
        has_action = any(marker in normalized for marker in action_markers)
        has_summary = any(marker in normalized for marker in summary_markers)
        return has_question or has_action or has_summary

    def _candidate_has_practical_action(self, candidate: str) -> bool:
        normalized = (candidate or "").lower()
        practical_markers = [
            "agora",
            "faça",
            "faca",
            "comece",
            "envie",
            "respire",
            "anote",
            "defina",
            "escolha",
            "próximo passo",
            "proximo passo",
        ]
        return any(marker in normalized for marker in practical_markers)

    def _contains_prayer_language(self, text: str) -> bool:
        normalized = (text or "").lower()
        return any(marker in normalized for marker in PRAYER_LANGUAGE_MARKERS)

    def _detect_explicit_user_intent(self, last_user_message: str) -> str:
        normalized = (last_user_message or "").lower()
        if any(marker in normalized for marker in PRAYER_REQUEST_MARKERS):
            return INTENT_ORACAO
        text_markers = [
            "escreva para mim",
            "escrever para mim",
            "faz uma mensagem",
            "me dá uma mensagem",
            "me de uma mensagem",
            "texto pronto",
            "pronto para copiar",
            "modelo de mensagem",
        ]
        if any(marker in normalized for marker in text_markers):
            return INTENT_TEXTO
        companionship_markers = [
            "fica comigo",
            "fique comigo",
            "fica aqui",
            "fique aqui",
            "fica online",
            "fique online",
            "trocando mensagem",
            "não me deixa sozinho",
            "nao me deixa sozinho",
        ]
        if any(marker in normalized for marker in companionship_markers):
            return INTENT_COMPANHIA
        practical_markers = [
            "o que faço agora",
            "o que eu faço agora",
            "como faço",
            "como começo",
            "por onde começo",
            "próximo passo",
            "proximo passo",
            "passo a passo",
        ]
        if any(marker in normalized for marker in practical_markers):
            return INTENT_PASSO_PRATICO
        return INTENT_DEFAULT

    def _select_strategy_for_intent(
        self,
        *,
        explicit_user_intent: str,
        direct_guidance_request: bool,
        previous_strategy_key: str,
        previous_strategy_repetition_count: int,
    ) -> Dict[str, Any]:
        primary_by_intent = {
            INTENT_COMPANHIA: STRATEGY_SUPPORT_COMPANHIA,
            INTENT_TEXTO: STRATEGY_OFFER_TEXT_ALTERNATIVES,
            INTENT_PASSO_PRATICO: STRATEGY_EXECUTE_PRACTICAL_STEP,
            INTENT_ORACAO: STRATEGY_PRAYER_BRIEF,
            INTENT_DEFAULT: (
                STRATEGY_CONFIRM_AND_EXECUTE
                if direct_guidance_request
                else STRATEGY_CONTEXT_EXPLORATION
            ),
        }
        alternative_by_strategy = {
            STRATEGY_SUPPORT_COMPANHIA: STRATEGY_EXECUTE_PRACTICAL_STEP,
            STRATEGY_OFFER_TEXT_ALTERNATIVES: STRATEGY_CONFIRM_AND_EXECUTE,
            STRATEGY_EXECUTE_PRACTICAL_STEP: STRATEGY_CLARIFY_BLOCKER,
            STRATEGY_PRAYER_BRIEF: STRATEGY_EXECUTE_PRACTICAL_STEP,
            STRATEGY_CONTEXT_EXPLORATION: STRATEGY_CONFIRM_AND_EXECUTE,
            STRATEGY_CONFIRM_AND_EXECUTE: STRATEGY_CLARIFY_BLOCKER,
            STRATEGY_CLARIFY_BLOCKER: STRATEGY_EXECUTE_PRACTICAL_STEP,
        }
        selected = primary_by_intent.get(
            explicit_user_intent, STRATEGY_CONTEXT_EXPLORATION
        )
        strategy_alternative_forced = False
        if (
            previous_strategy_key == selected
            and previous_strategy_repetition_count >= 2
        ):
            selected = alternative_by_strategy.get(
                selected, STRATEGY_EXECUTE_PRACTICAL_STEP
            )
            strategy_alternative_forced = True
        return {
            "strategy_key": selected,
            "strategy_alternative_forced": strategy_alternative_forced,
        }

    def _empathy_sentence_stats(self, candidate: str) -> Dict[str, int]:
        sentences = self._split_sentences(candidate)
        empathy_count = 0
        max_empathy_words = 0
        for sentence in sentences:
            normalized_sentence = sentence.lower()
            if any(marker in normalized_sentence for marker in EMPATHY_MARKERS):
                empathy_count += 1
                words = len(self._tokenize_for_ngram(sentence))
                if words > max_empathy_words:
                    max_empathy_words = words
        return {
            "count": empathy_count,
            "max_words": max_empathy_words,
        }

    def _has_strong_inference(self, candidate: str) -> bool:
        normalized = (candidate or "").lower()
        if any(marker in normalized for marker in STRONG_INFERENCE_MARKERS):
            return True
        # Broad causal assertions without hedge.
        causal_patterns = [
            "isso é porque",
            "isso acontece porque",
            "você está assim porque",
            "voce está assim porque",
            "você está desse jeito porque",
            "voce está desse jeito porque",
        ]
        return any(pattern in normalized for pattern in causal_patterns)

    def _contains_user_citation(self, candidate: str, last_user_message: str) -> bool:
        normalized_candidate = (candidate or "").lower()
        if any(marker in normalized_candidate for marker in USER_CITATION_MARKERS):
            return True
        user_tokens = self._tokenize_for_ngram(last_user_message)
        if len(user_tokens) < 3:
            return False
        user_trigrams = set()
        for index in range(0, len(user_tokens) - 3 + 1):
            user_trigrams.add(" ".join(user_tokens[index : index + 3]))  # noqa: E203
        candidate_trigrams = self._extract_ngrams(candidate, 3)
        overlap = user_trigrams.intersection(candidate_trigrams)
        return len(overlap) > 0

    def _has_conditional_inference_confirmation(self, candidate: str) -> bool:
        normalized = (candidate or "").lower()
        has_conditional = "pode ser que" in normalized
        has_confirmation_request = "?" in normalized and (
            "faz sentido" in normalized
            or "confere" in normalized
            or "é isso" in normalized
            or "me confirma" in normalized
        )
        return has_conditional and has_confirmation_request

    def _extract_progress_metric(self, text: str) -> Dict[str, bool]:
        normalized = (text or "").lower()
        decision_markers = [
            "vou",
            "decidi",
            "escolhi",
            "combinado",
            "fechado",
            "ok",
        ]
        action_markers = [
            "agora",
            "faça",
            "faca",
            "envie",
            "respire",
            "anote",
            "defina",
            "passo",
            "agende",
        ]
        confirm_markers = [
            "confirmo",
            "confirmar",
            "check-in",
            "retorno",
            "me avisa",
            "combinamos",
        ]
        return {
            "decision_taken": any(marker in normalized for marker in decision_markers),
            "action_defined": any(marker in normalized for marker in action_markers),
            "next_step_confirmed": any(
                marker in normalized for marker in confirm_markers
            ),
        }

    def _progress_metric_score(self, metric: Dict[str, bool]) -> int:
        return (
            int(bool(metric.get("decision_taken")))
            + int(bool(metric.get("action_defined")))
            + int(bool(metric.get("next_step_confirmed")))
        )

    def _progress_advanced(
        self, previous_metric: Dict[str, bool], current_metric: Dict[str, bool]
    ) -> bool:
        if self._progress_metric_score(current_metric) > self._progress_metric_score(
            previous_metric
        ):
            return True
        for key in ("decision_taken", "action_defined", "next_step_confirmed"):
            if bool(current_metric.get(key)) and not bool(previous_metric.get(key)):
                return True
        return False

    def _count_concrete_actions(self, candidate: str) -> int:
        sentences = self._split_sentences(candidate)
        if not sentences:
            return 0
        action_markers = [
            "faça",
            "faca",
            "agora",
            "envie",
            "respire",
            "anote",
            "defina",
            "agende",
            "comece",
            "escolha",
        ]
        count = 0
        for sentence in sentences:
            normalized = sentence.lower()
            if any(marker in normalized for marker in action_markers):
                count += 1
        return count

    def _build_assistant_message_chunks(
        self, *, text: str, conversation_mode: str
    ) -> List[str]:
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        if conversation_mode == MODE_VULNERABILIDADE_INICIAL:
            return [" ".join(sentences)]

        if conversation_mode == MODE_ORIENTACAO:
            if len(sentences) < 2:
                return [" ".join(sentences)]
            if len(sentences) <= 4:
                return [" ".join(sentences[:2]), " ".join(sentences[2:])]
            return [
                " ".join(sentences[:2]),
                " ".join(sentences[2:4]),
                " ".join(sentences[4:]),
            ]

        if (
            conversation_mode != MODE_PASTOR_INSTITUCIONAL
            or len(sentences) < MULTI_MESSAGE_MIN_PARTS
        ):
            if len(sentences) <= 2:
                return [" ".join(sentences)]
            if len(sentences) <= 6:
                return [" ".join(sentences[:3]), " ".join(sentences[3:])]
            return [
                " ".join(sentences[:3]),
                " ".join(sentences[3:6]),
                " ".join(sentences[6:]),
            ]

        target_parts = min(MULTI_MESSAGE_MAX_PARTS, len(sentences))
        total = len(sentences)
        base_size = total // target_parts
        remainder = total % target_parts
        chunks = []
        cursor = 0
        for index in range(target_parts):
            chunk_size = base_size + (1 if index < remainder else 0)
            if chunk_size <= 0:
                continue
            chunk = " ".join(
                sentences[cursor : cursor + chunk_size]  # noqa: E203
            ).strip()  # noqa: E203
            if chunk:
                chunks.append(chunk)
            cursor += chunk_size

        return chunks or [" ".join(sentences)]

    def _build_refinement_runtime_prompt(
        self,
        *,
        base_runtime_prompt: str,
        round_number: int,
        score: float,
        analysis: str,
        improvement_prompt: str,
    ) -> str:
        return (
            f"{base_runtime_prompt}\n\n"
            f"REFINAMENTO DA RESPOSTA (RODADA {round_number}):\n"
            f"- Score anterior: {score}\n"
            f"- Análise crítica: {analysis}\n"
            f"- Instrução de melhoria: {improvement_prompt}\n"
            "- Gere uma nova resposta incorporando integralmente a instrução de melhoria.\n"
            "- Evite os problemas apontados na análise.\n"
            "- Não mencione avaliação, score, análise ou refinamento na resposta final.\n"
        )

    def _extract_topic_signal(
        self,
        last_user_message: str,
        recent_messages: list,
        current_topic: Optional[str],
    ) -> Dict[str, Any]:
        transcript = ""
        for message in recent_messages[-5:]:
            transcript += f"{message.role.upper()}: {message.content}\n"
        topic_prompt_template = self._prompt_registry.get_active_prompt(
            "topic.extractor.main"
        ).content
        prompt = topic_prompt_template.format(
            current_topic=current_topic or "null",
            last_user_message=last_user_message,
            transcript=transcript if transcript else "sem histórico",
        )
        raw = self.basic_call(
            url_type="generate",
            prompt=prompt,
            max_tokens=FIXED_TOPIC_SIGNAL_MAX_COMPLETION_TOKENS,
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

        logger.info(
            f"Extracted topic signal: topic={topic} confidence={confidence} keep_current={keep_current}"
        )

        return {
            "topic": topic,
            "confidence": confidence,
            "keep_current": keep_current,
        }

    def _evaluate_response(
        self, *, user_message: str, assistant_response: str
    ) -> Dict[str, Any]:
        client = getattr(self._llm_service, "client", None)
        if client is None:
            raise RuntimeError(
                "OpenAI client is not available on configured LLM service."
            )

        evaluation_prompt_selection = self._prompt_registry.get_evaluation_prompt()
        evaluation_system_prompt = evaluation_prompt_selection.content

        evaluation_user_prompt = f"""
Última mensagem do usuário:
{user_message}

Resposta do assistente para avaliar:
{assistant_response}
""".strip()

        raw_content = None
        for attempt in range(1, EVALUATION_EMPTY_RETRY_ATTEMPTS + 1):
            response = client.chat.completions.create(
                model=EVALUATION_MODEL,
                messages=[
                    {"role": "system", "content": evaluation_system_prompt},
                    {"role": "user", "content": evaluation_user_prompt},
                ],
                max_completion_tokens=FIXED_EVALUATION_MAX_COMPLETION_TOKENS,
                reasoning_effort="low",
                timeout=FIXED_TIMEOUT_SECONDS,
                response_format={"type": "json_object"},
            )

            choices = getattr(response, "choices", None) or []
            message = getattr(choices[0], "message", None) if choices else None
            candidate_content = getattr(message, "content", None) if message else None
            if isinstance(candidate_content, str) and candidate_content.strip():
                raw_content = candidate_content
                break

            logger.warning(
                "Evaluation returned empty content attempt=%s/%s",
                attempt,
                EVALUATION_EMPTY_RETRY_ATTEMPTS,
            )

        if not isinstance(raw_content, str) or not raw_content.strip():
            raise RuntimeError("Evaluation model returned empty content.")

        try:
            parsed = json.loads(raw_content.strip())
        except json.JSONDecodeError as exc:
            raise RuntimeError("Evaluation JSON parsing failed.") from exc

        if not isinstance(parsed, dict):
            raise RuntimeError("Evaluation payload invalid: JSON must be an object.")

        score_raw = parsed.get("score")
        try:
            score = float(score_raw)
        except (TypeError, ValueError):
            raise RuntimeError("Evaluation payload invalid: score must be a number.")
        if not 0 <= score <= 10:
            raise RuntimeError(
                "Evaluation payload invalid: score must be between 0 and 10."
            )

        analysis = parsed.get("analysis")
        if not isinstance(analysis, str) or not analysis.strip():
            raise RuntimeError(
                "Evaluation payload invalid: analysis must be a non-empty string."
            )

        improvement_prompt = parsed.get("improvement_prompt")
        if not isinstance(improvement_prompt, str):
            raise RuntimeError(
                "Evaluation payload invalid: improvement_prompt must be a string."
            )
        improvement_prompt = improvement_prompt.strip()
        if len(improvement_prompt.splitlines()) > 6:
            raise RuntimeError(
                "Evaluation payload invalid: improvement_prompt must have up to 6 lines."
            )

        return {
            "score": score,
            "analysis": analysis.strip(),
            "improvement_prompt": improvement_prompt,
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

    def _is_relational_topic(self, active_topic: Optional[str]) -> bool:
        if not active_topic:
            return False
        normalized = active_topic.strip().lower()
        return any(topic in normalized for topic in RELATIONAL_TOPICS)

    def _build_dynamic_runtime_prompt(
        self,
        *,
        runtime_main_prompt: str,
        runtime_mode_prompt: str,
        mode_objective: str,
        conversation_mode: str,
        derived_mode: str,
        previous_mode: str,
        progress_state: str,
        previous_progress_state: str,
        spiritual_intensity: str,
        allow_spiritual_context: bool,
        direct_guidance_request: bool,
        repetition_complaint: bool,
        prayer_request_detected: bool,
        live_support_request_detected: bool,
        explicit_user_intent: str,
        strategy_key: str,
        strategy_alternative_forced: bool,
        prayer_cooldown_remaining: int,
        progress_stalled_turns: int,
        force_single_concrete_action: bool,
        practical_mode_forced: bool,
        practical_mode_cooldown_remaining: int,
        active_topic: Optional[str],
        top_topics: str,
        last_user_message: str,
        selected_theme_id: int,
        selected_theme_name: str,
        theme_prompt: Optional[str],
        context_messages: list,
        rag_contexts: list,
    ) -> str:
        runtime_mode = MODE_PASTOR_INSTITUCIONAL

        base_mode_actions = [
            "Só explique processo institucional se o usuário pedir orientação formal explícita.",
            "Em caso de vulnerabilidade emocional, priorize validação e presença antes de qualquer estrutura.",
            "Oriente passos concretos dentro da igreja.",
        ]
        derived_mode_actions = {
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
                "Neste turno, NÃO FAÇA pergunta.",
                "Se for necessário aprofundar, faça isso por reflexão, não por pergunta.",
                "Use tom contemplativo e misericordioso.",
                "Se houver pedido de ajuda direta, ofereça orientação concreta e segura com passos pequenos.",
            ],
            MODE_VULNERABILIDADE_INICIAL: [
                "Não explicar processo.",
                "Não diferenciar esferas.",
                "Não encaminhar.",
                "Não estruturar etapas.",
                "Apenas validar + permitir + 1 pergunta aberta simples.",
            ],
            MODE_WELCOME: [
                "Acolha com sobriedade e convide para continuidade.",
            ],
        }
        mode_actions = list(base_mode_actions)
        mode_actions.extend(
            derived_mode_actions.get(
                derived_mode, ["Escolha a melhor função para este turno."]
            )
        )
        intent_strategy_catalog = {
            INTENT_COMPANHIA: [
                "Pedido explícito de companhia é prioridade máxima do turno.",
                "Confirme permanência no canal e execute acompanhamento por mensagem imediatamente.",
            ],
            INTENT_TEXTO: [
                "Pedido explícito de texto/mensagem: trate como tarefa principal do turno.",
                "Ofereça duas alternativas operacionais e conduza escolha objetiva.",
            ],
            INTENT_PASSO_PRATICO: [
                "Pedido explícito de passo prático: entregar ação concreta no mesmo turno.",
                "Evite explorar demais antes da execução.",
            ],
            INTENT_ORACAO: [
                "Pedido explícito de oração: oração breve é permitida, sem tomar o turno inteiro.",
                "Após oração, adicionar 1 passo prático objetivo.",
            ],
            INTENT_DEFAULT: [
                "Sem pedido explícito: seguir estratégia do modo com progressão tática.",
            ],
        }
        mode_actions.extend(
            intent_strategy_catalog.get(
                explicit_user_intent, intent_strategy_catalog[INTENT_DEFAULT]
            )
        )
        strategy_rules = {
            STRATEGY_SUPPORT_COMPANHIA: "Estratégia ativa: confirmar presença no canal + check-in curto com ação imediata.",
            STRATEGY_OFFER_TEXT_ALTERNATIVES: "Estratégia ativa: propor alternativas objetivas ao texto pronto e pedir escolha.",
            STRATEGY_EXECUTE_PRACTICAL_STEP: "Estratégia ativa: executar passo prático curto e verificável neste turno.",
            STRATEGY_PRAYER_BRIEF: "Estratégia ativa: oração breve (se pedida) + próximo passo prático.",
            STRATEGY_CONTEXT_EXPLORATION: "Estratégia ativa: clarificar contexto com foco em destravar ação.",
            STRATEGY_CONFIRM_AND_EXECUTE: "Estratégia ativa: confirmar decisão do usuário e converter em execução.",
            STRATEGY_CLARIFY_BLOCKER: "Estratégia ativa: identificar bloqueador principal e contornar com plano de 1 passo.",
        }
        mode_actions.append(
            strategy_rules.get(
                strategy_key, strategy_rules[STRATEGY_CONTEXT_EXPLORATION]
            )
        )
        if strategy_alternative_forced:
            mode_actions.append(
                "Troca obrigatória de estratégia ativa: não repetir abordagem dos 2 últimos turnos do assistente."
            )

        spiritual_policy = "Mantenha base espiritual leve (esperança/propósito) sem linguagem explícita."
        if derived_mode == MODE_EXPLORACAO:
            spiritual_policy = (
                "Não ofereça oração/intervenção espiritual direta neste turno; "
                "mantenha foco em investigação concreta."
            )
        elif allow_spiritual_context or spiritual_intensity in {"media", "alta"}:
            spiritual_policy = (
                "Use 1 ou 2 frases espirituais claras e respeitosas, com menção explícita a Deus "
                "e, quando couber, a Jesus, oração ou Palavra, sem imposição."
            )
        if spiritual_intensity == "alta":
            spiritual_policy += (
                " Intensidade alta: linguagem de fé mais presente e pastoral, com esperança evangélica "
                "concreta, sem moralizar."
            )
        if practical_mode_forced:
            spiritual_policy = (
                "Modo prático anti-loop ativo: priorize orientação concreta e objetiva. "
                "Evite linguagem religiosa explícita; se usuário pedir oração, use no máximo 1 frase curta."
            )

        max_sentences = 9
        max_questions = 3
        max_words = 140
        if (
            direct_guidance_request
            or prayer_request_detected
            or live_support_request_detected
        ):
            max_sentences = 6
            max_questions = 1
            max_words = 110

        practical_mode_block = ""
        if practical_mode_forced:
            practical_mode_block = (
                f"\nMODO PRÁTICO ANTI-LOOP ATIVO ({practical_mode_cooldown_remaining} turnos restantes):\n"
                "- Entregue passo concreto neste turno.\n"
                "- Evite repetição de consolo religioso.\n"
                "- Não use mais de 1 frase espiritual curta.\n"
            )

        progress_strategy_block = ""
        if progress_state == PROGRESS_STATE_COLETA:
            progress_strategy_block = (
                "\nESTRATÉGIA DE PROGRESSO (COLETA):\n"
                "- Faça 1 pergunta concreta de contexto OU confirme 1 obstáculo específico.\n"
            )
        elif progress_state == PROGRESS_STATE_PROPOSTA:
            progress_strategy_block = (
                "\nESTRATÉGIA DE PROGRESSO (PROPOSTA):\n"
                "- Apresente 1 proposta tática clara conectada ao pedido do usuário.\n"
            )
        elif progress_state == PROGRESS_STATE_EXECUCAO:
            progress_strategy_block = (
                "\nESTRATÉGIA DE PROGRESSO (EXECUÇÃO):\n"
                "- Entregue uma ação executável agora (roteiro curto de conversa, ensaio guiado ou próximo passo explícito).\n"
            )
        elif progress_state == PROGRESS_STATE_CONFIRMACAO:
            progress_strategy_block = (
                "\nESTRATÉGIA DE PROGRESSO (CONFIRMAÇÃO):\n"
                "- Confirmar o plano em 1 frase e definir 1 check-in objetivo.\n"
            )
        elif progress_state == PROGRESS_STATE_FECHAMENTO:
            progress_strategy_block = (
                "\nESTRATÉGIA DE PROGRESSO (FECHAMENTO):\n"
                "- Encerrar com resumo breve e próximo ponto opcional, sem abrir novos tópicos.\n"
            )

        explicit_request_block = ""
        if (
            direct_guidance_request
            or prayer_request_detected
            or live_support_request_detected
        ):
            explicit_request_block = (
                "\nPEDIDO EXPLÍCITO DETECTADO:\n"
                "- Responda ao pedido explícito antes de investigar causas.\n"
                "- Se houver pedido de oração, inclua oração breve de 1-2 frases neste turno.\n"
                "- Se houver pedido de ligação/mensagem em tempo real, diga claramente o limite do canal e ofereça 2 alternativas práticas distintas.\n"
                "- Depois da resposta direta, ofereça no máximo 1 próximo passo prático.\n"
                "- Ao responder limite de canal, não repita de forma literal a expressão do usuário; use redação mais humana e próxima.\n"
            )

        presencial_request_markers = [
            "visita",
            "visitar",
            "ir comigo",
            "presencial",
            "pessoalmente",
            "na minha casa",
        ]
        has_presencial_request = any(
            marker in (last_user_message or "").lower()
            for marker in presencial_request_markers
        )
        presencial_limit_block = ""
        if has_presencial_request:
            presencial_limit_block = (
                "\nLIMITE DE CANAL (ONLINE-ONLY):\n"
                "- Não ofereça visita presencial, ida ao local ou acompanhamento físico.\n"
                "- Responda com limite claro e acolhedor: apoio apenas por mensagem/ligação online.\n"
                "- Ofereça 1 alternativa online concreta e imediata.\n"
            )

        actionable_artifact_markers = [
            "escreva para mim",
            "escrever para mim",
            "faz uma mensagem",
            "me dá uma mensagem",
            "me de uma mensagem",
            "texto pronto",
            "pronto para copiar",
            "modelo de mensagem",
        ]
        explicit_artifact_request = any(
            marker in (last_user_message or "").lower()
            for marker in actionable_artifact_markers
        )
        artifact_request_block = ""
        if explicit_artifact_request:
            artifact_request_block = (
                "\nPEDIDO DE ARTEFATO DETECTADO:\n"
                "- Não redija mensagem pronta neste primeiro turno.\n"
                "- Ofereça 2 alternativas práticas ao texto pronto (ex.: roteiro de fala em 3 pontos, ensaio de conversa aqui no chat).\n"
                "- Pergunte qual alternativa o usuário prefere executar agora.\n"
            )

        antiloop_block = (
            "\nANTILOOP DE CONTEÚDO:\n"
            "- Se uma frase já apareceu nos últimos 2 turnos do assistente, não repita literal nem com variação mínima.\n"
            "- Aplique bloqueio de n-grama (4 palavras) usando os 2 últimos turnos do assistente como banlist.\n"
            "- Cada resposta deve conter pelo menos 1 elemento novo obrigatório: ação, pergunta ou resumo.\n"
            "- Bloqueie abertura da resposta quando a similaridade semântica da primeira frase for alta em relação às 2 últimas aberturas do assistente.\n"
            "- Não repita oração em turnos consecutivos, exceto se o usuário pedir oração de novo explicitamente.\n"
            f"- Cooldown de oração ativo por {prayer_cooldown_remaining} turnos quando houver oração recente; sem pedido explícito no turno atual, não reintroduza oração.\n"
            "- Limite acolhimento a 1 frase curta no turno.\n"
            "- Se oração não for pedida no turno atual, priorize ação prática concreta.\n"
            "- Pedido explícito do usuário define tarefa principal do turno.\n"
            "- Se mesma estratégia repetir por 2 turnos, no próximo turno aplique estratégia alternativa obrigatória.\n"
            "- Em qualquer inferência sobre causa/motivo, use linguagem condicional: 'pode ser que...'.\n"
            "- Após inferência condicional, peça confirmação explícita ao usuário no mesmo turno.\n"
            "- É proibido afirmar causalidade sem citação textual do que o usuário disse.\n"
            '- Evite rotular padrão psicológico sem validação; use formulação condicional (ex.: "isso pode indicar...") ou peça confirmação.\n'
            "- Após 2 turnos sem avanço prático, entregue 1 ação operacional nova e objetiva neste turno.\n"
        )
        if force_single_concrete_action:
            antiloop_block += (
                f"- Gatilho ativo: {progress_stalled_turns} turnos sem avanço.\n"
                "- Neste turno, entregue EXATAMENTE uma ação concreta e verificável.\n"
                "- Não ofereça múltiplas ações concorrentes.\n"
            )

        distress_markers = [
            "chor",
            "desmoron",
            "arrasad",
            "não aguento",
            "nao aguento",
            "peito apertado",
            "desespero",
        ]
        has_high_distress = any(
            marker in (last_user_message or "").lower() for marker in distress_markers
        )
        distress_block = ""
        if has_high_distress:
            distress_block = (
                "\nSENSIBILIDADE DE ABERTURA:\n"
                '- Não use abertura celebratória ou potencialmente minimizadora (ex.: "que bom", "é bom saber").\n'
                "- Comece validando a dor concreta do usuário com linguagem sóbria.\n"
            )

        repetition_block = ""
        if repetition_complaint:
            repetition_block = (
                "\nUSUÁRIO SINALIZOU REPETIÇÃO: não repita pergunta; "
                "entregue orientação prática nova e específica para este caso.\n"
            )

        assistant_openers = []
        for msg in context_messages:
            if msg.role != "assistant":
                continue
            first_sentence = self._split_sentences(msg.content)
            if not first_sentence:
                continue
            assistant_openers.append(first_sentence[0].strip())
        assistant_openers_block = ""
        if assistant_openers:
            assistant_openers_block = (
                "\nEVITE REPETIR ABERTURAS RECENTES DO ASSISTENTE:\n"
            )
            for opener in assistant_openers[-3:]:
                assistant_openers_block += f"- {opener}\n"

        active_topic_block = ""
        if active_topic:
            active_topic_block = f"\nTÓPICO ATIVO: {active_topic}\n"

        top_topics_block = ""
        if top_topics:
            top_topics_block = f"\nTÓPICOS RECENTES: {top_topics}\n"

        theme_block = (
            f"\nTEMA IDENTIFICADO DA MENSAGEM DO USUÁRIO: "
            f"{selected_theme_name} ({selected_theme_id})\n"
        )
        theme_instruction_block = ""
        if theme_prompt:
            theme_instruction_block = f"\nINSTRUÇÃO TEMÁTICA:\n{theme_prompt}\n"

        mode_actions_block = "\nFUNÇÕES PRIORITÁRIAS DESTE TURNO:\n"
        for action in mode_actions:
            mode_actions_block += f"- {action}\n"

        history_block = ""
        for msg in context_messages:
            history_block += f"{msg.role.upper()}: {msg.content}\n"

        rag_block = ""
        if rag_contexts:
            rag_block += "\nRAG CONTEXT AUXILIAR:\n"
            for rag in rag_contexts:
                rag_block += f"- {rag}\n"

        runtime_template_context = {
            "runtime_mode_prompt": runtime_mode_prompt,
            "runtime_mode": runtime_mode,
            "derived_mode": derived_mode,
            "previous_mode": previous_mode,
            "progress_state": progress_state,
            "previous_progress_state": previous_progress_state,
            "mode_objective": mode_objective,
            "spiritual_intensity": spiritual_intensity,
            "max_sentences": max_sentences,
            "max_words": max_words,
            "max_questions": max_questions,
            "spiritual_policy": spiritual_policy,
            "practical_mode_block": practical_mode_block,
            "progress_strategy_block": progress_strategy_block,
            "explicit_request_block": explicit_request_block,
            "presencial_limit_block": presencial_limit_block,
            "artifact_request_block": artifact_request_block,
            "antiloop_block": antiloop_block,
            "distress_block": distress_block,
            "repetition_block": repetition_block,
            "assistant_openers_block": assistant_openers_block,
            "active_topic_block": active_topic_block,
            "top_topics_block": top_topics_block,
            "theme_block": theme_block,
            "theme_instruction_block": theme_instruction_block,
            "mode_actions_block": mode_actions_block,
            "last_user_message": last_user_message,
            "history_block": history_block,
            "rag_block": rag_block,
        }
        return self._render_runtime_main_prompt(
            runtime_main_prompt=runtime_main_prompt, context=runtime_template_context
        )

    def _render_runtime_main_prompt(
        self, *, runtime_main_prompt: str, context: Dict[str, Any]
    ) -> str:
        required_fields = []
        for _, field_name, _, _ in Formatter().parse(runtime_main_prompt):
            if field_name:
                required_fields.append(field_name)
        for field_name in required_fields:
            if field_name not in context:
                raise RuntimeError(
                    f"Runtime main prompt missing required context key '{field_name}'."
                )
        try:
            return runtime_main_prompt.format(**context).strip()
        except KeyError as exc:
            raise RuntimeError(
                f"Runtime main prompt rendering missing key '{exc.args[0]}'."
            ) from exc
        except ValueError as exc:
            raise RuntimeError(
                f"Runtime main prompt has invalid format syntax: {exc}."
            ) from exc

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

    def _last_assistant_runtime_metadata(self, queryset) -> Dict[str, Any]:
        last_assistant = (
            queryset.filter(role="assistant").order_by("-created_at").first()
        )
        if not last_assistant:
            return {}
        payload = getattr(last_assistant, "ollama_prompt", None)
        if not isinstance(payload, dict):
            return {}
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            return {}
        return metadata

    def _detect_progress_state(
        self,
        *,
        last_user_message: str,
        previous_progress_state: str,
        direct_guidance_request: bool,
        explicit_user_intent: str,
    ) -> str:
        normalized = (last_user_message or "").lower()

        closing_markers = [
            "obrigado",
            "obrigada",
            "já ajudou",
            "ja ajudou",
            "era isso",
            "vamos encerrar",
            "pode encerrar",
            "tá bom por hoje",
            "ta bom por hoje",
        ]
        if any(marker in normalized for marker in closing_markers):
            return PROGRESS_STATE_FECHAMENTO

        execution_done_markers = [
            "fiz",
            "feito",
            "enviei",
            "mande",
            "agendei",
            "combinei",
            "coloquei em prática",
            "coloquei em pratica",
        ]
        if any(marker in normalized for marker in execution_done_markers):
            return PROGRESS_STATE_CONFIRMACAO

        confirmation_markers = [
            "sim",
            "aceito",
            "topo",
            "vou fazer",
            "vou tentar",
            "combinado",
            "fechado",
            "pode ser",
        ]
        if any(marker in normalized for marker in confirmation_markers):
            if previous_progress_state in {
                PROGRESS_STATE_EXECUCAO,
                PROGRESS_STATE_CONFIRMACAO,
            }:
                return PROGRESS_STATE_CONFIRMACAO
            if previous_progress_state == PROGRESS_STATE_PROPOSTA:
                return PROGRESS_STATE_EXECUCAO

        if direct_guidance_request:
            if previous_progress_state in {
                PROGRESS_STATE_COLETA,
                PROGRESS_STATE_PROPOSTA,
            }:
                return PROGRESS_STATE_EXECUCAO
            return PROGRESS_STATE_EXECUCAO

        if explicit_user_intent in {
            INTENT_COMPANHIA,
            INTENT_TEXTO,
            INTENT_PASSO_PRATICO,
        }:
            if previous_progress_state == PROGRESS_STATE_COLETA:
                return PROGRESS_STATE_PROPOSTA
            if previous_progress_state == PROGRESS_STATE_PROPOSTA:
                return PROGRESS_STATE_EXECUCAO

        if previous_progress_state in VALID_PROGRESS_STATES:
            return previous_progress_state
        return PROGRESS_STATE_COLETA

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

        last_runtime_metadata = self._last_assistant_runtime_metadata(queryset)
        previous_progress_state = str(
            last_runtime_metadata.get("progress_state", PROGRESS_STATE_COLETA)
        )
        if previous_progress_state not in VALID_PROGRESS_STATES:
            previous_progress_state = PROGRESS_STATE_COLETA
        previous_progress_metric = last_runtime_metadata.get("progress_metric", {})
        if not isinstance(previous_progress_metric, dict):
            previous_progress_metric = {}
        previous_progress_stalled_turns = last_runtime_metadata.get(
            "progress_stalled_turns", 0
        )
        try:
            previous_progress_stalled_turns = int(previous_progress_stalled_turns)
        except (TypeError, ValueError):
            previous_progress_stalled_turns = 0
        previous_practical_cooldown = last_runtime_metadata.get(
            "practical_mode_cooldown_remaining", 0
        )
        try:
            previous_practical_cooldown = int(previous_practical_cooldown)
        except (TypeError, ValueError):
            previous_practical_cooldown = 0
        practical_mode_cooldown_remaining = max(previous_practical_cooldown - 1, 0)
        previous_prayer_cooldown = last_runtime_metadata.get(
            "prayer_cooldown_remaining", 0
        )
        try:
            previous_prayer_cooldown = int(previous_prayer_cooldown)
        except (TypeError, ValueError):
            previous_prayer_cooldown = 0
        prayer_cooldown_remaining = max(previous_prayer_cooldown - 1, 0)
        previous_strategy_key = str(last_runtime_metadata.get("strategy_key", "") or "")
        previous_strategy_repetition_count = last_runtime_metadata.get(
            "strategy_repetition_count", 0
        )
        try:
            previous_strategy_repetition_count = int(previous_strategy_repetition_count)
        except (TypeError, ValueError):
            previous_strategy_repetition_count = 0

        is_first_message = queryset.filter(role="assistant").count() == 0
        signals = detect_user_signals(last_user_message)
        direct_guidance_request = bool(signals.get("guidance_request"))
        repetition_complaint = bool(signals.get("repetition_complaint"))
        if repetition_complaint:
            direct_guidance_request = True
        # 🔥 OVERRIDE: pedido explícito de oração tem prioridade máxima
        prayer_request_detected = any(
            phrase in last_user_message.lower() for phrase in PRAYER_REQUEST_MARKERS
        )
        live_support_request_detected = any(
            phrase in last_user_message.lower()
            for phrase in LIVE_SUPPORT_REQUEST_MARKERS
        )

        if prayer_request_detected or live_support_request_detected:
            direct_guidance_request = True
        explicit_user_intent = self._detect_explicit_user_intent(last_user_message)
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
        assistant_similarity_loop = False
        if len(recent_assistant_messages) >= 2:
            assistant_similarity_loop = (
                semantic_similarity(
                    recent_assistant_messages[-1], recent_assistant_messages[-2]
                )
                > LOOP_SIMILARITY_THRESHOLD
            )
            loop_detected = loop_detected or assistant_similarity_loop

        if assistant_similarity_loop:
            practical_mode_cooldown_remaining = LOOP_PRACTICAL_COOLDOWN_TURNS
        practical_mode_forced = practical_mode_cooldown_remaining > 0

        progress_state = self._detect_progress_state(
            last_user_message=last_user_message,
            previous_progress_state=previous_progress_state,
            direct_guidance_request=direct_guidance_request,
            explicit_user_intent=explicit_user_intent,
        )

        conversation_mode = choose_conversation_mode(
            previous_mode=previous_mode,
            is_first_message=is_first_message,
            loop_detected=loop_detected,
            has_new_info=new_information,
            repeated_user_pattern=repeated_user_pattern,
            signals=signals,
        )
        institutional_request = any(
            phrase in last_user_message.lower()
            for phrase in [
                "como fazer",
                "como realizar",
                "me instrua",
                "qual o processo",
                "como funciona",
                "preciso saber como",
                "me explique o processo",
            ]
        )
        if institutional_request:
            conversation_mode = MODE_PASTOR_INSTITUCIONAL
        elif prayer_request_detected or live_support_request_detected:
            conversation_mode = MODE_ORIENTACAO
        elif force_deep_presence:
            conversation_mode = MODE_PRESENCA_PROFUNDA
        if practical_mode_forced:
            conversation_mode = MODE_ORIENTACAO
            direct_guidance_request = True
            progress_state = PROGRESS_STATE_EXECUCAO
            allow_spiritual_context = False
        strategy_selection = self._select_strategy_for_intent(
            explicit_user_intent=explicit_user_intent,
            direct_guidance_request=direct_guidance_request,
            previous_strategy_key=previous_strategy_key,
            previous_strategy_repetition_count=previous_strategy_repetition_count,
        )
        strategy_key = strategy_selection["strategy_key"]
        strategy_alternative_forced = strategy_selection["strategy_alternative_forced"]
        spiritual_intensity = choose_spiritual_intensity(
            mode=conversation_mode,
            spiritual_context=explicit_spiritual_context,
            high_spiritual_need=high_spiritual_need,
        )
        if progress_state == PROGRESS_STATE_EXECUCAO and not prayer_request_detected:
            spiritual_intensity = "leve"
        if practical_mode_forced:
            spiritual_intensity = "leve"
        derived_mode = conversation_mode
        force_single_concrete_action = (
            previous_progress_stalled_turns >= STALL_TURNS_FORCE_ACTION
        )
        prayer_allowed_this_turn = prayer_request_detected or (
            prayer_cooldown_remaining == 0
        )
        return {
            "previous_mode": previous_mode,
            "conversation_mode": conversation_mode,
            "derived_mode": derived_mode,
            "progress_state": progress_state,
            "previous_progress_state": previous_progress_state,
            "spiritual_intensity": spiritual_intensity,
            "direct_guidance_request": direct_guidance_request,
            "repetition_complaint": repetition_complaint,
            "allow_spiritual_context": allow_spiritual_context,
            "loop_detected": loop_detected,
            "assistant_similarity_loop": assistant_similarity_loop,
            "practical_mode_forced": practical_mode_forced,
            "practical_mode_cooldown_remaining": practical_mode_cooldown_remaining,
            "deep_presence_trigger": deep_presence_trigger,
            "prayer_request_detected": prayer_request_detected,
            "live_support_request_detected": live_support_request_detected,
            "explicit_user_intent": explicit_user_intent,
            "strategy_key": strategy_key,
            "strategy_alternative_forced": strategy_alternative_forced,
            "previous_strategy_key": previous_strategy_key,
            "previous_strategy_repetition_count": previous_strategy_repetition_count,
            "prayer_cooldown_remaining": prayer_cooldown_remaining,
            "prayer_allowed_this_turn": prayer_allowed_this_turn,
            "previous_progress_metric": previous_progress_metric,
            "previous_progress_stalled_turns": previous_progress_stalled_turns,
            "force_single_concrete_action": force_single_concrete_action,
        }

    def _build_response_prompt(
        self,
        *,
        profile: Profile,
        queryset,
        last_person_message: Message,
        generation_state: Dict[str, Any],
        active_topic: Optional[str],
        selected_theme: Theme,
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
        rag_contexts = self._rag_service.retrieve(
            query=last_person_message.content,
            theme_id=selected_theme.id,
            limit=3,
        )
        runtime_selection = self._prompt_registry.get_runtime_prompt_for_mode(
            generation_state["derived_mode"]
        )
        runtime_main_selection = self._prompt_registry.get_runtime_main_prompt()
        objective_selection = self._prompt_registry.get_runtime_mode_objective_for_mode(
            generation_state["derived_mode"]
        )
        mode_objective = (objective_selection.content or "").strip()
        if not mode_objective:
            raise RuntimeError(
                (
                    "Runtime mode objective is empty for mode "
                    f"'{generation_state['derived_mode']}'."
                )
            )

        return self._build_dynamic_runtime_prompt(
            runtime_main_prompt=runtime_main_selection.content,
            runtime_mode_prompt=runtime_selection.content,
            mode_objective=mode_objective,
            conversation_mode=generation_state["conversation_mode"],
            derived_mode=generation_state["derived_mode"],
            previous_mode=generation_state["previous_mode"],
            progress_state=generation_state["progress_state"],
            previous_progress_state=generation_state["previous_progress_state"],
            spiritual_intensity=generation_state["spiritual_intensity"],
            allow_spiritual_context=generation_state["allow_spiritual_context"],
            direct_guidance_request=generation_state["direct_guidance_request"],
            repetition_complaint=generation_state["repetition_complaint"],
            prayer_request_detected=generation_state["prayer_request_detected"],
            live_support_request_detected=generation_state[
                "live_support_request_detected"
            ],
            explicit_user_intent=generation_state["explicit_user_intent"],
            strategy_key=generation_state["strategy_key"],
            strategy_alternative_forced=generation_state["strategy_alternative_forced"],
            prayer_cooldown_remaining=generation_state["prayer_cooldown_remaining"],
            progress_stalled_turns=generation_state["previous_progress_stalled_turns"],
            force_single_concrete_action=generation_state[
                "force_single_concrete_action"
            ],
            practical_mode_forced=generation_state["practical_mode_forced"],
            practical_mode_cooldown_remaining=generation_state[
                "practical_mode_cooldown_remaining"
            ],
            active_topic=active_topic,
            top_topics=top_topics,
            last_user_message=last_person_message.content,
            selected_theme_id=selected_theme.id,
            selected_theme_name=selected_theme.name,
            theme_prompt=selected_theme.meta_prompt,
            context_messages=context_messages,
            rag_contexts=rag_contexts,
        )

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

    def generate_response_message(
        self,
        profile: Profile,
        channel: str,
        forced_theme: Optional[Theme] = None,
    ) -> str:
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
        if forced_theme is not None:
            selected_theme = forced_theme
            if last_person_message.theme_id != forced_theme.id:
                last_person_message.theme = forced_theme
                last_person_message.save(update_fields=["theme"])
        else:
            selected_theme = self._classify_and_persist_message_theme(
                last_person_message
            )
        prompt_aux = self._build_response_prompt(
            profile=profile,
            queryset=queryset,
            last_person_message=last_person_message,
            generation_state=generation_state,
            active_topic=active_topic,
            selected_theme=selected_theme,
        )

        model_name = WACHAT_RESPONSE_MODEL
        max_completion_tokens = FIXED_RESPONSE_MAX_COMPLETION_TOKENS
        client = getattr(self._llm_service, "client", None)
        if client is None:
            raise RuntimeError(
                "OpenAI client is not available on configured LLM service."
            )

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

        selected_temperature = FIXED_TEMPERATURE
        selected_max_completion_tokens = max_completion_tokens
        regeneration_counter = 0
        semantic_loop_regenerations = 0
        system_prompt_selection = self._prompt_registry.get_system_prompt()
        selected_system_prompt = system_prompt_selection.content
        runtime_prompt_selection = self._prompt_registry.get_runtime_prompt_for_mode(
            generation_state["derived_mode"]
        )

        request_messages = [
            {"role": "system", "content": selected_system_prompt},
            {"role": "user", "content": prompt_aux},
        ]
        system_messages_count = sum(
            1 for item in request_messages if item.get("role") == "system"
        )
        if system_messages_count != 1:
            raise RuntimeError(
                "Invalid OpenAI request: expected exactly one system message."
            )
        request_kwargs = {
            "model": model_name,
            "messages": request_messages,
            "max_completion_tokens": selected_max_completion_tokens,
            "reasoning_effort": "low",
            "timeout": FIXED_TIMEOUT_SECONDS,
            "temperature": selected_temperature,
            "n": 2,
        }
        response = client.chat.completions.create(**request_kwargs)
        response_metadata = _usage_metadata(response)
        response_metadata["round"] = 1
        response_rounds_metadata = [response_metadata]

        def _extract_text_from_choice(choice: Any) -> str:
            message = getattr(choice, "message", None)
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
            return ""

        attempts: List[Dict[str, Any]] = []
        best_attempt: Optional[Dict[str, Any]] = None
        selected_runtime_prompt = prompt_aux
        selected_response_metadata = response_metadata
        banned_ngrams = self._build_recent_assistant_ngram_ban(
            recent_assistant_messages
        )
        previous_progress_metric = generation_state["previous_progress_metric"]
        force_single_concrete_action = generation_state["force_single_concrete_action"]

        current_runtime_prompt = prompt_aux
        for round_number in range(1, MAX_SCORE_REFINEMENT_ROUNDS + 2):
            if round_number == 1:
                current_response = response
                current_metadata = response_metadata
            else:
                if not best_attempt:
                    raise RuntimeError(
                        "Cannot refine response without evaluated attempts."
                    )
                current_runtime_prompt = self._build_refinement_runtime_prompt(
                    base_runtime_prompt=prompt_aux,
                    round_number=round_number,
                    score=float(best_attempt["score"]),
                    analysis=str(best_attempt["analysis"]),
                    improvement_prompt=str(best_attempt["improvement_prompt"]),
                )
                refined_messages = [
                    {"role": "system", "content": selected_system_prompt},
                    {"role": "user", "content": current_runtime_prompt},
                ]
                refined_system_count = sum(
                    1 for item in refined_messages if item.get("role") == "system"
                )
                if refined_system_count != 1:
                    raise RuntimeError(
                        "Invalid OpenAI refinement request: expected exactly one system message."
                    )
                refined_kwargs = {
                    "model": model_name,
                    "messages": refined_messages,
                    "max_completion_tokens": selected_max_completion_tokens,
                    "reasoning_effort": "low",
                    "timeout": FIXED_TIMEOUT_SECONDS,
                    "temperature": selected_temperature,
                    "n": 2,
                }
                current_response = client.chat.completions.create(**refined_kwargs)
                current_metadata = _usage_metadata(current_response)
                current_metadata["round"] = round_number
                response_rounds_metadata.append(current_metadata)
                regeneration_counter += 1

            choices = getattr(current_response, "choices", None) or []
            if len(choices) < 2:
                raise RuntimeError("OpenAI did not return the expected 2 candidates.")
            non_empty_candidates_in_round = 0
            evaluated_candidates_in_round = 0
            for regen_attempt in range(0, MAX_INFERENCE_REGEN_PER_ROUND + 1):
                for attempt_number, choice in enumerate(choices[:2], start=1):
                    assistant_text_candidate = _extract_text_from_choice(choice)
                    logger.info(
                        "Raw assistant response received profile_id=%s channel=%s round=%s content=%r",
                        profile.id,
                        channel,
                        round_number,
                        assistant_text_candidate,
                    )
                    if not assistant_text_candidate:
                        logger.warning(
                            "Empty assistant candidate ignored profile_id=%s channel=%s round=%s attempt=%s",
                            profile.id,
                            channel,
                            round_number,
                            attempt_number,
                        )
                        continue
                    non_empty_candidates_in_round += 1
                    if self._candidate_has_banned_ngram(
                        assistant_text_candidate, banned_ngrams
                    ):
                        logger.warning(
                            "Candidate blocked by ngram ban profile_id=%s channel=%s round=%s attempt=%s",
                            profile.id,
                            channel,
                            round_number,
                            attempt_number,
                        )
                        continue
                    opening_similarity = self._candidate_opening_similarity(
                        assistant_text_candidate, recent_assistant_messages
                    )
                    if opening_similarity >= OPENING_SIMILARITY_BLOCK_THRESHOLD:
                        logger.warning(
                            "Candidate blocked by opening similarity profile_id=%s channel=%s round=%s attempt=%s similarity=%.3f",
                            profile.id,
                            channel,
                            round_number,
                            attempt_number,
                            opening_similarity,
                        )
                        continue
                    if not self._candidate_has_required_new_element(
                        assistant_text_candidate
                    ):
                        logger.warning(
                            "Candidate blocked: missing required new element profile_id=%s channel=%s round=%s attempt=%s",
                            profile.id,
                            channel,
                            round_number,
                            attempt_number,
                        )
                        continue
                    candidate_has_prayer = self._contains_prayer_language(
                        assistant_text_candidate
                    )
                    candidate_has_action = self._candidate_has_practical_action(
                        assistant_text_candidate
                    )
                    if (
                        candidate_has_prayer
                        and not generation_state["prayer_request_detected"]
                        and generation_state["prayer_cooldown_remaining"] > 0
                    ):
                        logger.warning(
                            "Candidate blocked by prayer cooldown profile_id=%s channel=%s round=%s attempt=%s",
                            profile.id,
                            channel,
                            round_number,
                            attempt_number,
                        )
                        continue
                    if (
                        candidate_has_prayer
                        and not generation_state["prayer_request_detected"]
                        and not candidate_has_action
                    ):
                        logger.warning(
                            "Candidate blocked: prayer without practical action profile_id=%s channel=%s round=%s attempt=%s",
                            profile.id,
                            channel,
                            round_number,
                            attempt_number,
                        )
                        continue
                    empathy_stats = self._empathy_sentence_stats(
                        assistant_text_candidate
                    )
                    if empathy_stats["count"] > MAX_EMPATHY_SENTENCES_PER_RESPONSE:
                        logger.warning(
                            "Candidate blocked: excessive empathy sentences profile_id=%s channel=%s round=%s attempt=%s count=%s",
                            profile.id,
                            channel,
                            round_number,
                            attempt_number,
                            empathy_stats["count"],
                        )
                        continue
                    if (
                        empathy_stats["count"] == 1
                        and empathy_stats["max_words"] > MAX_EMPATHY_SENTENCE_WORDS
                    ):
                        logger.warning(
                            "Candidate blocked: empathy sentence too long profile_id=%s channel=%s round=%s attempt=%s words=%s",
                            profile.id,
                            channel,
                            round_number,
                            attempt_number,
                            empathy_stats["max_words"],
                        )
                        continue
                    if self._has_strong_inference(assistant_text_candidate):
                        has_citation = self._contains_user_citation(
                            assistant_text_candidate, last_person_message.content
                        )
                        has_conditional_confirmation = (
                            self._has_conditional_inference_confirmation(
                                assistant_text_candidate
                            )
                        )
                        if not has_citation or not has_conditional_confirmation:
                            logger.warning(
                                "Candidate blocked: strong inference without citation/confirmation profile_id=%s channel=%s round=%s attempt=%s",
                                profile.id,
                                channel,
                                round_number,
                                attempt_number,
                            )
                            continue
                    candidate_progress_metric = self._extract_progress_metric(
                        assistant_text_candidate
                    )
                    if force_single_concrete_action:
                        concrete_actions = self._count_concrete_actions(
                            assistant_text_candidate
                        )
                        if concrete_actions != 1:
                            logger.warning(
                                "Candidate blocked: force single concrete action profile_id=%s channel=%s round=%s attempt=%s actions=%s",
                                profile.id,
                                channel,
                                round_number,
                                attempt_number,
                                concrete_actions,
                            )
                            continue
                        if not self._progress_advanced(
                            previous_progress_metric, candidate_progress_metric
                        ):
                            logger.warning(
                                "Candidate blocked: no progress advance under force mode profile_id=%s channel=%s round=%s attempt=%s",
                                profile.id,
                                channel,
                                round_number,
                                attempt_number,
                            )
                            continue

                    evaluation = self._evaluate_response(
                        user_message=last_person_message.content,
                        assistant_response=assistant_text_candidate,
                    )
                    score = evaluation["score"]
                    analysis = evaluation["analysis"]
                    improvement_prompt = evaluation["improvement_prompt"]
                    logger.info(
                        "Evaluation round %s attempt %s | score=%s",
                        round_number,
                        attempt_number,
                        score,
                    )
                    logger.info("Improvement prompt: %s", improvement_prompt)

                    attempt = {
                        "round": round_number,
                        "attempt": attempt_number,
                        "response": assistant_text_candidate,
                        "score": score,
                        "analysis": analysis,
                        "improvement_prompt": improvement_prompt,
                        "progress_metric": candidate_progress_metric,
                    }
                    attempts.append(attempt)
                    evaluated_candidates_in_round += 1
                    if not best_attempt or score > float(best_attempt["score"]):
                        best_attempt = attempt
                        selected_runtime_prompt = current_runtime_prompt
                        selected_response_metadata = current_metadata
                if evaluated_candidates_in_round > 0:
                    break
                if regen_attempt >= MAX_INFERENCE_REGEN_PER_ROUND:
                    break
                if round_number == 1:
                    current_response = client.chat.completions.create(**request_kwargs)
                else:
                    current_response = client.chat.completions.create(**refined_kwargs)
                current_metadata = _usage_metadata(current_response)
                current_metadata["round"] = round_number
                current_metadata["regenerated_after_guard"] = True
                response_rounds_metadata.append(current_metadata)
                regeneration_counter += 1
                choices = getattr(current_response, "choices", None) or []
                if len(choices) < 2:
                    raise RuntimeError(
                        "OpenAI did not return the expected 2 candidates."
                    )

            if non_empty_candidates_in_round == 0:
                raise RuntimeError("OpenAI returned empty assistant content.")
            if evaluated_candidates_in_round == 0:
                raise RuntimeError(
                    "Post-generation inference guard blocked all assistant candidates."
                )
            if not best_attempt:
                raise RuntimeError("No evaluated attempts were produced.")
            if float(best_attempt["score"]) >= TARGET_RESPONSE_SCORE:
                break
            if (
                float(best_attempt["score"]) > LOW_SCORE_REFINEMENT_THRESHOLD
                and round_number >= 1
            ):
                break

        if not best_attempt:
            raise RuntimeError("No attempts available for response selection.")
        assistant_text = best_attempt["response"]
        best_score = best_attempt["score"]
        logger.info("Selected best score=%s", best_score)
        selected_has_prayer = self._contains_prayer_language(assistant_text)
        next_prayer_cooldown_remaining = generation_state["prayer_cooldown_remaining"]
        if selected_has_prayer:
            next_prayer_cooldown_remaining = PRAYER_COOLDOWN_TURNS
        selected_progress_metric = best_attempt.get("progress_metric")
        if not isinstance(selected_progress_metric, dict):
            selected_progress_metric = self._extract_progress_metric(assistant_text)
        progress_advanced = self._progress_advanced(
            generation_state["previous_progress_metric"], selected_progress_metric
        )
        next_progress_stalled_turns = (
            0
            if progress_advanced
            else generation_state["previous_progress_stalled_turns"] + 1
        )
        next_strategy_repetition_count = 1
        if (
            generation_state["previous_strategy_key"]
            == generation_state["strategy_key"]
        ):
            next_strategy_repetition_count = (
                generation_state["previous_strategy_repetition_count"] + 1
            )

        loop_counter = 1 if generation_state["loop_detected"] else 0
        self._save_runtime_counters(
            profile=profile,
            conversation_mode=generation_state["conversation_mode"],
            loop_counter=loop_counter,
            regeneration_counter=regeneration_counter,
        )

        chunks = self._build_assistant_message_chunks(
            text=assistant_text, conversation_mode=generation_state["derived_mode"]
        )
        response_payload = {
            "provider": "openai",
            "request_params": {
                "model": model_name,
                "temperature": selected_temperature,
                "max_completion_tokens": selected_max_completion_tokens,
            },
            "prompts": {
                "system_prompt": selected_system_prompt,
                "runtime_prompt": selected_runtime_prompt,
                "selected_theme": {
                    "id": selected_theme.id,
                    "name": selected_theme.name,
                },
                "versions": {
                    "system_component": system_prompt_selection.component_key,
                    "system_version": system_prompt_selection.version,
                    "runtime_component": runtime_prompt_selection.component_key,
                    "runtime_version": runtime_prompt_selection.version,
                },
            },
            "payload": {
                "messages": [
                    {"role": "system", "content_ref": "prompts.system_prompt"},
                    {"role": "user", "content_ref": "prompts.runtime_prompt"},
                ],
                "message_roles": [m.get("role") for m in request_messages],
                "reasoning_effort": "low",
                "timeout": FIXED_TIMEOUT_SECONDS,
            },
            "metadata": {
                **selected_response_metadata,
                "regeneration_counter": regeneration_counter,
                "semantic_loop_regenerations": semantic_loop_regenerations,
                "response_rounds": response_rounds_metadata,
                "progress_state": generation_state["progress_state"],
                "previous_progress_state": generation_state["previous_progress_state"],
                "practical_mode_forced": generation_state["practical_mode_forced"],
                "practical_mode_cooldown_remaining": generation_state[
                    "practical_mode_cooldown_remaining"
                ],
                "prayer_cooldown_remaining": next_prayer_cooldown_remaining,
                "explicit_user_intent": generation_state["explicit_user_intent"],
                "strategy_key": generation_state["strategy_key"],
                "strategy_alternative_forced": generation_state[
                    "strategy_alternative_forced"
                ],
                "strategy_repetition_count": next_strategy_repetition_count,
                "progress_metric": selected_progress_metric,
                "progress_advanced": progress_advanced,
                "progress_stalled_turns": next_progress_stalled_turns,
            },
            "evaluation": {
                "attempts": attempts,
                "best_score": best_score,
            },
            "delivery": {
                "parts_count": len(chunks),
                "mode": ("multi_message" if len(chunks) > 1 else "single_message"),
            },
        }
        first_message = None
        for index, chunk in enumerate(chunks):
            payload = response_payload if index == 0 else None
            message = Message.objects.create(
                profile=profile,
                role="assistant",
                content=chunk,
                channel=channel,
                ollama_prompt=payload,
                score=float(best_score),
                bot_mode=generation_state["derived_mode"],
                theme=selected_theme,
                block_root=first_message,
            )
            if first_message is None:
                first_message = message
                message.block_root = message
                message.save(update_fields=["block_root"])
        return assistant_text

    def _classify_and_persist_message_theme(self, message: Message) -> Theme:
        theme_id = self._theme_classifier.classify(message.content)
        theme = Theme.objects.filter(id=theme_id).first()
        if not theme:
            raise RuntimeError(f"Theme '{theme_id}' not found in database.")
        if message.theme_id != theme.id:
            message.theme = theme
            message.save(update_fields=["theme"])
        return theme

    def classify_and_persist_message_theme(self, message: Message) -> Theme:
        return self._classify_and_persist_message_theme(message)

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
            max_tokens=FIXED_GENDER_INFERENCE_MAX_COMPLETION_TOKENS,
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

        welcome_template = self._prompt_registry.get_active_prompt(
            "welcome.generator"
        ).content
        system_prompt = welcome_template.format(
            name=profile.name,
            gender_context=gender_context,
        )

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

        message = Message.objects.create(
            profile=profile,
            role="assistant",
            content=response,
            channel=channel,
            ollama_prompt=welcome_payload,
            bot_mode=MODE_WELCOME,
        )
        message.block_root = message
        message.save(update_fields=["block_root"])
        return message

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
            max_tokens=FIXED_THEME_PROMPT_MAX_COMPLETION_TOKENS,
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
            max_tokens=FIXED_SIMULATION_ANALYSIS_MAX_COMPLETION_TOKENS,
        )

        analysis = response_text
        logger.info("Generated critical analysis of simulated conversation")
        return analysis
