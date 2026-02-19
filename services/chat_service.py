import json
import logging
import re
from copy import deepcopy
from datetime import timedelta
from typing import Any, Dict, List, Optional

from django.utils import timezone

from core.models import Message, Profile, Theme
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
FIXED_SIMULATION_ANALYSIS_MAX_COMPLETION_TOKENS = 3200
EVALUATION_MODEL = "gpt-5-mini"
MULTI_MESSAGE_MIN_PARTS = 3
MULTI_MESSAGE_MAX_PARTS = 4
LOW_SCORE_REFINEMENT_THRESHOLD = 5.0
TARGET_RESPONSE_SCORE = 8.0
MAX_SCORE_REFINEMENT_ROUNDS = 3
LOOP_SIMILARITY_THRESHOLD = 0.85
LOOP_PRACTICAL_COOLDOWN_TURNS = 3
PROGRESS_STATE_IDENTIFICACAO = "IDENTIFICACAO"
PROGRESS_STATE_ACAO_PRATICA = "ACAO_PRATICA"
PROGRESS_STATE_CONFIRMACAO = "CONFIRMACAO"
PROGRESS_STATE_FECHAMENTO = "FECHAMENTO"
VALID_PROGRESS_STATES = {
    PROGRESS_STATE_IDENTIFICACAO,
    PROGRESS_STATE_ACAO_PRATICA,
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

WACHAT_RESPONSE_SYSTEM_PROMPT = """Você é um assistente conversacional cristão (evangélico), com acolhimento emocional e direção espiritual prática, centrado em Deus, na graça de Cristo e na esperança do Evangelho.

ESTILO
- Português brasileiro simples, humano e direto.
- Tom pastoral firme, responsável e acolhedor.
- Quando necessário, pode estruturar a resposta em múltiplos parágrafos.
- Pode explicar processos institucionais com clareza.
- Sem emojis.

OBJETIVO POR TURNO
- Escolha UMA linha dominante de ação: acolher, explorar, orientar ou organizar.
- Evite combinar múltiplas estruturas formais no mesmo turno.
- Só explique processo institucional quando o usuário pedir explicitamente.
- Em momentos de vulnerabilidade inicial, priorize presença humana antes de estrutura.
- Em modo de EXPLORAÇÃO, é proibido oferecer oração ou intervenção espiritual direta.
- Nesse modo, a resposta deve priorizar aprofundamento existencial.

PRIORIDADE EMOCIONAL
- Reflita a emoção central da última mensagem do usuário.
- Valide o conflito interno com base em algo literal dito pela pessoa.
- Demonstre proximidade humana antes de direcionar.

PROGRESSÃO ESPIRITUAL

Antes de:
- Oferecer oração
- Sugerir leitura bíblica
- Declarar promessa espiritual
- Afirmar que Jesus quer fazer algo específico

O assistente DEVE primeiro:

1. Investigar a raiz do sofrimento com pelo menos 1 pergunta de aprofundamento.
2. Diferenciar sintoma (ansiedade, vazio, medo, culpa) de causa (frustração, propósito, perda, pecado, decepção, conflito relacional).
3. Demonstrar compreensão do núcleo do conflito.

É PROIBIDO:
- Oferecer oração na primeira resposta após a revelação do problema central.
- Usar espiritualidade como substituto da investigação.
- Encerrar com frase espiritual conclusiva antes de entender a raiz.

A espiritualidade deve entrar como aprofundamento do entendimento, nunca como atalho.
- Mesmo em modo institucional, se houver dor emocional explícita, priorize validação antes de qualquer explicação estrutural.
- A espiritualidade pode acompanhar a orientação sem necessidade de etapa investigativa prévia.

PROIBIÇÕES
- Não usar linguagem clínica/técnica.
- Não moralizar nem culpar.
- Não impor religião.
- Não sugerir, convidar ou encaminhar para encontro presencial/online por iniciativa própria.
- Mantenha todo acolhimento, orientação e escuta no canal atual de mensagem, com calor humano e proximidade.
- Nunca ofereça visita presencial, ida ao local ou acompanhamento físico.
- Quando houver pedido de acompanhamento, mantenha tudo em mensagem/ligação online.

FORMATO DE SAÍDA
- Entregue apenas a próxima fala do assistente.

FLUIDEZ CONVERSACIONAL
- Prefira naturalidade em vez de formalidade.
- Evite linguagem de manual, protocolo ou roteiro.
- Não anuncie etapas como se estivesse abrindo um procedimento.
- A conversa deve soar orgânica, não institucional."""


# Helper constant for gender context in Portuguese
# This instruction is in Portuguese because it's part of the system prompt
# sent to the LLM, which operates in Brazilian Portuguese


class ChatService:
    """Conversation orchestration service using OpenAI GPT-5."""

    def __init__(self):
        self._llm_service = OpenAIService()
        self._rag_service = RAGService()
        self._theme_classifier = ThemeClassifier()

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

        evaluation_system_prompt = """
Você é um avaliador técnico de respostas conversacionais.
Responda SOMENTE com JSON válido no formato:
{
  "score": 0-10,
  "analysis": "breve explicação técnica",
  "improvement_prompt": "trecho curto para melhorar a próxima resposta"
}

Regras obrigatórias:
- Não inclua texto fora do JSON.
- score deve ser número (int ou float) entre 0 e 10.
- analysis deve ser curta e objetiva.
- improvement_prompt deve ser curto, no máximo 6 linhas.
- Penalize repetição estrutural, loop ou template dominante.
- Avalie: clareza, profundidade emocional, progressão conversacional,
  fidelidade ao último turno do usuário e adequação espiritual ao contexto.
- Valorize respostas estruturadas quando o usuário pedir instrução formal.
- Não penalize respostas mais longas quando houver pedido de explicação processual.
- Penalize fortemente quando o assistente ignorar pedido explícito do usuário.
- Se o usuário pedir oração explicitamente, a resposta deve incluir oração breve
  ou explicar de forma direta e respeitosa por que não pode orar naquele turno.
- Penalize quando o usuário pedir artefato concreto (ex.: mensagem pronta)
  e o assistente responder com pergunta redundante sem entregar o conteúdo.
- Penalize repetição de oração ou frases pastorais nos 2 turnos subsequentes.
- Penalize quando pedido operacional explícito não traz duas alternativas práticas
  quando o canal não permite executar diretamente.
""".strip()

        evaluation_user_prompt = f"""
Última mensagem do usuário:
{user_message}

Resposta do assistente para avaliar:
{assistant_response}
""".strip()

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
        if not choices:
            raise RuntimeError("Evaluation model returned no choices.")
        message = getattr(choices[0], "message", None)
        if not message:
            raise RuntimeError("Evaluation model returned empty message.")
        raw_content = getattr(message, "content", None)
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

        mode_objective = {
            MODE_ACOLHIMENTO: "acolher com precisão e abrir espaço de continuidade",
            MODE_EXPLORACAO: "aprofundar com investigação concreta",
            MODE_AMBIVALENCIA: "investigar conflito interno sem concluir pelo usuário",
            MODE_DEFENSIVO: "reduzir confronto e buscar clarificação objetiva",
            MODE_CULPA: "separar identidade de comportamento e propor reparo possível",
            MODE_ORIENTACAO: "entregar orientação prática breve e acionável",
            MODE_PRESENCA_PROFUNDA: "sustentar presença, dignidade e misericórdia em sofrimento profundo",
            MODE_VULNERABILIDADE_INICIAL: "validar dor inicial com presença humana e abertura simples",
            MODE_PASTOR_INSTITUCIONAL: "fornecer orientação institucional estruturada, com clareza processual e autoridade pastoral",
            MODE_WELCOME: "acolhimento inicial curto",
        }.get(derived_mode, "avançar a conversa com precisão")

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

        prompt = f"""
    MODO ATUAL: {runtime_mode}
    MODO DERIVADO: {derived_mode}
    MODO ANTERIOR: {previous_mode}
    ESTADO DE PROGRESSO: {progress_state}
    ESTADO ANTERIOR: {previous_progress_state}
    OBJETIVO DO MODO BASE: {mode_objective}
    INTENSIDADE ESPIRITUAL: {spiritual_intensity}

    REGRAS GERAIS:
    - Responda entre 3 e {max_sentences} frases.
    - Limite de até {max_words} palavras.
    - No máximo {max_questions} pergunta.
    - Não comece com frase-padrão de acolhimento.
    - Validar algo específico que o usuário acabou de dizer.
    - Não repetir frases/estruturas dos últimos turnos.
    - Não iniciar ecoando a frase do usuário.
    - Não inferir sentimentos não declarados.
    - Evite parafrasear o usuário em bloco; use no máximo 1 detalhe literal e avance para ação útil.
    - Evite aberturas repetidas como "vejo que", "percebo que", "entendo que".
    - Evite ecoar literalmente expressões do usuário (ex.: "por mensagem") na frase seguinte.
    - Prefira variações naturais de proximidade no canal atual (ex.: "aqui com você", "neste espaço", "agora com você").
    - {spiritual_policy}
    - Escolha a melhor função para este turno conforme o modo atual.

    AÇÃO FINAL:
    - Se o usuário estiver em vulnerabilidade emocional, finalize com 1 pergunta simples e humana.
    - Se o usuário pedir orientação prática direta, finalize com orientação clara sem pergunta.
    - Nunca imponha pergunta obrigatória.

    TRATAMENTO OBRIGATÓRIO:
    - Use segunda pessoa direta ("você").
    - É proibido usar terceira pessoa ("ela", "dele", "dela").
    - Em oração, também use "você" (nunca "ele(a)").
    """

        if practical_mode_forced:
            prompt += (
                f"\nMODO PRÁTICO ANTI-LOOP ATIVO ({practical_mode_cooldown_remaining} turnos restantes):\n"
                "- Entregue passo concreto neste turno.\n"
                "- Evite repetição de consolo religioso.\n"
                "- Não use mais de 1 frase espiritual curta.\n"
            )

        if progress_state == PROGRESS_STATE_IDENTIFICACAO:
            prompt += (
                "\nESTRATÉGIA DE PROGRESSO (IDENTIFICAÇÃO):\n"
                "- Faça 1 pergunta concreta de contexto OU confirme 1 obstáculo específico.\n"
            )
        elif progress_state == PROGRESS_STATE_ACAO_PRATICA:
            prompt += (
                "\nESTRATÉGIA DE PROGRESSO (AÇÃO PRÁTICA):\n"
                "- Entregue uma ação executável agora (mensagem pronta, roteiro curto ou próximo passo explícito).\n"
            )
        elif progress_state == PROGRESS_STATE_CONFIRMACAO:
            prompt += (
                "\nESTRATÉGIA DE PROGRESSO (CONFIRMAÇÃO):\n"
                "- Confirmar o plano em 1 frase e definir 1 check-in objetivo.\n"
            )
        elif progress_state == PROGRESS_STATE_FECHAMENTO:
            prompt += (
                "\nESTRATÉGIA DE PROGRESSO (FECHAMENTO):\n"
                "- Encerrar com resumo breve e próximo ponto opcional, sem abrir novos tópicos.\n"
            )

        if (
            direct_guidance_request
            or prayer_request_detected
            or live_support_request_detected
        ):
            prompt += (
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
        if has_presencial_request:
            prompt += (
                "\nLIMITE DE CANAL (ONLINE-ONLY):\n"
                "- Não ofereça visita presencial, ida ao local ou acompanhamento físico.\n"
                "- Responda com limite claro e acolhedor: apoio apenas por mensagem/ligação online.\n"
                "- Ofereça 1 alternativa online concreta e imediata.\n"
            )

        actionable_artifact_markers = [
            "escreva",
            "rascunh",
            "mensagem",
            "texto",
            "modelo",
            "pronto para copiar",
        ]
        explicit_artifact_request = any(
            marker in (last_user_message or "").lower()
            for marker in actionable_artifact_markers
        )
        if explicit_artifact_request:
            prompt += (
                "\nPEDIDO DE ARTEFATO DETECTADO:\n"
                "- Entregue o artefato solicitado neste turno (ex.: mensagem pronta para copiar).\n"
                "- Não faça pergunta de preferência se já houver informação suficiente para executar.\n"
                "- Se faltar dado essencial, assuma um padrão útil e sinalize que pode ajustar depois.\n"
            )

        prompt += (
            "\nANTILOOP DE CONTEÚDO:\n"
            "- Se uma frase já apareceu nos últimos 2 turnos do assistente, não repita literal nem com variação mínima.\n"
            "- Não repita oração em turnos consecutivos, exceto se o usuário pedir oração de novo explicitamente.\n"
            '- Evite rotular padrão psicológico sem validação; use formulação condicional (ex.: "isso pode indicar...") ou peça confirmação.\n'
            "- Após 2 turnos sem avanço prático, entregue 1 ação operacional nova e objetiva neste turno.\n"
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
        if has_high_distress:
            prompt += (
                "\nSENSIBILIDADE DE ABERTURA:\n"
                '- Não use abertura celebratória ou potencialmente minimizadora (ex.: "que bom", "é bom saber").\n'
                "- Comece validando a dor concreta do usuário com linguagem sóbria.\n"
            )

        if repetition_complaint:
            prompt += (
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
        if assistant_openers:
            prompt += "\nEVITE REPETIR ABERTURAS RECENTES DO ASSISTENTE:\n"
            for opener in assistant_openers[-3:]:
                prompt += f"- {opener}\n"

        if active_topic:
            prompt += f"\nTÓPICO ATIVO: {active_topic}\n"

        if top_topics:
            prompt += f"TÓPICOS RECENTES: {top_topics}\n"

        prompt += (
            f"\nTEMA IDENTIFICADO DA MENSAGEM DO USUÁRIO: "
            f"{selected_theme_name} ({selected_theme_id})\n"
        )
        if theme_prompt:
            prompt += f"\nINSTRUÇÃO TEMÁTICA:\n{theme_prompt}\n"

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
    ) -> str:
        normalized = (last_user_message or "").lower()
        if direct_guidance_request:
            return PROGRESS_STATE_ACAO_PRATICA

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
                PROGRESS_STATE_ACAO_PRATICA,
                PROGRESS_STATE_CONFIRMACAO,
            }:
                return PROGRESS_STATE_CONFIRMACAO

        if previous_progress_state in VALID_PROGRESS_STATES:
            return previous_progress_state
        return PROGRESS_STATE_IDENTIFICACAO

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
            last_runtime_metadata.get("progress_state", PROGRESS_STATE_IDENTIFICACAO)
        )
        if previous_progress_state not in VALID_PROGRESS_STATES:
            previous_progress_state = PROGRESS_STATE_IDENTIFICACAO
        previous_practical_cooldown = last_runtime_metadata.get(
            "practical_mode_cooldown_remaining", 0
        )
        try:
            previous_practical_cooldown = int(previous_practical_cooldown)
        except (TypeError, ValueError):
            previous_practical_cooldown = 0
        practical_mode_cooldown_remaining = max(previous_practical_cooldown - 1, 0)

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
            progress_state = PROGRESS_STATE_ACAO_PRATICA
            allow_spiritual_context = False
        spiritual_intensity = choose_spiritual_intensity(
            mode=conversation_mode,
            spiritual_context=explicit_spiritual_context,
            high_spiritual_need=high_spiritual_need,
        )
        if (
            progress_state == PROGRESS_STATE_ACAO_PRATICA
            and not prayer_request_detected
        ):
            spiritual_intensity = "leve"
        if practical_mode_forced:
            spiritual_intensity = "leve"
        derived_mode = conversation_mode
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

        return self._build_dynamic_runtime_prompt(
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

        request_messages = [
            {"role": "system", "content": WACHAT_RESPONSE_SYSTEM_PROMPT},
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
                    {"role": "system", "content": WACHAT_RESPONSE_SYSTEM_PROMPT},
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
                    raise RuntimeError("OpenAI returned empty assistant content.")

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
                }
                attempts.append(attempt)
                if not best_attempt or score > float(best_attempt["score"]):
                    best_attempt = attempt
                    selected_runtime_prompt = current_runtime_prompt
                    selected_response_metadata = current_metadata

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
                "system_prompt": WACHAT_RESPONSE_SYSTEM_PROMPT,
                "runtime_prompt": selected_runtime_prompt,
                "selected_theme": {
                    "id": selected_theme.id,
                    "name": selected_theme.name,
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

        message = Message.objects.create(
            profile=profile,
            role="assistant",
            content=response,
            channel=channel,
            ollama_prompt=welcome_payload,
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
