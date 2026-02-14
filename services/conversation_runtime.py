import math
import os
import random
import re
from difflib import SequenceMatcher
from typing import Iterable, List

import requests

BLOCKED_PATTERNS = [
    "isso pode ser muito difícil",
    "deus está ao seu lado",
    "procure apoio",
    "você não está sozinho",
    "isso não é um caminho fácil",
    "eu sinto muito pelo peso",
    "eu recebo o que você trouxe com respeito",
    "eu continuo ao seu lado com presença e cuidado",
    "obrigado por confiar isso aqui",
    "você não precisa carregar esse peso sozinho neste instante",
]

MODE_WELCOME = "WELCOME"
MODE_ACOLHIMENTO = "ACOLHIMENTO"
MODE_EXPLORACAO = "EXPLORACAO"
MODE_AMBIVALENCIA = "AMBIVALENCIA"
MODE_DEFENSIVO = "DEFENSIVO"
MODE_CULPA = "CULPA"
MODE_ORIENTACAO = "ORIENTACAO"
MODE_PRESENCA_PROFUNDA = "PRESENCA_PROFUNDA"

MAX_SENTENCES = 4
MAX_WORDS = 180
MAX_QUESTIONS = 1
SEMANTIC_LOOP_THRESHOLD = 0.85

_GREETING_PREFIXES = (
    "oi",
    "olá",
    "ola",
    "bom dia",
    "boa tarde",
    "boa noite",
    "paz",
)

_INFERENCE_TRIGGERS = [
    "você está lutando",
    "você está se sentindo",
    "você se sente",
    "você tá se sentindo",
]

_FEELING_TERMS = [
    "ansioso",
    "ansiosa",
    "culpado",
    "culpada",
    "triste",
    "sozinho",
    "sozinha",
    "perdido",
    "perdida",
    "com medo",
    "frustrado",
    "frustrada",
    "exausto",
    "exausta",
    "angustiado",
    "angustiada",
    "desesperado",
    "desesperada",
    "deprimido",
    "deprimida",
]

_AMBIVALENCE_MARKERS = [
    "ao mesmo tempo",
    "mas também",
    "não sei",
    "talvez",
    "por outro lado",
    "quero mas",
    "não quero mas",
    "eu sei que é errado mas",
    "quero parar, mas",
    "quero parar mas",
]

_DIRECT_GUIDANCE_MARKERS = [
    "como começo",
    "por onde começo",
    "o que faço agora",
    "o que eu faço agora",
    "como faço",
    "me ajuda a começar",
    "qual o primeiro passo",
]

_REPETITION_COMPLAINT_MARKERS = [
    "já falei",
    "ja falei",
    "já disse",
    "ja disse",
    "você já perguntou",
    "voce ja perguntou",
    "vai ficar repetindo",
    "parar de repetir",
    "mesma pergunta",
    "de novo essa pergunta",
    "não repete",
    "nao repete",
]

_GENERIC_EMPATHY_PATTERNS = [
    "eu sinto muito",
    "imagino como isso é difícil",
    "deve ser muito difícil",
    "isso não é fácil",
    "eu entendo sua dor",
]

_EXPLICIT_SPIRITUAL_TERMS = [
    "deus",
    "oração",
    "orar",
    "jesus",
    "igreja",
    "salmo",
    "bíblia",
    "biblia",
]

_SPIRITUAL_BASELINE_TERMS = [
    "esperança",
    "esperanca",
    "graça",
    "graca",
    "paz",
    "misericórdia",
    "misericordia",
    "propósito",
    "proposito",
    "fé",
]

_HIGH_SPIRITUAL_NEED_MARKERS = [
    "culpa",
    "culpado",
    "culpada",
    "recaída",
    "recaida",
    "não consigo me perdoar",
    "sem saída",
    "desesper",
    "sozinho",
    "sozinha",
    "não tenho ninguém",
    "não tenho ninguem",
    "não tenho com quem contar",
]

_DEFENSIVE_MARKERS = [
    "não é bem assim",
    "você não entende",
    "vocês não entendem",
    "não tenho problema",
    "não é minha culpa",
    "todo mundo faz",
    "não exagera",
]

_GUILT_MARKERS = [
    "culpa",
    "culpado",
    "culpada",
    "erro",
    "vergonha",
    "me odeio",
    "sou um fracasso",
    "não presto",
]

_DEEP_SUFFERING_MARKERS = [
    "angústia",
    "angustia",
    "no peito",
    "aperto no peito",
    "pesado demais",
    "insuportável",
    "insuportavel",
    "não aguento",
    "nao aguento",
    "desespero",
    "desesperado",
    "desesperada",
    "morte",
    "medo da morte",
    "sem saída",
    "sem saida",
]

_REPETITIVE_GUILT_MARKERS = [
    "de novo",
    "sempre",
    "mais uma vez",
    "outra vez",
    "de novo eu",
    "sempre eu",
]

_FAMILY_CONFLICT_MARKERS = [
    "família",
    "familia",
    "mãe",
    "mae",
    "pai",
    "esposa",
    "marido",
    "filho",
    "filha",
    "relacionamento",
    "casamento",
    "brig",
    "discuss",
    "conflito",
]

_IMPOTENCE_MARKERS = [
    "não consigo",
    "nao consigo",
    "não dá",
    "nao da",
    "não adianta",
    "nao adianta",
    "sem saída",
    "sem saida",
    "impotente",
]

_SPIRITUAL_IMPOSITION_PATTERNS = [
    "deus quer que",
    "deus exige",
    "deus manda",
    "se você tiver fé tudo",
]

_SUPPORT_REFERRAL_MARKERS = [
    "caps",
    "caps ad",
    "aa",
    "alcoólicos anônimos",
    "grupo",
    "pastor",
    "igreja",
    "pessoa de confiança",
    "familiar",
    "amigo",
    "profissional",
    "médico",
    "terapeuta",
    "psicólogo",
]

_SELF_GUIDED_HELP_MARKERS = [
    "oração curta",
    "ore agora",
    "oração agora",
    "salmo",
    "respire",
    "escreva uma oração",
    "entregue a deus",
    "peça perdão a deus",
    "graça",
]

_FALLBACK_QUESTIONS = [
    "O que exatamente aconteceu ontem?",
    "Qual foi a última decisão que você tomou sobre isso?",
    "Qual foi o momento mais difícil nas últimas 24 horas?",
]

_PRACTICAL_ACTION_MARKERS = [
    "agora",
    "hoje",
    "primeiro passo",
    "faça",
    "anote",
    "escreva",
    "defina",
    "responda",
    "escolha",
    "envie",
    "pare",
]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _extract_first_sentence(text: str) -> str:
    normalized = _normalize(text)
    if not normalized:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", normalized, maxsplit=1)
    return parts[0].strip()


def _extract_greeting(text: str) -> str:
    first = _extract_first_sentence(text)
    for prefix in _GREETING_PREFIXES:
        if first.startswith(prefix):
            return first
    return ""


def _embedding_base_url() -> str:
    return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def _embedding_model() -> str:
    return os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def _embedding_for_text(text: str) -> List[float]:
    response = requests.post(
        f"{_embedding_base_url()}/api/embeddings",
        json={"model": _embedding_model(), "prompt": text},
        timeout=12,
    )
    response.raise_for_status()
    embedding = response.json().get("embedding")
    if not isinstance(embedding, list):
        raise RuntimeError("Invalid embedding response")
    return embedding


def semantic_similarity(text_a: str, text_b: str) -> float:
    if not text_a or not text_b:
        return 0.0
    if _normalize(text_a) == _normalize(text_b):
        return 1.0
    try:
        vec_a = _embedding_for_text(text_a)
        vec_b = _embedding_for_text(text_b)
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if not norm_a or not norm_b:
            return 0.0
        return dot / (norm_a * norm_b)
    except Exception:
        return SequenceMatcher(None, _normalize(text_a), _normalize(text_b)).ratio()


def is_semantic_loop(last_message: str, new_message: str) -> bool:
    if not last_message or not new_message:
        return False
    last_greeting = _extract_greeting(last_message)
    new_greeting = _extract_greeting(new_message)
    if last_greeting and new_greeting and last_greeting == new_greeting:
        return True
    if _extract_first_sentence(last_message) == _extract_first_sentence(new_message):
        return True
    return semantic_similarity(last_message, new_message) > SEMANTIC_LOOP_THRESHOLD


def contains_unverified_inference(user_message: str, assistant_message: str) -> bool:
    user_norm = _normalize(user_message)
    assistant_norm = _normalize(assistant_message)
    if not assistant_norm:
        return False
    for trigger in _INFERENCE_TRIGGERS:
        if trigger in assistant_norm and trigger not in user_norm:
            return True
    for feeling in _FEELING_TERMS:
        if feeling in assistant_norm and feeling not in user_norm:
            if "você " in assistant_norm or "voce " in assistant_norm:
                return True
    return False


def detect_direct_guidance_request(user_message: str) -> bool:
    msg = _normalize(user_message)
    return any(marker in msg for marker in _DIRECT_GUIDANCE_MARKERS)


def detect_repetition_complaint(user_message: str) -> bool:
    msg = _normalize(user_message)
    return any(marker in msg for marker in _REPETITION_COMPLAINT_MARKERS)


def has_spiritual_context(user_message: str) -> bool:
    msg = _normalize(user_message)
    return any(term in msg for term in _EXPLICIT_SPIRITUAL_TERMS) or "fé" in msg


def has_high_spiritual_need(user_message: str) -> bool:
    msg = _normalize(user_message)
    return any(marker in msg for marker in _HIGH_SPIRITUAL_NEED_MARKERS)


def contains_unsolicited_spiritualization(
    user_message: str, assistant_message: str
) -> bool:
    user_norm = _normalize(user_message)
    assistant_norm = _normalize(assistant_message)
    if not assistant_norm:
        return False
    assistant_has_spiritual = any(
        term in assistant_norm for term in _EXPLICIT_SPIRITUAL_TERMS
    )
    user_has_spiritual = any(term in user_norm for term in _EXPLICIT_SPIRITUAL_TERMS)
    return assistant_has_spiritual and not user_has_spiritual


def contains_spiritual_imposition(assistant_message: str) -> bool:
    assistant_norm = _normalize(assistant_message)
    return any(pattern in assistant_norm for pattern in _SPIRITUAL_IMPOSITION_PATTERNS)


def contains_generic_empathy_without_grounding(
    user_message: str, assistant_message: str
) -> bool:
    user_norm = _normalize(user_message)
    assistant_norm = _normalize(assistant_message)
    if not assistant_norm:
        return False

    has_generic_empathy = any(
        pattern in assistant_norm for pattern in _GENERIC_EMPATHY_PATTERNS
    )
    if not has_generic_empathy:
        return False

    user_tokens = set(
        token for token in re.findall(r"[a-zA-ZÀ-ÿ]{4,}", user_norm) if len(token) >= 4
    )
    assistant_tokens = set(
        token
        for token in re.findall(r"[a-zA-ZÀ-ÿ]{4,}", assistant_norm)
        if len(token) >= 4
    )
    overlap = user_tokens.intersection(assistant_tokens)
    return len(overlap) < 2


def contains_repeated_blocked_pattern(
    assistant_message: str, recent_assistant_messages: Iterable[str]
) -> bool:
    candidate = _normalize(assistant_message)
    history = [_normalize(msg) for msg in list(recent_assistant_messages)[-3:]]
    for pattern in BLOCKED_PATTERNS:
        if pattern in candidate and any(pattern in old for old in history):
            return True
    return False


def has_repeated_opening_structure(
    assistant_message: str, recent_assistant_messages: Iterable[str]
) -> bool:
    candidate_first = _extract_first_sentence(assistant_message)
    if not candidate_first:
        return False

    candidate_tokens = candidate_first.split()[:8]
    if len(candidate_tokens) < 4:
        return False

    for old in list(recent_assistant_messages)[-3:]:
        old_first = _extract_first_sentence(old)
        if not old_first:
            continue

        old_tokens = old_first.split()[:8]
        if candidate_tokens[:4] == old_tokens[:4]:
            return True

        if semantic_similarity(candidate_first, old_first) > 0.84:
            return True
    return False


def enforce_hard_limits(message: str, max_sentences: int = MAX_SENTENCES) -> str:
    chunks = [part.strip() for part in re.findall(r"[^.!?]+[.!?]?", message or "")]
    chunks = [part for part in chunks if part]

    kept = []
    question_count = 0
    for chunk in chunks:
        if "?" in chunk:
            if question_count >= MAX_QUESTIONS:
                chunk = chunk.replace("?", ".")
            else:
                question_count += 1
        kept.append(chunk)
        if len(kept) >= max_sentences:
            break

    limited = " ".join(kept).strip()
    words = limited.split()
    if len(words) > MAX_WORDS:
        limited = " ".join(words[:MAX_WORDS]).strip()
        if limited and limited[-1] not in ".!?":
            limited += "."
    return re.sub(r"\s+", " ", limited).strip()


def strip_opening_name_if_recently_used(
    message: str, name: str, recent_assistant_messages: Iterable[str]
) -> str:
    if not name:
        return message
    name_norm = _normalize(name)
    recent = [_normalize(m) for m in list(recent_assistant_messages)[-2:]]
    if not any(name_norm in old for old in recent):
        return message

    pattern = re.compile(
        rf"^((oi|olá|ola|bom dia|boa tarde|boa noite)\s+)?{re.escape(name)}[,\-:\s]+",
        re.IGNORECASE,
    )
    return pattern.sub("", message.strip(), count=1).strip() or message.strip()


def detect_ambivalence(user_message: str) -> bool:
    msg = _normalize(user_message)
    return any(marker in msg for marker in _AMBIVALENCE_MARKERS)


def detect_defensiveness(user_message: str) -> bool:
    msg = _normalize(user_message)
    return any(marker in msg for marker in _DEFENSIVE_MARKERS)


def detect_guilt(user_message: str) -> bool:
    msg = _normalize(user_message)
    return any(marker in msg for marker in _GUILT_MARKERS)


def detect_deep_suffering(user_message: str) -> bool:
    msg = _normalize(user_message)
    return any(marker in msg for marker in _DEEP_SUFFERING_MARKERS)


def detect_repetitive_guilt(user_message: str) -> bool:
    msg = _normalize(user_message)
    has_guilt = any(marker in msg for marker in _GUILT_MARKERS)
    if not has_guilt:
        return False
    return any(marker in msg for marker in _REPETITIVE_GUILT_MARKERS)


def detect_family_conflict_impotence(user_message: str) -> bool:
    msg = _normalize(user_message)
    has_family_conflict = any(marker in msg for marker in _FAMILY_CONFLICT_MARKERS)
    has_impotence = any(marker in msg for marker in _IMPOTENCE_MARKERS)
    return has_family_conflict and has_impotence


def detect_explicit_despair(user_message: str) -> bool:
    msg = _normalize(user_message)
    return any(
        marker in msg
        for marker in [
            "desespero",
            "desesperado",
            "desesperada",
            "não aguento mais",
            "nao aguento mais",
            "acabou pra mim",
        ]
    )


def detect_user_signals(user_message: str) -> dict:
    return {
        "guidance_request": detect_direct_guidance_request(user_message),
        "repetition_complaint": detect_repetition_complaint(user_message),
        "ambivalence": detect_ambivalence(user_message),
        "defensive": detect_defensiveness(user_message),
        "guilt": detect_guilt(user_message),
        "deep_suffering": detect_deep_suffering(user_message),
        "repetitive_guilt": detect_repetitive_guilt(user_message),
        "family_conflict_impotence": detect_family_conflict_impotence(user_message),
        "explicit_despair": detect_explicit_despair(user_message),
        "spiritual_context": has_spiritual_context(user_message),
        "high_spiritual_need": has_high_spiritual_need(user_message),
    }


def choose_conversation_mode(
    *,
    previous_mode: str,
    is_first_message: bool,
    loop_detected: bool,
    has_new_info: bool,
    repeated_user_pattern: bool,
    signals: dict,
) -> str:
    if (
        signals.get("deep_suffering")
        or signals.get("repetitive_guilt")
        or signals.get("family_conflict_impotence")
        or signals.get("explicit_despair")
    ):
        return MODE_PRESENCA_PROFUNDA
    if signals.get("guidance_request"):
        return MODE_ORIENTACAO
    if signals.get("repetition_complaint"):
        return MODE_ORIENTACAO
    if is_first_message or previous_mode == MODE_WELCOME:
        return MODE_ACOLHIMENTO
    if signals.get("ambivalence"):
        return MODE_AMBIVALENCIA
    if signals.get("defensive"):
        return MODE_DEFENSIVO
    if signals.get("guilt"):
        return MODE_CULPA
    if loop_detected:
        return MODE_ORIENTACAO
    if has_new_info:
        return MODE_EXPLORACAO
    if repeated_user_pattern and previous_mode == MODE_AMBIVALENCIA:
        return MODE_ORIENTACAO
    return previous_mode


def choose_spiritual_intensity(
    *,
    mode: str,
    spiritual_context: bool,
    high_spiritual_need: bool,
) -> str:
    if spiritual_context:
        return "alta"
    if mode == MODE_PRESENCA_PROFUNDA:
        return "alta"
    if high_spiritual_need:
        return "media"
    if mode in {MODE_CULPA, MODE_AMBIVALENCIA} or (
        mode == MODE_ORIENTACAO and high_spiritual_need
    ):
        return "media"
    return "leve"


def has_repeated_user_pattern(user_messages: Iterable[str]) -> bool:
    recent = [msg for msg in list(user_messages)[-2:] if msg]
    if len(recent) < 2:
        return False
    return semantic_similarity(recent[0], recent[1]) > SEMANTIC_LOOP_THRESHOLD


def has_new_information(user_messages: Iterable[str]) -> bool:
    recent = [msg for msg in list(user_messages)[-2:] if msg]
    if len(recent) < 2:
        return True
    return semantic_similarity(recent[0], recent[1]) < 0.8


def should_force_progress_fallback(
    user_messages: Iterable[str], assistant_messages: Iterable[str]
) -> bool:
    recent_users = [msg for msg in list(user_messages)[-2:] if msg]
    recent_assistant = [msg for msg in list(assistant_messages)[-2:] if msg]
    if len(recent_users) < 2 or len(recent_assistant) < 2:
        return False
    user_similarity = semantic_similarity(recent_users[0], recent_users[1])
    assistant_similarity = semantic_similarity(recent_assistant[0], recent_assistant[1])
    return user_similarity > 0.85 and assistant_similarity > 0.85


def make_progress_fallback_question() -> str:
    return random.choice(_FALLBACK_QUESTIONS)


def has_practical_action_step(assistant_message: str) -> bool:
    msg = _normalize(assistant_message)
    return any(marker in msg for marker in _PRACTICAL_ACTION_MARKERS)


def has_human_support_suggestion(assistant_message: str) -> bool:
    msg = _normalize(assistant_message)
    return any(marker in msg for marker in _SUPPORT_REFERRAL_MARKERS)


def has_self_guided_help(assistant_message: str) -> bool:
    msg = _normalize(assistant_message)
    return any(marker in msg for marker in _SELF_GUIDED_HELP_MARKERS)


def has_spiritual_baseline_signal(assistant_message: str) -> bool:
    msg = _normalize(assistant_message)
    return any(term in msg for term in _SPIRITUAL_BASELINE_TERMS)


def contains_spiritual_template_repetition(
    assistant_message: str, recent_assistant_messages: Iterable[str]
) -> bool:
    if not has_spiritual_baseline_signal(assistant_message):
        return False
    for old in list(recent_assistant_messages)[-5:]:
        if not has_spiritual_baseline_signal(old):
            continue
        if semantic_similarity(old, assistant_message) > 0.87:
            return True
    return False


def starts_with_user_echo(user_message: str, assistant_message: str) -> bool:
    user_first = _extract_first_sentence(user_message)
    assistant_first = _extract_first_sentence(assistant_message)
    if not user_first or not assistant_first:
        return False

    user_tokens = user_first.split()
    assistant_tokens = assistant_first.split()
    if len(user_tokens) < 4 or len(assistant_tokens) < 4:
        return False

    overlap = 0
    for a, b in zip(user_tokens[:8], assistant_tokens[:8]):
        if a == b:
            overlap += 1
        else:
            break
    if overlap >= 4:
        return True

    return semantic_similarity(user_first, assistant_first) > 0.9
