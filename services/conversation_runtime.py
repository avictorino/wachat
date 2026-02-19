import math
import os
import re
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
MODE_PASTOR_INSTITUCIONAL = "pastor_institucional"
MODE_VULNERABILIDADE_INICIAL = "vulnerabilidade_inicial"

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
    vec_a = _embedding_for_text(text_a)
    vec_b = _embedding_for_text(text_b)
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


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
    if is_first_message and (
        signals.get("deep_suffering")
        or signals.get("repetitive_guilt")
        or signals.get("family_conflict_impotence")
        or signals.get("explicit_despair")
    ):
        return MODE_VULNERABILIDADE_INICIAL
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
