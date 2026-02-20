"""Simulation service focused on generating only the next user turn."""

import hashlib
from enum import Enum
from typing import Iterable, List, Optional, Union

from core.models import Message, Profile, Theme
from services.openai_service import OpenAIService
from services.theme_classifier import ThemeClassifier

SIMULATION_MAX_COMPLETION_TOKENS = 1200


class SimulatedUserProfile(Enum):
    AMBIVALENTE = "ambivalente"
    DEFENSIVO = "defensivo"
    CULPA_FORTE = "culpa_forte"
    DESESPERANCA = "desesperanca"
    RACIONAL = "racional"
    PEDIDO_AJUDA = "pedido_ajuda_implicito"
    FECHAMENTO = "fechamento_emocional"


PROFILE_INSTRUCTIONS = {
    SimulatedUserProfile.AMBIVALENTE: "Quer mudar, mas sente que não consegue manter constância.",
    SimulatedUserProfile.DEFENSIVO: "Quer ser ouvido sem julgamento e sem bronca.",
    SimulatedUserProfile.CULPA_FORTE: "Quer aliviar a culpa e encontrar perdão.",
    SimulatedUserProfile.DESESPERANCA: "Quer voltar a ter esperança e força para continuar.",
    SimulatedUserProfile.RACIONAL: "Quer clareza para tomar uma decisão prática.",
    SimulatedUserProfile.PEDIDO_AJUDA: "Quer direção prática para o próximo passo.",
    SimulatedUserProfile.FECHAMENTO: "Quer encerrar o assunto por cansaço emocional.",
}

PROFILE_LABELS = {
    SimulatedUserProfile.AMBIVALENTE: "quer mudar, mas não consegue sustentar a mudança",
    SimulatedUserProfile.DEFENSIVO: "quer ser ouvida sem julgamento",
    SimulatedUserProfile.CULPA_FORTE: "quer aliviar a culpa que está carregando",
    SimulatedUserProfile.DESESPERANCA: "quer voltar a ter esperança",
    SimulatedUserProfile.RACIONAL: "quer clareza para decidir o que fazer",
    SimulatedUserProfile.PEDIDO_AJUDA: "quer orientação prática para o próximo passo",
    SimulatedUserProfile.FECHAMENTO: "quer fugir do assunto por desgaste emocional",
}

PREDEFINED_SCENARIOS = {
    "sobrecarregado": "está sobrecarregada e no limite",
    "ansioso_com_medo": "se sente ansiosa e com medo",
    "culpado_envergonhado": "está culpada e envergonhada",
    "irritado_com_raiva": "está irritada e com raiva",
    "sozinho_sem_apoio": "se sente sozinha e sem apoio",
    "confuso_perdido": "está confusa e perdida",
    "desanimado_sem_forca": "está desanimada e sem força",
    "sem_esperanca": "se sente sem esperança",
}

ORIENTACAO_MODE = "ORIENTACAO"

RELATED_TOPIC_ANGLES = [
    "impacto na família e na convivência da casa",
    "efeito no trabalho/estudo e na rotina da semana",
    "como isso afeta sono, energia e foco",
    "um conflito parecido que já aconteceu antes",
    "um caso parecido de alguém próximo que gerou medo de repetir o padrão",
    "o que piora a situação quando tenta resolver rápido demais",
]


def _theme_options_from_db() -> dict:
    options = {}
    for theme in Theme.objects.all().order_by("name"):
        options[theme.id] = theme.name
    return options


def _parse_optional_theme_id(theme: Union[int, str, None]) -> Optional[int]:
    if theme is None:
        return None
    raw = str(theme).strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid simulation theme id: '{raw}'.") from exc


def _to_recent_history(conversation: Iterable, limit: int = 5) -> List[dict]:
    normalized = []
    for item in list(conversation)[-limit:]:
        if isinstance(item, dict):
            role = item.get("role", "user")
            content = (item.get("content") or "").strip()
            bot_mode = item.get("bot_mode")
        else:
            role = getattr(item, "role", "user")
            content = (getattr(item, "content", "") or "").strip()
            bot_mode = getattr(item, "bot_mode", None)
        if content:
            normalized.append(
                {"role": role, "content": content, "bot_mode": bot_mode or ""}
            )
    return normalized


def _stable_index(seed_text: str, modulo: int) -> int:
    if modulo <= 0:
        raise ValueError("modulo must be greater than zero.")
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def _build_persona_signature(
    *,
    profile: SimulatedUserProfile,
    scenario: str,
    inferred_gender: Optional[str],
    full_conversation: List,
) -> dict:
    first_user_message = ""
    for item in full_conversation:
        role = getattr(item, "role", None)
        content = (getattr(item, "content", "") or "").strip()
        if role == "user" and content:
            first_user_message = content
            break
    seed = f"{profile.value}|{scenario}|{inferred_gender or 'unknown'}|{first_user_message}"

    tone_options = ["informal_coloquial", "acolhedor_direto", "intimo_reflexivo"]
    rhythm_options = ["frases_curta_media", "frases_medias", "frases_quebradas"]
    vocabulary_options = [
        "simples_comum",
        "simples_com_girias_leves",
        "simples_com_mais_detalhe",
    ]
    tic_options = [
        "pra ser sincera",
        "na real",
        "sendo bem honesta",
        "tipo assim",
        "eu fico pensando",
    ]

    return {
        "tone": tone_options[_stable_index(seed + "|tone", len(tone_options))],
        "rhythm": rhythm_options[_stable_index(seed + "|rhythm", len(rhythm_options))],
        "vocabulary": vocabulary_options[
            _stable_index(seed + "|vocabulary", len(vocabulary_options))
        ],
        "writing_tic": tic_options[_stable_index(seed + "|tic", len(tic_options))],
    }


def _build_conversation_state(recent_history: List[dict]) -> dict:
    pending_question = ""
    last_assistant_action = ""
    last_user_message = ""
    last_user_messages = []

    for message in recent_history:
        role = message.get("role")
        content = (message.get("content") or "").strip()
        if role == "assistant":
            if "?" in content:
                pending_question = content
            if any(
                marker in content.lower()
                for marker in ("posso", "vamos", "topa", "escolhe", "tenta")
            ):
                last_assistant_action = content
        if role == "user":
            last_user_message = content
            last_user_messages.append(content)

    repeated_hint = ""
    if len(last_user_messages) >= 2:
        previous = last_user_messages[-2].lower()
        current = last_user_messages[-1].lower()
        for marker in (
            "não consigo",
            "to cansada",
            "estou cansada",
            "me sinto sozinha",
            "sem força",
        ):
            if marker in previous and marker in current:
                repeated_hint = marker
                break

    return {
        "pending_question": pending_question,
        "last_assistant_action": last_assistant_action,
        "last_user_message": last_user_message,
        "repeated_hint": repeated_hint,
    }


def _detect_post_orientacao_phase(recent_history: List[dict]) -> bool:
    assistant_modes = [
        (message.get("bot_mode") or "").strip().upper()
        for message in recent_history
        if message.get("role") == "assistant"
    ]
    if len(assistant_modes) >= 2 and assistant_modes[-1] == ORIENTACAO_MODE:
        if assistant_modes[-2] == ORIENTACAO_MODE:
            return True
    assistant_texts = [
        (message.get("content") or "").lower()
        for message in recent_history
        if message.get("role") == "assistant"
    ]
    practical_markers = ("passo", "plano", "ação", "acao", "roteiro", "mensagem pronta")
    practical_hits = 0
    for text in assistant_texts[-3:]:
        if any(marker in text for marker in practical_markers):
            practical_hits += 1
    return practical_hits >= 2


def _build_related_topic_hint(
    *,
    problem_label: str,
    selected_profile: SimulatedUserProfile,
    next_user_turn: int,
) -> str:
    if not problem_label or problem_label == "não ficou claro":
        seed = f"{selected_profile.value}|{next_user_turn}"
    else:
        seed = f"{problem_label}|{selected_profile.value}|{next_user_turn}"
    return RELATED_TOPIC_ANGLES[_stable_index(seed, len(RELATED_TOPIC_ANGLES))]


def _build_turn_style_plan(next_user_turn: int) -> dict:
    shape_cycle = [
        "media_com_detalhe",
        "curta_direta",
        "media_com_pergunta_no_final",
        "curta_com_desabafo",
    ]
    shape = shape_cycle[(next_user_turn - 1) % len(shape_cycle)]

    return {
        "shape": shape,
        "allow_pause_marker": next_user_turn % 2 == 0,
        "allow_whatsapp_marker": next_user_turn % 3 == 0,
        "allow_self_correction": next_user_turn % 4 == 0,
        "allow_minor_typo": next_user_turn % 5 == 0,
        "word_limit": 52 if next_user_turn % 2 == 0 else 68,
    }


def _build_turn_intent(
    *,
    profile: SimulatedUserProfile,
    last_assistant_message: str,
    feeling_label: str,
    force_context_expansion: bool,
    post_orientacao_phase: bool,
    related_topic_hint: str,
) -> dict:
    normalized_last_assistant = (last_assistant_message or "").lower()
    if post_orientacao_phase:
        objective = (
            "trazer um assunto relacionado e contar um caso parecido, "
            "conectando com o problema principal sem reiniciar a conversa"
        )
    elif "?" in (last_assistant_message or ""):
        objective = "responder a pergunta do bot com detalhe concreto do que aconteceu"
    elif any(
        marker in normalized_last_assistant
        for marker in ("posso", "vou te ajudar", "passo", "plano", "escolhe")
    ):
        objective = (
            "aceitar, ajustar ou recusar de forma objetiva um passo prático proposto"
        )
    elif force_context_expansion:
        objective = (
            "trazer mais contexto real do problema antes de assumir um novo compromisso"
        )
    else:
        objective = PROFILE_LABELS.get(
            profile, "pedir ajuda prática sem repetir exatamente o turno anterior"
        )

    return {
        "objective": objective,
        "emotion": feeling_label,
        "post_orientacao_phase": post_orientacao_phase,
        "related_topic_hint": related_topic_hint,
    }


def _parse_profile(profile: Union[SimulatedUserProfile, str]) -> SimulatedUserProfile:
    if isinstance(profile, SimulatedUserProfile):
        return profile
    for option in SimulatedUserProfile:
        if option.value == profile:
            return option
    return SimulatedUserProfile.AMBIVALENTE


def _build_simple_simulation_prompt(
    *,
    feeling_context: str,
    desire_context: str,
    problem_context: str,
    history_text: str,
    last_assistant_message: str,
    grammatical_gender: str,
    should_avoid_explicit_spiritual_request: bool,
    force_context_expansion: bool,
    persona_signature: dict,
    conversation_state: dict,
    turn_style_plan: dict,
    turn_intent: dict,
) -> str:
    spiritual_request_rule = (
        "- Neste turno, não peça oração, leitura bíblica ou prática espiritual explícita; foque em desabafo e contexto concreto.\n"
        if should_avoid_explicit_spiritual_request
        else "- Se fizer sentido pelo contexto, pode pedir oração ou apoio espiritual de forma natural.\n"
    )
    context_expansion_rule = (
        "- Neste turno, continue explicando melhor seu problema e obstáculos; ainda não assuma compromisso de hábito específico.\n"
        "- Se o bot perguntar por uma mudança prática, responda com contexto adicional (gatilho/obstáculo) antes de fechar tarefa.\n"
        if force_context_expansion
        else "- Se já houver contexto suficiente, pode avançar para um compromisso prático simples.\n"
    )
    related_topic_flow_rule = (
        "- A conversa entrou em fase pós-orientação: avance para um assunto relacionado, sem abandonar o tema central.\n"
        "- Conte um caso parecido curto (seu ou de alguém próximo) e conecte esse caso ao que você está vivendo agora.\n"
        "- Evite parecer troca brusca de assunto; faça ponte explícita com a última fala do bot.\n"
        "- Assunto relacionado sugerido para este turno: "
        f"{turn_intent['related_topic_hint']}.\n"
        if turn_intent.get("post_orientacao_phase")
        else "- Mantenha foco principal, mas pode trazer um detalhe relacionado se ajudar a conversa a evoluir.\n"
    )

    return f"""
        Gere UMA única mensagem simulando a fala de uma pessoa real, como em conversa de WhatsApp.

        Contexto obrigatório desta fala:
        - Como me sinto: {feeling_context}
        - O que eu quero: {desire_context}
        - Meu problema: {problem_context}
        - Concordância gramatical obrigatória: {grammatical_gender}
        - Objetivo do turno atual: {turn_intent["objective"]}
        - Emoção dominante neste turno: {turn_intent["emotion"]}

        Assinatura estável da personagem (não mudar entre turnos):
        - Tom: {persona_signature["tone"]}
        - Ritmo: {persona_signature["rhythm"]}
        - Vocabulário: {persona_signature["vocabulary"]}
        - Tique de linguagem recorrente eventual: {persona_signature["writing_tic"]}

        Histórico recente da conversa (use apenas como referência):
        {history_text if history_text else "Sem histórico relevante."}

        Última fala do bot (você deve responder a ela):
        {last_assistant_message if last_assistant_message else "Sem fala recente do bot."}

        Estado de continuidade que deve ser respeitado:
        - Pergunta pendente do bot: {conversation_state["pending_question"] or "nenhuma"}
        - Última ação sugerida pelo bot: {conversation_state["last_assistant_action"] or "nenhuma"}
        - Última fala do usuário: {conversation_state["last_user_message"] or "nenhuma"}
        - Expressão possivelmente repetida e a evitar: {conversation_state["repeated_hint"] or "nenhuma"}

        Plano de variação para este turno:
        - Formato desejado da mensagem: {turn_style_plan["shape"]}
        - Limite de palavras deste turno: {turn_style_plan["word_limit"]}
        - Pode usar pausa curta ("...", "hmm"): {"sim" if turn_style_plan["allow_pause_marker"] else "não"}
        - Pode usar marcador de WhatsApp ("kk", "rs", "ah"): {"sim" if turn_style_plan["allow_whatsapp_marker"] else "não"}
        - Pode usar autocorreção pequena ("*corrigindo"): {"sim" if turn_style_plan["allow_self_correction"] else "não"}
        - Pode ter micro erro de digitação não crítico: {"sim" if turn_style_plan["allow_minor_typo"] else "não"}

        Regras obrigatórias:
        - Responda com uma única mensagem natural, com 2 a 4 frases.
        - Limite total: no máximo {turn_style_plan["word_limit"]} palavras.
        - Não use listas.
        - Não copie literalmente os textos de contexto acima.
        - Não explique a situação de forma abstrata.
        - Não use linguagem técnica, psicológica ou conceitual.
        - Evite clichês e frases genéricas.
        - Mostre a situação por sentimentos, pensamentos ou comportamentos concretos.
        - Use tom de confissão para alguém de confiança da igreja, sem mencionar explicitamente "pastor".
        {spiritual_request_rule}
        {context_expansion_rule}
        {related_topic_flow_rule}
        - A resposta deve continuar a conversa de forma coerente com a ÚLTIMA fala do bot.
        - Se o bot fizer pergunta, responda pelo menos uma parte da pergunta com detalhe concreto.
        - Se o bot pedir escolha direta (ex.: "formal ou pessoal"), responda objetivamente essa escolha na primeira frase.
        - Se o bot oferecer ação prática ("posso escrever agora"), avance com execução imediata em vez de repetir desabafo.
        - Se o bot sugerir passos práticos, reaja a pelo menos um passo (aceitando, recusando, pedindo ajuda para executar).
        - Não reinicie o assunto do zero; avance a conversa com base no que o bot acabou de dizer.
        - Evite repetir as mesmas expressões dos dois últimos turnos do usuário (ex.: "estou quebrado", "choro todas as noites").
        - Cada nova fala deve adicionar 1 informação nova (gatilho, horário, contexto, obstáculo ou pedido objetivo).
        - Em fase pós-orientação, prefira incluir um caso parecido breve e um novo ângulo relacionado.
        - Use no máximo UM destes elementos opcionais: pausa, marcador de WhatsApp, autocorreção ou micro erro.
        - Não transforme imperfeição em exagero; mantenha leitura natural.
        - Antes de escrever, confirme mentalmente: "o que quero neste turno" e "como estou me sentindo agora".
        - Não peça visita presencial, ida ao local ou acompanhamento físico; peça apenas apoio por mensagem/ligação online.
        """


def simulate_next_user_message(
    conversation,
    profile: SimulatedUserProfile,
    predefined_scenario: str = "",
    theme: Union[int, str, None] = None,
    inferred_gender: Optional[str] = None,
    force_context_expansion: bool = False,
) -> str:
    """Generate only the next user message based on recent history and emotional profile."""
    return SimulationUseCase().simulate_next_user_message(
        conversation=conversation,
        profile=profile,
        predefined_scenario=predefined_scenario,
        theme=theme,
        inferred_gender=inferred_gender,
        force_context_expansion=force_context_expansion,
    )


class SimulationUseCase:
    def __init__(self):
        self._llm_service = OpenAIService()
        self._theme_classifier = ThemeClassifier()

    def simulate_next_user_message(
        self,
        conversation,
        profile: SimulatedUserProfile,
        predefined_scenario: str = "",
        theme: Union[int, str, None] = None,
        inferred_gender: Optional[str] = None,
        force_context_expansion: bool = False,
        profile_instance: Optional[Profile] = None,
    ) -> str:
        result = self.simulate_next_user_message_with_metadata(
            conversation=conversation,
            profile=profile,
            predefined_scenario=predefined_scenario,
            theme=theme,
            inferred_gender=inferred_gender,
            force_context_expansion=force_context_expansion,
            profile_instance=profile_instance,
        )
        return result["content"]

    def _persist_simulated_behavior(
        self,
        *,
        profile_instance: Profile,
        controls: dict,
        selected_profile: SimulatedUserProfile,
        selected_scenario: str,
        selected_theme: Optional[int],
        content: str,
        payload: Optional[dict],
        force_context_expansion: bool,
    ) -> None:
        profile_instance.simulated_behavior = {
            "emotional_profile": selected_profile.value,
            "predefined_scenario": selected_scenario,
            "theme_id": selected_theme,
            "force_context_expansion": force_context_expansion,
            "generated_content": content,
            "llm_payload": payload,
            "controls": controls,
        }
        profile_instance.save(update_fields=["simulated_behavior", "updated_at"])

    def simulate_next_user_message_with_metadata(
        self,
        conversation,
        profile: SimulatedUserProfile,
        predefined_scenario: str = "",
        theme: Union[int, str, None] = None,
        inferred_gender: Optional[str] = None,
        force_context_expansion: bool = False,
        profile_instance: Optional[Profile] = None,
    ) -> dict:
        selected_profile = _parse_profile(profile)
        selected_scenario = (
            predefined_scenario if predefined_scenario in PREDEFINED_SCENARIOS else ""
        )
        full_conversation = list(conversation)
        available_theme_ids = set(Theme.objects.values_list("id", flat=True))
        selected_theme = _parse_optional_theme_id(theme)
        if selected_theme is not None and selected_theme not in available_theme_ids:
            raise ValueError(f"Theme '{selected_theme}' not found for simulation.")
        theme_options = _theme_options_from_db()
        feeling_label = PREDEFINED_SCENARIOS.get(
            selected_scenario, "está emocionalmente abalada"
        )
        problem_label = theme_options.get(selected_theme, "não ficou claro")
        recent_history = _to_recent_history(conversation=full_conversation, limit=8)
        history_text = ""
        last_assistant_message = ""
        assistant_messages_count = 0
        total_user_turns = 0
        for message in recent_history:
            history_text += f"{message['role'].upper()}: {message['content']}\n"
            if message["role"] == "assistant":
                last_assistant_message = message["content"]
                assistant_messages_count += 1
            elif message["role"] == "user":
                total_user_turns += 1

        grammatical_gender = "feminino"
        if inferred_gender == "male":
            grammatical_gender = "masculino"
        elif inferred_gender == "unknown":
            grammatical_gender = "neutro, evitando adjetivos com marca de gênero"

        should_avoid_explicit_spiritual_request = assistant_messages_count <= 1

        desire_label = PROFILE_LABELS.get(
            selected_profile, selected_profile.value.replace("_", " ")
        )
        persona_signature = _build_persona_signature(
            profile=selected_profile,
            scenario=selected_scenario,
            inferred_gender=inferred_gender,
            full_conversation=full_conversation,
        )
        conversation_state = _build_conversation_state(recent_history)
        next_user_turn = total_user_turns + 1
        turn_style_plan = _build_turn_style_plan(next_user_turn=next_user_turn)
        post_orientacao_phase = _detect_post_orientacao_phase(recent_history)
        related_topic_hint = _build_related_topic_hint(
            problem_label=problem_label,
            selected_profile=selected_profile,
            next_user_turn=next_user_turn,
        )
        turn_intent = _build_turn_intent(
            profile=selected_profile,
            last_assistant_message=last_assistant_message,
            feeling_label=feeling_label,
            force_context_expansion=force_context_expansion,
            post_orientacao_phase=post_orientacao_phase,
            related_topic_hint=related_topic_hint,
        )
        prompt = _build_simple_simulation_prompt(
            feeling_context=feeling_label,
            desire_context=desire_label,
            problem_context=problem_label,
            history_text=history_text,
            last_assistant_message=last_assistant_message,
            grammatical_gender=grammatical_gender,
            should_avoid_explicit_spiritual_request=should_avoid_explicit_spiritual_request,
            force_context_expansion=force_context_expansion,
            persona_signature=persona_signature,
            conversation_state=conversation_state,
            turn_style_plan=turn_style_plan,
            turn_intent=turn_intent,
        )
        content = (
            self._llm_service.basic_call(
                url_type="generate",
                prompt=prompt,
                max_tokens=SIMULATION_MAX_COMPLETION_TOKENS,
            )
            or ""
        ).strip()
        payload = self._llm_service.get_last_prompt_payload()
        if not content:
            raise ValueError(
                "Simulador retornou resposta vazia. Ajuste o prompt/modelo e tente novamente."
            )
        controls = {
            "persona_signature": persona_signature,
            "conversation_state": conversation_state,
            "turn_style_plan": turn_style_plan,
            "turn_intent": turn_intent,
        }
        if profile_instance is not None:
            self._persist_simulated_behavior(
                profile_instance=profile_instance,
                controls=controls,
                selected_profile=selected_profile,
                selected_scenario=selected_scenario,
                selected_theme=selected_theme,
                content=content,
                payload=payload,
                force_context_expansion=force_context_expansion,
            )

        return {
            "selected_profile": selected_profile.value,
            "selected_scenario": selected_scenario,
            "selected_theme_id": selected_theme,
            "content": content,
            "prompt": prompt,
            "payload": payload,
            "simulation_controls": controls,
        }

    def handle(
        self,
        profile_id: int,
        emotional_profile: Union[SimulatedUserProfile, str],
        predefined_scenario: str = "",
        theme: Union[int, str, None] = None,
        force_context_expansion: bool = False,
    ) -> int:
        profile = Profile.objects.get(id=profile_id)
        conversation = (
            Message.objects.filter(profile=profile)
            .exclude(role="system")
            .exclude(role="analysis")
            .exclude(exclude_from_context=True)
            .order_by("created_at")
        )
        simulation = self.simulate_next_user_message_with_metadata(
            conversation=conversation,
            profile=_parse_profile(emotional_profile),
            predefined_scenario=predefined_scenario,
            theme=theme,
            inferred_gender=profile.inferred_gender,
            force_context_expansion=force_context_expansion,
            profile_instance=profile,
        )
        theme_id = self._theme_classifier.classify(simulation["content"])
        selected_theme = Theme.objects.filter(id=theme_id).first()
        if not selected_theme:
            raise RuntimeError(f"Theme '{theme_id}' not found in database.")

        message = Message.objects.create(
            profile=profile,
            role="user",
            content=simulation["content"],
            channel="simulation",
            generated_by_simulator=True,
            ollama_prompt=simulation.get("payload"),
            theme=selected_theme,
        )
        message.block_root = message
        message.save(update_fields=["block_root"])
        return profile.id
