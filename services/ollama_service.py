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
import logging
import os
from typing import Any, Dict, Literal, Optional, Union
from urllib.parse import urljoin

import requests

from core.models import Message, Profile
from services.conversation_runtime import (
    MODE_ACOLHIMENTO,
    MODE_AMBIVALENCIA,
    MODE_EXPLORACAO,
    MODE_ORIENTACAO,
    MODE_WELCOME,
    contains_generic_empathy_without_grounding,
    contains_repeated_blocked_pattern,
    contains_unsolicited_spiritualization,
    contains_unverified_inference,
    detect_ambivalence,
    detect_direct_guidance_request,
    enforce_hard_limits,
    has_new_information,
    has_practical_action_step,
    has_repeated_user_pattern,
    is_semantic_loop,
    make_progress_fallback_question,
    semantic_similarity,
    should_force_progress_fallback,
    strip_opening_name_if_recently_used,
)
from services.rag_service import get_rag_context

logger = logging.getLogger(__name__)

VALID_CONVERSATION_MODES = {
    MODE_WELCOME,
    MODE_ACOLHIMENTO,
    MODE_EXPLORACAO,
    MODE_AMBIVALENCIA,
    MODE_ORIENTACAO,
}

LEGACY_MODE_MAP = {
    "acolhimento": MODE_ACOLHIMENTO,
    "exploração": MODE_EXPLORACAO,
    "orientação": MODE_ORIENTACAO,
}


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

        previous_mode = LEGACY_MODE_MAP.get(
            profile.conversation_mode, profile.conversation_mode
        )
        if previous_mode not in VALID_CONVERSATION_MODES:
            previous_mode = MODE_WELCOME

        is_first_message = queryset.filter(role="assistant").count() == 0
        direct_guidance_request = detect_direct_guidance_request(
            last_person_message.content
        )
        ambivalence_or_repeated = detect_ambivalence(
            last_person_message.content
        ) or has_repeated_user_pattern(recent_user_messages)
        new_information = has_new_information(recent_user_messages)
        loop_detected = ambivalence_or_repeated
        if len(recent_assistant_messages) >= 2:
            loop_detected = loop_detected or (
                semantic_similarity(
                    recent_assistant_messages[-1], recent_assistant_messages[-2]
                )
                > 0.85
            )

        # Explicit state machine:
        # WELCOME -> ACOLHIMENTO -> EXPLORACAO -> AMBIVALENCIA -> ORIENTACAO
        if direct_guidance_request:
            conversation_mode = MODE_ORIENTACAO
        elif is_first_message:
            conversation_mode = MODE_ACOLHIMENTO
        elif previous_mode == MODE_WELCOME:
            conversation_mode = MODE_ACOLHIMENTO
        elif ambivalence_or_repeated:
            conversation_mode = MODE_AMBIVALENCIA
        elif loop_detected:
            conversation_mode = MODE_ORIENTACAO
        elif new_information:
            conversation_mode = MODE_EXPLORACAO
        else:
            conversation_mode = previous_mode

        if conversation_mode == MODE_AMBIVALENCIA and (
            direct_guidance_request or has_repeated_user_pattern(recent_user_messages)
        ):
            conversation_mode = MODE_ORIENTACAO

        force_progress_fallback = (
            should_force_progress_fallback(
                recent_user_messages, recent_assistant_messages
            )
            and not new_information
        )

        state_runtime_rules = {
            MODE_WELCOME: """
            - objetivo: acolhimento inicial e abertura
            - não aprofundar diagnóstico
            - no máximo 1 pergunta breve
            """,
            MODE_ACOLHIMENTO: """
            - validar ponto específico dito pelo usuário
            - evitar reflexão longa
            - preparar transição para exploração objetiva
            """,
            MODE_EXPLORACAO: """
            - avançar a conversa com 1 pergunta concreta
            - usar evidência textual da última fala
            - não repetir moldes dos últimos turnos
            """,
            MODE_AMBIVALENCIA: """
            - reconhecer conflito explícito (sem moralizar)
            - separar sentimento de comportamento
            - perguntar um próximo critério de decisão
            """,
            MODE_ORIENTACAO: """
            - responder com 1 ação prática simples e imediata
            - sem espiritualização automática
            - sem reflexão emocional profunda
            - não fazer mais de 1 pergunta
            """,
        }

        prompt_aux = f"""
            MODO CONVERSACIONAL ATUAL: {conversation_mode}
            MODO ANTERIOR: {previous_mode}

            REGRAS DURAS:
            - Máximo 3 frases
            - Máximo 120 palavras
            - Máximo 1 pergunta
            - Não repetir saudação, primeira frase ou estrutura do último turno
            - Não usar frases genéricas repetidas
            - Não inferir sentimentos não explicitados pelo usuário
            - Não espiritualizar automaticamente sem contexto do usuário
            - Manter linguagem espiritual cristã com respeito e sem pregação
            - Empatia deve citar algo específico que o usuário disse

            REGRAS ESPECÍFICAS DO ESTADO:
            {state_runtime_rules.get(conversation_mode, state_runtime_rules[MODE_EXPLORACAO])}

            Se houver estagnação, faça pergunta destravadora concreta.
            Se o usuário pediu orientação prática, dê 1 passo acionável agora.
            Foque na última mensagem do usuário e avance o tema.
        """

        prompt_aux += (
            f"\n\nÚLTIMA MENSAGEM DO USUÁRIO:\n{last_person_message.content}\n"
        )
        if profile.theme:
            prompt_aux += f"\nTEMA DA RESPOSTA:\n{profile.theme.prompt}\n"

        prompt_aux += "\nULTIMAS CONVERSAS:\n"
        context_messages = queryset.exclude(id=last_person_message.id).order_by(
            "-created_at"
        )[:6]
        for message in reversed(list(context_messages)):
            prompt_aux += f"{message.role.upper()}: {message.content}\n\n"

        for rag_context in get_rag_context(last_person_message.content, limit=3):
            prompt_aux += f"RAG CONTEXT AUXILIAR (BAIXA PRIORIDADE): {rag_context}\n"

        max_regenerations = 3
        regeneration_counter = 0
        loop_counter = 1 if loop_detected else 0

        base_temperature = 0.6
        if conversation_mode in {MODE_ORIENTACAO, MODE_AMBIVALENCIA}:
            base_temperature = 0.45
        selected_temperature = base_temperature
        approved_content = ""

        if force_progress_fallback:
            approved_content = make_progress_fallback_question()
            loop_counter += 1

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
            unsolicited_spiritualization = contains_unsolicited_spiritualization(
                last_person_message.content, candidate
            )
            generic_empathy = contains_generic_empathy_without_grounding(
                last_person_message.content, candidate
            )
            missing_practical_step = (
                direct_guidance_request and not has_practical_action_step(candidate)
            )

            if (
                semantic_loop
                or blocked_template
                or has_unverified_inference
                or unsolicited_spiritualization
                or generic_empathy
                or missing_practical_step
            ):
                regeneration_counter += 1
                if semantic_loop:
                    loop_counter += 1
                continue

            approved_content = candidate

        if not approved_content:
            approved_content = enforce_hard_limits(make_progress_fallback_question())

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
