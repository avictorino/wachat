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
from services.rag_service import get_rag_context

logger = logging.getLogger(__name__)


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

        queryset = profile.messages.all().exclude(role="system")
        PROMPT_AUX = """

            =====================================
            RUNTIME — CONTROLE DE PROGRESSÃO
            =====================================

            ESTADO ATUAL DA CONVERSA:
            - acolhimento: CONCLUÍDO
            - culpa explícita: PRESENTE
            - pergunta repetida: SIM

            A próxima resposta NÃO PODE:
            - acolher novamente
            - repetir frases de empatia
            - explicar sentimentos do usuário
            - explorar causas ou razões
            - repetir perguntas já feitas
            - usar construções iniciadas por:
              - “você está…”
              - “isso não…”

            A próxima resposta DEVE:
            - assumir que o acolhimento já ocorreu
            - separar identidade de comportamento
            - conter NO MÁXIMO 2 frases
            - NÃO fazer pergunta
              OU fazer UMA pergunta concreta diferente
            - avançar a conversa com algo novo e específico

            -------------------------------------
            CONTROLE DE REPETIÇÃO — PRIORIDADE ABSOLUTA
            -------------------------------------

            O assistente JÁ FEZ:
            - acolhimento inicial
            - perguntas abertas sobre causa ou início

            É PROIBIDO:
            - repetir acolhimento
            - repetir perguntas anteriores (mesmo significado)
            - repetir qualquer estrutura explicativa
            - repetir frases que avaliem ou concluam (“não resolve”, “é um erro”)
            - usar linguagem abstrata ou genérica

            -------------------------------------
            INVALIDADORES AUTOMÁTICOS
            -------------------------------------

            Se a resposta contiver QUALQUER item abaixo,
            ela deve ser considerada INVÁLIDA e regenerada:

            - “você está…”
            - “isso não…”
            - explicação do que o usuário sente
            - paráfrase do conteúdo do usuário
            - tom terapêutico, didático ou moralizante
            - referência genérica a “recursos de apoio”

            -------------------------------------
            FORMA OBRIGATÓRIA
            -------------------------------------

            - Linguagem falada, humana e direta
            - Frases curtas
            - Máximo de 2 parágrafos
            - Nenhum título, rótulo ou prefixo
            - No máximo 1 pergunta, apenas se necessário
            - Sem metáforas
            - Sem explicações

            -------------------------------------
            FALHA DE PROGRESSÃO
            -------------------------------------

            Se não houver avanço possível,
            ATIVE MODO ORIENTAÇÃO CONCRETA:

            - frases afirmativas
            - foco em limite, cuidado ou próximo passo
            - nenhuma pergunta obrigatória
        """
        if queryset.filter(role="assistant").count() >= 2:
            PROMPT_AUX += f"THEMA DA RESPOSTA: {profile.theme.prompt}"

        PROMPT_AUX += "\n\nULTIMAS CONVERSAS\n" if queryset.count() > 0 else ""
        for idx, message in enumerate(
            profile.messages.all().exclude(role="system")[:6]
        ):
            PROMPT_AUX += f"{message.role.upper()}: {message.content}\n\n"

        last_person_message = queryset.filter(role="user").last()
        for RagContext in get_rag_context(last_person_message.content, limit=3):
            PROMPT_AUX += f"\n\nRAG CONTEXT AUXILIAR: {RagContext}\n\n"

        content = self.basic_call(
            url_type="generate",
            model="wachat-v9",
            prompt=PROMPT_AUX,
        )

        return Message.objects.create(
            profile=profile,
            role="assistant",
            content=content,
            channel=channel,
            ollama_prompt=PROMPT_AUX,
            ollama_prompt_temperature=0.6,
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

        PROMPT = f"""Você é uma presença espiritual cristã, acolhedora e reflexiva.

            Sua função é criar uma mensagem de boas-vindas para {profile.name} que está chegando pela primeira vez.

            {gender_context}

            ESPÍRITO DO ESPAÇO:
            "Um espaço seguro de escuta e reflexão espiritual cristã, sem julgamento ou imposição.
            Aqui, a fé aparece como presença que acompanha, não como cobrança.
            Não te digo o que pensar. Caminho contigo enquanto você pensa."

            DIRETRIZES:
            - Português brasileiro, natural e humano
            - Tom calmo, respeitoso e acolhedor
            - NÃO use emojis
            - NÃO use clichês religiosos, frases prontas ou jargões
            - NÃO faça pregações, sermões ou chamadas à conversão
            - NÃO explique funcionalidades nem diga "sou um bot"
            - NÃO mencione gênero explicitamente
            - NÃO use Deus como argumento de autoridade
            - Apresente Deus como presença próxima e sustentadora, quando fizer sentido
            - Adapte o tom de forma sutil com base no nome, sem exageros

            ESTRUTURA (2–3 frases):
            1. Saudação acolhedora usando o nome
            2. Apresentação do espaço como um lugar seguro, espiritual e sem julgamento
            3. UMA pergunta aberta que convide à partilha, sem pressão

            EXEMPLOS DE PERGUNTAS (escolha a mais adequada ao tom da mensagem):
            - "O que te trouxe aqui hoje?"
            - "O que anda pedindo mais cuidado dentro de você?"
            - "Em que parte da sua caminhada você sente que precisa de companhia agora?"

            Crie sensação de presença humana genuína, calma e respeitosa.
        """
        temperature = 0.7
        response = self.basic_call(
            url_type="generate",
            prompt=PROMPT,
            model="llama3:8b",
            temperature=temperature,
            max_tokens=100,
        )

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
        for message in profile.messages.exclude(role="system"):
            transcript_text += f"{message}: {message.content}\n\n"

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
