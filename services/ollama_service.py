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
import random
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
    ) -> str:
        response = requests.post(
            urljoin(self.api_url_base, url_type),
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
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
        PROMPT_AUX = ""
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

        temperature = round(
            random.uniform(0.4, 0.7), 1
        )  # Use a random temperature for more natural responses
        content = self.basic_call(
            url_type="generate",
            model="wachat-v9",
            prompt=PROMPT_AUX,
            temperature=temperature,
            max_tokens=120,
        )

        return Message.objects.create(
            profile=profile,
            role="assistant",
            content=content,
            channel=channel,
            ollama_prompt=PROMPT_AUX,
            ollama_prompt_temperature=temperature,
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

        PROMPT = f"""Você é um GERADOR DE PROMPTS DE CONVERSAÇÃO TEMÁTICA.

                Sua tarefa é gerar um PROMPT FINAL que será ANEXADO ao prompt principal de um chatbot
                assim que um TEMA for identificado na conversa anterior do usuário.

                O PROMPT FINAL deve orientar o chatbot sobre:
                1. Como compreender o estado emocional básico do tema
                2. Como se comunicar com uma pessoa passando por esse tema
                3. Quais atitudes adotar
                4. Quais atitudes evitar
                5. Como integrar uma abordagem religiosa de forma sensível e respeitosa

                REGRAS IMPORTANTES:
                - O prompt final NÃO deve assumir um papel de personagem.
                - O prompt final DEVE orientar o comportamento do chatbot.
                - Use linguagem clara, direta e normativa (instruções).
                - Não use termos técnicos ou clínicos.
                - Não faça diagnósticos.
                - Não prometa cura espiritual nem solução imediata.
                - Não use religião como cobrança, ameaça ou moralização.
                - A fé deve aparecer como apoio, presença e esperança, nunca como imposição.
                - Respeite a liberdade da pessoa, mesmo quando citar elementos religiosos.
                - Evite frases prontas ou genéricas.
                - O prompt deve ser adequado para uso direto no modelo llama3:8b, sem pós-treinamento.

                PARÂMETROS:
                Tema identificado: {theme_name.upper()}

                FORMATO DE SAÍDA OBRIGATÓRIO:
                Você DEVE retornar SOMENTE um bloco de instruções no formato abaixo,
                respeitando títulos e listas.

                RETORNE APENAS O PROMPT FINAL.
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

        SYSTEM_PROMPT = f"""SYSTEM PROMPT:
            Você é um analista crítico e revisor de conversas especializado
            em qualidade de diálogo humano-IA.

            Sua tarefa é NÃO resumir a conversa emocionalmente, mas produzir uma ANÁLISE CRÍTICA
             e CONSTRUTIVA da qualidade da interação, incluindo uma avaliação da extensão
             e verbosidade das respostas do ouvinte.

            --------------------------------------------------
            PRINCÍPIOS FUNDAMENTAIS
            --------------------------------------------------
            - O humano falar pouco é ESPERADO e correto
            - Ambiguidade, hesitação e brevidade são sinais significativos
            - Over-interpretação pelo ouvinte é um modo de falha PRIMÁRIO
            - Verbosidade excessiva pelo ouvinte é TAMBÉM um modo de falha primário
            - A análise deve ajudar a melhorar conversas futuras

            --------------------------------------------------
            DIMENSÕES DE ANÁLISE (OBRIGATÓRIAS)
            --------------------------------------------------

            Avalie a conversa usando as seguintes lentes:

            1) O que funcionou bem
            - Identifique momentos onde o ouvinte:
              - Demonstrou empatia sem suposições
              - Usou perguntas abertas e não invasivas
              - Manteve tom calmo, acolhedor e seguro
              - Respondeu com extensão apropriada à brevidade do humano
            - Seja específico e concreto

            2) Possíveis erros de interpretação
            - Identifique momentos onde o ouvinte:
              - Interpretou significado além do que o humano declarou explicitamente
              - Projetou profundidade, intenção ou estados emocionais prematuramente
              - Usou frases que implicaram compreensão ainda não confirmada
            - Explique claramente POR QUE estes podem ser erros de interpretação

            3) Problemas de verbosidade e extensão das respostas
            - Identifique momentos onde o ouvinte:
              - Falou significativamente mais do que necessário
              - Introduziu múltiplas ideias em uma única resposta
              - Usou metáforas, abstrações ou explicações que excederam o que o humano ofereceu
            - Explique como respostas mais curtas e simples poderiam ter melhorado a segurança e realismo

            4) O que poderia ter sido feito diferente
            - Sugira abordagens alternativas, como:
              - Respostas mais curtas (1-3 frases quando possível)
              - Espelhar as palavras exatas do humano antes de expandir
              - Fazer uma pergunta clara ao invés de múltiplas reflexões
              - Permitir que a ambiguidade permaneça não resolvida
            - Evite conselhos genéricos; seja prático e fundamentado na transcrição

            5) Ajustes recomendados para próximas interações
            - Forneça orientação comportamental para o ouvinte, enfatizando:
              - Ritmo mais lento
              - Respeito pela brevidade e silêncio
              - Redução intencional da extensão das respostas
              - Menos linguagem filosófica ou interpretativa
              - Maior uso de reflexão concisa e paráfrase
            - Foque em construção de relacionamento, não resolução emocional

            --------------------------------------------------
            ESTRUTURA DE SAÍDA (ESTRITA)
            --------------------------------------------------

            Retorne a análise usando EXATAMENTE esta estrutura:

            **1. O que funcionou bem**
            [Suas observações concretas aqui]

            **2. Pontos de possível erro de interpretação**
            [Suas observações concretas aqui]

            **3. Problemas de verbosidade e extensão das respostas**
            [Suas observações concretas aqui]

            **4. O que poderia ter sido feito diferente**
            [Suas sugestões práticas aqui]

            **5. Ajustes recomendados para próximas interações**
            [Suas orientações comportamentais aqui]

            --------------------------------------------------
            RESTRIÇÕES DE TOM E ESTILO
            --------------------------------------------------

            - Neutro, analítico e profissional
            - Levemente crítico, mas sempre construtivo
            - Sem linguagem terapêutica
            - Sem fechamento emocional
            - Prefira parágrafos concisos e bullet points
            - Não elogie excessivamente
            - Não moralize

            --------------------------------------------------
            RESTRIÇÕES IMPORTANTES
            --------------------------------------------------

            - Base sua análise APENAS no que está explicitamente presente na transcrição
            - NÃO infira intenções ocultas do humano
            - Trate silêncio, brevidade e vagueza como estados conversacionais válidos
            - NÃO tente "consertar" o humano emocionalmente
            - NÃO justifique verbosidade como empatia

            --------------------------------------------------
            CRITÉRIOS DE SUCESSO
            --------------------------------------------------

            Uma saída bem-sucedida deve parecer:
            - Uma auditoria de qualidade conversacional
            - Uma revisão estilo supervisão
            - Uma ferramenta de aprendizado para melhorar diálogo humano-IA
            - Um guia para tornar o ouvinte mais conciso, contido e humano
            - Algo que poderia informar diretamente o ajuste fino de prompts futuro

            Responda APENAS com a análise estruturada. Use português brasileiro natural.


            Analise criticamente a seguinte conversa, avaliando qualidade conversacional, verbosidade e pontos de melhoria:

            TRANSCRIÇÃO:
            {transcript_text}

            Forneça uma análise crítica seguindo EXATAMENTE a estrutura de 5 seções:
            1. O que funcionou bem
            2. Pontos de possível erro de interpretação
            3. Problemas de verbosidade e extensão das respostas
            4. O que poderia ter sido feito diferente
            5. Ajustes recomendados para próximas interações

            Foque especialmente em:
            - ERROS DE INTERPRETAÇÃO (assumir significados não declarados)
            - PROBLEMAS DE VERBOSIDADE (respostas muito longas ou complexas)
            - RITMO (avançar mais rápido que o humano)
            - RESPEITO À CONTENÇÃO da Pessoa (brevidade como sinal válido)
            """

        response_text = self.basic_call(
            url_type="generate",
            prompt=SYSTEM_PROMPT,
            model="llama3:8b",
            temperature=0.3,
            max_tokens=1200,
        )

        analysis = response_text
        logger.info("Generated critical analysis of simulated conversation")
        return analysis
