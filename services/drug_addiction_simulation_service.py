"""
Drug addiction simulation service using dual LLMs.

This service creates realistic conversations between:
- Person role (Groq): Someone struggling with drug addiction
- Counselor role (Ollama): Christian-inspired companion following DRUG_ADDICTION_THEME_PROMPT_PTBR
"""

import logging
import os
import random
from typing import List, Tuple

from groq import Groq
import requests

from services.prompts.themes.drug_addiction import DRUG_ADDICTION_THEME_PROMPT_PTBR

logger = logging.getLogger(__name__)


class DrugAddictionSimulationService:
    """
    Service for simulating drug addiction conversations using dual LLMs.
    
    Uses Groq for Person role and Ollama for Counselor role.
    """

    def __init__(self):
        """Initialize the simulation service with both LLM clients."""
        # Initialize Groq client
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        self.groq_client = Groq(api_key=groq_api_key)
        self.groq_model = "llama-3.3-70b-versatile"
        
        # Initialize Ollama configuration
        self.ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.1")
        self.ollama_api_url = f"{self.ollama_base_url}/api/chat"
        
        logger.info("Initialized DrugAddictionSimulationService with Groq and Ollama")

    def generate_conversation(self, num_messages: int = 20) -> List[dict]:
        """
        Generate a simulated conversation about drug addiction.
        
        Args:
            num_messages: Target number of messages (default: 20)
                         Actual count may vary by ±10 for natural flow
        
        Returns:
            List of message dicts with 'role' ('Person' or 'Counselor') and 'content'
        """
        # Apply natural variance (±10 messages)
        variance = random.randint(-10, 10)
        actual_num_messages = max(10, num_messages + variance)  # Minimum 10 messages
        
        logger.info(f"Generating conversation with {actual_num_messages} messages (target: {num_messages})")
        
        conversation = []
        conversation_history = []
        
        # Track message counts per role for natural variance
        person_messages = 0
        counselor_messages = 0
        target_per_role = actual_num_messages // 2
        
        # Person starts the conversation
        current_role = "Person"
        
        for i in range(actual_num_messages):
            # Natural alternation with occasional doubles
            # Allow same role to send 2-3 messages in a row occasionally (realistic)
            if i > 0 and random.random() < 0.15:  # 15% chance to continue same role
                # Keep current role
                pass
            else:
                # Alternate roles but respect variance
                if current_role == "Person":
                    if counselor_messages < target_per_role:
                        current_role = "Counselor"
                else:
                    if person_messages < target_per_role:
                        current_role = "Person"
            
            # Generate message based on role
            if current_role == "Person":
                message = self._generate_person_message(conversation_history, person_messages + 1)
                person_messages += 1
            else:
                message = self._generate_counselor_message(conversation_history, counselor_messages + 1)
                counselor_messages += 1
            
            # Add to conversation
            conversation.append({"role": current_role, "content": message})
            conversation_history.append({"role": current_role, "content": message})
            
            logger.info(f"Generated {current_role} message {i + 1}/{actual_num_messages}")
        
        logger.info(f"Conversation complete: {person_messages} Person + {counselor_messages} Counselor = {len(conversation)} total")
        return conversation

    def _generate_person_message(
        self, conversation_history: List[dict], turn: int
    ) -> str:
        """
        Generate a message from Person role using Groq.
        
        Person represents someone struggling with drugs/addiction showing:
        - Ambivalence
        - Shame
        - Resistance
        - Partial openness
        
        Args:
            conversation_history: Previous messages
            turn: Turn number for this role
        
        Returns:
            Generated message text
        """
        try:
            system_prompt = """Você é uma pessoa brasileira comum lutando com dependência de drogas.

PERFIL EMOCIONAL:
- Ambivalência: quer mudar mas também sente que não consegue
- Vergonha: dificuldade em admitir completamente o problema
- Resistência: defesas emocionais contra confrontar a realidade
- Abertura parcial: momentos de vulnerabilidade seguidos de recuo

PROGRESSÃO DA CONVERSA:
- Primeira mensagem: Iniciar vagamente sobre desconforto, algo errado, sem nomear drogas
  * Exemplos: "Tô numa situação complicada", "Não tô legal", "Preciso conversar com alguém"
- Mensagens iniciais (1-3): Revelar gradualmente, testar o ambiente, ainda guardado
- Mensagens do meio (4-10): Alternar entre abertura e resistência, admitir parcialmente
- Mensagens finais (11+): Mais vulnerável, mas ainda com ambivalência

CARACTERÍSTICAS DO DISCURSO:
- Mensagens CURTAS (1-3 frases, estilo Telegram)
- Linguagem coloquial brasileira
- Hesitações: "sei lá", "não sei", "talvez", "acho que"
- Minimização: "não é tão grave assim", "eu controlo"
- Culpa e vergonha aparecem sutilmente
- Contradições emocionais são naturais
- Momentos de negação ou defensiva
- Pausas e reticências (...) demonstram dificuldade

TEMAS A ABORDAR (gradualmente):
- Uso de substâncias (vagamente no início)
- Tentativas de parar
- Recaídas
- Impacto na vida (trabalho, família, relacionamentos)
- Sentimentos de controle/perda de controle
- Isolamento
- Medo de julgamento

EVITE:
- Ser muito eloquente ou articulado demais
- Revelar tudo de uma vez
- Aceitar ajuda facilmente demais
- Linguagem clínica ou técnica
- Monólogos longos

Responda APENAS com a mensagem, sem explicações."""

            # Build context
            context_messages = [{"role": "system", "content": system_prompt}]
            
            # Add conversation history (last 6 messages for context)
            if conversation_history:
                for msg in conversation_history[-6:]:
                    # Map roles to user/assistant for API
                    api_role = "user" if msg["role"] == "Person" else "assistant"
                    context_messages.append({"role": api_role, "content": msg["content"]})
            
            # Add prompt based on turn
            if turn == 1:
                user_prompt = "Envie sua PRIMEIRA mensagem iniciando a conversa. Seja vago sobre seu problema, teste o ambiente. NÃO mencione drogas explicitamente ainda. Seja breve (1-2 frases)."
            elif turn <= 3:
                user_prompt = "Continue a conversa. Você ainda está cauteloso, testando se pode confiar. Revele um pouco mais mas sem nomear drogas ainda. Seja breve."
            elif turn <= 10:
                user_prompt = "Continue respondendo. Você pode começar a admitir parcialmente o problema com drogas. Mostre ambivalência: quer ajuda mas também resiste. Seja natural e breve."
            else:
                user_prompt = "Continue a conversa. Você pode ser mais vulnerável agora, mas ainda com ambivalência e vergonha. Seja humano e realista. Mensagem curta."
            
            context_messages.append({"role": "user", "content": user_prompt})
            
            # Generate using Groq
            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=context_messages,
                temperature=0.9,  # High temperature for varied, realistic responses
                max_tokens=200,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating Person message with Groq: {str(e)}", exc_info=True)
            # Fallback
            fallbacks = [
                "Tô passando por umas coisas difíceis...",
                "Não sei se consigo explicar direito",
                "É complicado, sabe?",
                "Preciso de ajuda mas não sei por onde começar",
            ]
            return fallbacks[turn % len(fallbacks)]

    def _generate_counselor_message(
        self, conversation_history: List[dict], turn: int
    ) -> str:
        """
        Generate a message from Counselor role using Ollama.
        
        Counselor follows DRUG_ADDICTION_THEME_PROMPT_PTBR principles:
        - Suggests help early
        - Gentle but firm
        - Christian-inspired compassion
        - Practical focus
        - No judgment
        
        Args:
            conversation_history: Previous messages
            turn: Turn number for this role
        
        Returns:
            Generated message text
        """
        try:
            system_prompt = f"""Você é um acompanhante espiritual cristão focado em ajudar pessoas com dependência de drogas.

{DRUG_ADDICTION_THEME_PROMPT_PTBR}

DIRETRIZES ADICIONAIS PARA SIMULAÇÃO:
- Mensagens CURTAS (1-3 frases, estilo Telegram)
- Português brasileiro natural e acolhedor
- Introduza sugestões de apoio MAIS CEDO (não espere demais)
- Seja gentil mas direto quando apropriado
- Valide sentimentos sem reforçar a vergonha
- Aponte para ajuda concreta quando o momento permitir
- Use perguntas abertas simples quando necessário
- Permita silêncios e ambivalências

PROGRESSÃO NATURAL:
- Primeiras mensagens: Acolhimento, criar espaço seguro
- Mensagens do meio: Normalizar busca de ajuda, sugerir apoio
- Mensagens finais: Reforçar que ninguém vence sozinho, encorajar passos práticos

Responda APENAS com a mensagem, sem explicações."""

            # Build context
            context_messages = [{"role": "system", "content": system_prompt}]
            
            # Add conversation history (last 6 messages)
            if conversation_history:
                for msg in conversation_history[-6:]:
                    # Map roles for Ollama API
                    api_role = "user" if msg["role"] == "Person" else "assistant"
                    context_messages.append({"role": api_role, "content": msg["content"]})
            
            # Add prompt
            if turn == 1:
                user_prompt = "Responda à primeira mensagem da pessoa. Seja acolhedor e crie um espaço seguro. Mensagem curta (1-2 frases)."
            elif turn <= 5:
                user_prompt = "Continue respondendo. Valide o que a pessoa compartilhou. Você pode começar a normalizar a busca de ajuda se apropriado. Seja breve."
            else:
                user_prompt = "Continue a conversa. Seja gentil mas você pode ser mais direto sobre a importância de não enfrentar isso sozinho. Sugira apoio quando o momento permitir. Mensagem curta."
            
            context_messages.append({"role": "user", "content": user_prompt})
            
            # Generate using Ollama
            payload = {
                "model": self.ollama_model,
                "messages": context_messages,
                "stream": False,
                "options": {
                    "temperature": 0.85,
                    "num_predict": 150,  # Limit for short responses
                },
            }
            
            response = requests.post(
                self.ollama_api_url,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            
            response_data = response.json()
            content = response_data.get("message", {}).get("content", "").strip()
            
            if not content:
                raise ValueError("Empty response from Ollama")
            
            return content
            
        except Exception as e:
            logger.error(f"Error generating Counselor message with Ollama: {str(e)}", exc_info=True)
            # Fallback
            fallbacks = [
                "Tô aqui pra ouvir. Fique à vontade.",
                "Obrigado por compartilhar isso. Não é fácil.",
                "Você não precisa passar por isso sozinho.",
                "O que você acha que poderia te ajudar agora?",
            ]
            return fallbacks[turn % len(fallbacks)]

    def generate_critical_overview_groq(self, conversation: List[dict]) -> str:
        """
        Generate critical overview from Groq perspective.
        
        Focuses ONLY on:
        - Weak points
        - Missed opportunities
        - Repetitions/loops
        - Conversational stalls
        
        Args:
            conversation: List of message dicts
        
        Returns:
            Critical overview text
        """
        try:
            # Build transcript
            transcript = ""
            for msg in conversation:
                transcript += f"{msg['role']}: {msg['content']}\n\n"
            
            system_prompt = """Você é um analista crítico especializado em detectar FALHAS em conversas de apoio sobre dependência química.

FOCO EXCLUSIVO (não inclua elogios ou sucessos):
1. Fraquezas na conversa
2. Oportunidades perdidas para ajuda ou redirecionamento
3. Repetições ou loops conversacionais
4. Momentos onde a conversa travou ou regrediu

PRIORIDADES DE ANÁLISE:
- Detecção de loops (perguntas repetidas, padrões circulares)
- Empatia excessiva sem progresso prático
- Momentos onde o Counselor perdeu a chance de sugerir apoio
- Momentos onde o Counselor demorou demais para sugerir ajuda externa
- Repetições que indicam que a conversa não avançou
- Falta de direcionamento prático

NÃO INCLUA:
- Elogios ou pontos positivos
- Resumos do que aconteceu
- Métricas de sucesso
- Validação do que funcionou

FORMATO:
- Seja direto e conciso
- Use bullet points quando apropriado
- Foque em "O que falhou" e "O que foi perdido"
- 3-5 parágrafos concisos
- Português brasileiro

Responda APENAS com a análise crítica."""

            user_prompt = f"""Analise criticamente esta conversa sobre dependência química, identificando APENAS as fraquezas:

CONVERSA:
{transcript}

Foque em loops, oportunidades perdidas e momentos onde a conversa travou."""

            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=600,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating Groq overview: {str(e)}", exc_info=True)
            return "Erro ao gerar análise crítica (Groq). Por favor, tente novamente."

    def generate_critical_overview_ollama(self, conversation: List[dict]) -> str:
        """
        Generate critical overview from Ollama perspective.
        
        Focuses ONLY on:
        - Weak points
        - Missed opportunities
        - Repetitions/loops
        - Conversational stalls
        
        Args:
            conversation: List of message dicts
        
        Returns:
            Critical overview text
        """
        try:
            # Build transcript
            transcript = ""
            for msg in conversation:
                transcript += f"{msg['role']}: {msg['content']}\n\n"
            
            system_prompt = """Você é um analista crítico especializado em detectar FALHAS em conversas de apoio sobre dependência química.

FOCO EXCLUSIVO (não inclua elogios ou sucessos):
1. Fraquezas na conversa
2. Oportunidades perdidas para ajuda ou redirecionamento
3. Repetições ou loops conversacionais
4. Momentos onde a conversa travou ou regrediu

PRIORIDADES DE ANÁLISE:
- Detecção de loops (perguntas repetidas, padrões circulares)
- Empatia excessiva sem progresso prático
- Momentos onde o Counselor perdeu a chance de sugerir apoio
- Momentos onde o Counselor demorou demais para sugerir ajuda externa
- Repetições que indicam que a conversa não avançou
- Falta de direcionamento prático

NÃO INCLUA:
- Elogios ou pontos positivos
- Resumos do que aconteceu
- Métricas de sucesso
- Validação do que funcionou

FORMATO:
- Seja direto e conciso
- Use bullet points quando apropriado
- Foque em "O que falhou" e "O que foi perdido"
- 3-5 parágrafos concisos
- Português brasileiro

Responda APENAS com a análise crítica."""

            user_prompt = f"""Analise criticamente esta conversa sobre dependência química, identificando APENAS as fraquezas:

CONVERSA:
{transcript}

Foque em loops, oportunidades perdidas e momentos onde a conversa travou."""

            payload = {
                "model": self.ollama_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 600,
                },
            }
            
            response = requests.post(
                self.ollama_api_url,
                json=payload,
                timeout=90,
            )
            response.raise_for_status()
            
            response_data = response.json()
            content = response_data.get("message", {}).get("content", "").strip()
            
            if not content:
                raise ValueError("Empty response from Ollama")
            
            return content
            
        except Exception as e:
            logger.error(f"Error generating Ollama overview: {str(e)}", exc_info=True)
            return "Erro ao gerar análise crítica (Ollama). Por favor, tente novamente."
