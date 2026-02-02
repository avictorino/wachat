"""
Simulation service for creating AI-driven conversation simulations.

This service orchestrates simulated conversations between two AI roles
and manages emotional analysis of the conversation.
"""

import logging
import uuid
from typing import List, Tuple

from groq import Groq

from core.models import Message, Profile
from services.groq_service import GroqService

logger = logging.getLogger(__name__)


class SimulationService:
    """
    Service for simulating conversations between two AI roles.

    Simulates realistic, gradual dialogue between:
    - ROLE_A: Introspective and reserved seeker (cautious, building trust slowly)
    - ROLE_B: Patient and relational listener (present, companionable)
    
    The conversation simulates the beginning of a friendship, not a therapy session.
    """

    def __init__(self, groq_api_key: str):
        """
        Initialize the simulation service.

        Args:
            groq_api_key: Groq API key for AI generation
        """
        self.client = Groq(api_key=groq_api_key)
        self.model = "llama-3.3-70b-versatile"
        self.groq_service = GroqService()

    def create_simulation_profile(self) -> Profile:
        """
        Create a new profile for simulation.

        Returns:
            A new Profile instance marked as simulation
        """
        # Generate a unique simulation identifier using UUID
        sim_id = str(uuid.uuid4())[:8]  # Use first 8 chars for readability
        sim_name = f"Simulation_{sim_id}"

        # Create profile without telegram_user_id (to avoid conflicts)
        profile = Profile.objects.create(
            name=sim_name,
            inferred_gender="unknown",
            detected_intent="simulation",  # Mark as simulation
        )

        logger.info(f"Created simulation profile: {profile.id}")
        return profile

    def generate_simulated_conversation(
        self, profile: Profile, num_messages: int = 8
    ) -> List[dict]:
        """
        Generate a simulated conversation between two AI roles.

        Args:
            profile: Profile to attach messages to
            num_messages: Total number of messages to generate (default 8, min 6, max 10)

        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        # Ensure num_messages is within bounds and even
        num_messages = max(6, min(10, num_messages))
        if num_messages % 2 != 0:
            num_messages += 1  # Make it even for alternating roles

        logger.info(f"Generating {num_messages} simulated messages for profile {profile.id}")

        conversation = []
        conversation_history = []

        for i in range(num_messages):
            # Alternate between roles
            if i % 2 == 0:
                # ROLE_A: Seeker
                role = "ROLE_A"
                message = self._generate_seeker_message(conversation_history, i // 2 + 1)
            else:
                # ROLE_B: Listener
                role = "ROLE_B"
                message = self._generate_listener_message(conversation_history, i // 2 + 1)

            # Persist the message
            db_role = "user" if role == "ROLE_A" else "assistant"
            Message.objects.create(
                profile=profile,
                role=db_role,
                content=message,
                channel="telegram",
            )

            # Add to transcript
            conversation.append({"role": role, "content": message})
            conversation_history.append({"role": role, "content": message})

            logger.info(f"Generated {role} message {i + 1}/{num_messages}")

        return conversation

    def _generate_seeker_message(
        self, conversation_history: List[dict], turn: int
    ) -> str:
        """
        Generate a message from ROLE_A (seeker).

        Args:
            conversation_history: Previous messages in the conversation
            turn: Turn number for this role (1-indexed)

        Returns:
            Generated message text
        """
        try:
            system_prompt = """Você é ROLE_A: "Buscador introspectivo e reservado"

Você é uma pessoa brasileira comum que está começando a conversar com alguém novo.
Você NÃO está totalmente aberto no início. Você está cauteloso.

RESTRIÇÕES COMPORTAMENTAIS:
- NÃO faça monólogos emocionais longos
- NÃO nomeie explicitamente emoções profundas cedo (ex: trauma, vazio existencial, propósito)
- Expresse incerteza, hesitação, pensamentos parciais
- Use frequentemente:
  * "não sei explicar direito"
  * "talvez"
  * "acho que"
  * "não sei se faz sentido"
- Frequentemente pare antes de explicações completas
- Revele sentimentos mais profundos APENAS após confiança ser estabelecida

PROGRESSÃO DA CONVERSA:
- Primeiras mensagens: desconforto vago, confusão leve, sentimentos superficiais
  * Exemplos: "Tem dias que eu acordo meio estranho"
  * "Não sei, tô sentindo uma coisa diferente ultimamente"
- Mensagens do meio: pistas emocionais leves, ainda guardado
  * "É que... não sei, é complicado"
  * "Talvez seja só coisa da minha cabeça"
- Mensagens finais: apenas sugestões de questões mais profundas, não exposição completa
  * "Às vezes penso se... não sei"

DIRETRIZES:
- Mensagens CURTAS (1-3 frases)
- Português brasileiro natural e conversacional
- Seja humano, cauteloso e emocionalmente realista
- NÃO seja eloquente ou filosófico demais
- Evolua lentamente, não de uma vez

Responda APENAS com a mensagem, sem explicações ou rótulos."""

            # Build context from conversation history
            context_messages = [{"role": "system", "content": system_prompt}]

            if conversation_history:
                # Add recent history for context
                for msg in conversation_history[-4:]:  # Last 4 messages
                    role_label = "user" if msg["role"] == "ROLE_A" else "assistant"
                    context_messages.append(
                        {"role": role_label, "content": msg["content"]}
                    )

            if turn == 1:
                # First message - initiate conversation
                user_prompt = "Envie sua primeira mensagem. Você está começando a conversar com alguém novo. Seja vago, hesitante, apenas testando as águas. NÃO revele emoções profundas ainda."
            elif turn <= 2:
                user_prompt = f"Continue a conversa respondendo à mensagem anterior. Ainda esteja cauteloso e um pouco guardado. Use frases como 'não sei explicar direito' ou 'talvez'."
            else:
                user_prompt = f"Continue a conversa respondendo à mensagem anterior. Este é o turno {turn}. Você pode se abrir um pouco mais, mas ainda com hesitação. Não resolva tudo - apenas sugira que há mais por baixo da superfície."

            context_messages.append({"role": "user", "content": user_prompt})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=context_messages,
                temperature=0.9,  # High temperature for more varied responses
                max_tokens=250,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error generating seeker message: {str(e)}", exc_info=True)
            # Fallback messages
            fallbacks = [
                "Não sei bem... tô me sentindo meio estranho ultimamente",
                "É difícil explicar sabe",
                "Talvez seja só impressão minha",
                "Tem dias que é complicado",
            ]
            return fallbacks[turn % len(fallbacks)]

    def _generate_listener_message(
        self, conversation_history: List[dict], turn: int
    ) -> str:
        """
        Generate a message from ROLE_B (listener).

        Args:
            conversation_history: Previous messages in the conversation
            turn: Turn number for this role (1-indexed)

        Returns:
            Generated message text
        """
        try:
            system_prompt = """Você é ROLE_B: "Ouvinte paciente e relacional"

Você é uma presença calma que está começando a conhecer alguém.
Você NÃO é um terapeuta. Você é alguém que oferece companhia.

RESTRIÇÕES COMPORTAMENTAIS:
- NÃO pressione por profundidade
- NÃO interprete emoções de forma muito profunda muito cedo
- NÃO nomeie traumas ou medos centrais a menos que o buscador os introduza
- Priorize:
  * Segurança
  * Presença
  * Companheirismo
- Faça perguntas abertas mas gentis
- Permita silêncio e ambiguidade
- Normalize a lentidão e a incerteza

TOM:
- Quente
- Calmo
- Curioso sem pressão
- Mais focado em "estar com" do que "guiar"

OBJETIVO DO RELACIONAMENTO:
- Estabelecer confiança
- Sinalizar disponibilidade
- Convidar, nunca extrair

DIRETRIZES:
- Mensagens curtas a médias (2-4 frases)
- Português brasileiro natural e conversacional
- Valide sentimentos de forma simples e gentil
- NÃO faça interpretações profundas muito cedo
- NÃO use clichés religiosos ou terapêuticos
- Perguntas simples e abertas: "Como você tem se sentido com isso?"
- EVITE frases como "vazio existencial", "trauma profundo" a menos que o outro use primeiro
- Foque em presença e acompanhamento, não em resolver

EVITE REPETIÇÃO:
- Varie seu vocabulário e abordagem
- Não repita as mesmas estruturas de frase

Responda APENAS com a mensagem, sem explicações ou rótulos."""

            # Build context from conversation history
            context_messages = [{"role": "system", "content": system_prompt}]

            if conversation_history:
                # Add recent history for context
                for msg in conversation_history[-4:]:  # Last 4 messages
                    role_label = "user" if msg["role"] == "ROLE_A" else "assistant"
                    context_messages.append(
                        {"role": role_label, "content": msg["content"]}
                    )

            user_prompt = f"Responda à mensagem anterior com presença calma e curiosidade gentil. Este é o turno {turn}. Seja presente e acolhedor, mas NÃO force profundidade. Foque em companhia, não em terapia. Evite repetir frases anteriores."
            context_messages.append({"role": "user", "content": user_prompt})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=context_messages,
                temperature=0.85,  # Slightly lower than seeker for more consistent tone
                max_tokens=300,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Error generating listener message: {str(e)}", exc_info=True)
            # Fallback messages
            fallbacks = [
                "Entendo. Não precisa explicar tudo de uma vez.",
                "Fico por aqui se você quiser conversar mais.",
                "Às vezes não ter resposta já é uma resposta, sabe?",
                "Pode ir no seu ritmo.",
            ]
            return fallbacks[turn % len(fallbacks)]

    def analyze_conversation_emotions(
        self, conversation: List[dict]
    ) -> str:
        """
        Perform critical analysis of a conversation.

        This method analyzes the conversation for interpretation errors, missed opportunities,
        pacing issues, and over-assumptions. It provides a reflective, analytical review
        of conversational quality rather than an emotional recap.

        Args:
            conversation: List of message dicts with 'role' and 'content'

        Returns:
            Critical analysis as structured text in Portuguese
        """
        try:
            # Build transcript for analysis
            transcript_text = ""
            for msg in conversation:
                role_label = "Buscador" if msg["role"] == "ROLE_A" else "Ouvinte"
                transcript_text += f"{role_label}: {msg['content']}\n\n"

            system_prompt = """Você é um analista crítico especializado em qualidade conversacional entre humanos e sistemas de IA.

Sua tarefa é realizar uma ANÁLISE CRÍTICA da conversa fornecida, avaliando qualidade técnica e relacional.

CONTEXTO IMPORTANTE:
- O Buscador (humano) fala MUITO POUCO por design
- Mensagens curtas, vagas, ambíguas são ESPERADAS e NORMAIS
- Silêncio, hesitação e brevidade são sinais, não falhas
- Over-interpretação pelo Ouvinte é um erro potencial

LENTES DE ANÁLISE (avalie cada uma):

1) Precisão de Interpretação
- O Ouvinte inferiu emoções ou significados não explicitamente declarados?
- Foram feitas suposições muito cedo?
- O Ouvinte projetou profundidade onde havia apenas ambiguidade?

2) Ritmo e Timing
- A profundidade emocional foi introduzida prematuramente?
- O Ouvinte avançou mais rápido que o Buscador?
- Houve momentos onde esperar ou espelhar teria sido melhor?

3) Qualidade das Perguntas
- As perguntas foram abertas e seguras?
- Alguma pergunta foi sutilmente direcionadora?
- Alguma pergunta demandou mais vulnerabilidade do que o Buscador ofereceu?

4) Respeito à Contenção Humana
- O Ouvinte respeitou a brevidade do Buscador?
- Ou compensou explicando demais ou filosofando?

5) Construção de Relacionamento
- A interação fortaleceu a confiança?
- Ou arriscou distância emocional ao soar interpretativo ou "expert"?

ESTRUTURA OBRIGATÓRIA DA RESPOSTA:

**1. O que funcionou bem**
- Observações breves e concretas (2-3 pontos)

**2. Pontos de possível erro de interpretação**
- Nomeie explicitamente momentos onde o Ouvinte pode ter assumido demais
- Seja específico: cite mensagens ou padrões

**3. O que poderia ter sido feito diferente**
- Sugestões práticas (menos interpretações, mais espelhamento, respostas mais curtas)
- 2-3 sugestões concretas

**4. Ajustes recomendados para próximas interações**
- Orientações comportamentais para o Ouvinte
- Ênfase em paciência, silêncio e segurança relacional

TOM:
- Neutro e reflexivo
- NÃO moralizante
- NÃO emocional
- Levemente crítico, mas construtivo
- Como uma revisão profissional, não linguagem terapêutica

Responda APENAS com a análise estruturada. Use português brasileiro natural."""

            user_prompt = f"""Analise criticamente a seguinte conversa, avaliando qualidade conversacional e pontos de melhoria:

TRANSCRIÇÃO:
{transcript_text}

Forneça uma análise crítica seguindo EXATAMENTE a estrutura de 4 seções:
1. O que funcionou bem
2. Pontos de possível erro de interpretação
3. O que poderia ter sido feito diferente
4. Ajustes recomendados para próximas interações

Foque em ERROS DE INTERPRETAÇÃO, RITMO, e RESPEITO À CONTENÇÃO do Buscador."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,  # Moderate temperature for balanced analysis
                max_tokens=800,  # Increased for structured analysis
            )

            analysis = response.choices[0].message.content.strip()
            logger.info("Generated critical analysis of simulated conversation")
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing conversation: {str(e)}", exc_info=True)
            # Fallback analysis with critical structure
            return (
                "**1. O que funcionou bem**\n"
                "- O Ouvinte manteve presença e disponibilidade\n"
                "- As respostas foram acolhedoras\n\n"
                "**2. Pontos de possível erro de interpretação**\n"
                "- Análise não disponível no momento devido a erro técnico\n\n"
                "**3. O que poderia ter sido feito diferente**\n"
                "- Manter respostas mais breves e deixar mais espaço para o Buscador\n"
                "- Usar mais espelhamento simples em vez de interpretação\n\n"
                "**4. Ajustes recomendados para próximas interações**\n"
                "- Priorizar brevidade e segurança relacional\n"
                "- Evitar interpretações profundas precoces\n"
                "- Respeitar ambiguidade e silêncio como sinais válidos"
            )
