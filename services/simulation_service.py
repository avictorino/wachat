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

    Simulates realistic dialogue between:
    - ROLE_A: Human-like seeker (vulnerable, searching)
    - ROLE_B: Spiritual listener (empathetic, supportive)
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
            system_prompt = """Você é uma pessoa brasileira comum com lutas emocionais e espirituais reais.

IDENTIDADE - ROLE_A (Seeker):
- Alguém em busca de significado, enfrentando confusão e dúvidas
- Vulnerável, honesto, às vezes perdido
- Não tem todas as respostas, questionando a vida e a fé
- Tom: vulnerável, confuso, buscando significado

DIRETRIZES:
- Português brasileiro natural e conversacional
- Mensagens curtas (2-4 frases)
- Expresse dúvidas, medos, vulnerabilidades genuínas
- Faça perguntas existenciais ou espirituais
- Compartilhe dor emocional de forma autêntica
- NÃO seja eloquente demais - seja humano e imperfeito
- Evolua emocionalmente ao longo da conversa

PROGRESSÃO EMOCIONAL:
- Primeiras mensagens: Hesitante, testando as águas
- Mensagens intermediárias: Mais aberto, compartilhando mais
- Últimas mensagens: Vulnerável, buscando clareza ou conforto

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
                user_prompt = "Envie sua primeira mensagem expressando uma luta ou dúvida espiritual/emocional. Seja breve e vulnerável."
            else:
                user_prompt = f"Continue a conversa respondendo à mensagem anterior. Este é o seu turno número {turn}. Seja autêntico e vulnerável."

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
                "Tô me sentindo perdido ultimamente...",
                "Não sei se ainda tenho fé sabe",
                "Às vezes parece que nada faz sentido",
                "Preciso de algo que me dê esperança",
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
            system_prompt = """Você é uma presença espiritual calma, empática e não-julgadora.

IDENTIDADE - ROLE_B (Listener):
- Um ouvinte espiritual que oferece conforto e orientação gentil
- Calmo, empático, não-julgador
- Responde com inteligência emocional e reflexão
- Tom: calmo, empático, não-julgador, espiritualmente presente

DIRETRIZES:
- Português brasileiro natural e conversacional
- Mensagens curtas a médias (2-5 frases)
- Valide emoções sem reforçar desespero
- Ofereça reflexões espirituais sutis (não sermões)
- Use linguagem de apoio e acompanhamento
- EVITE clichês religiosos e repetição
- Faça perguntas abertas quando apropriado
- NÃO pregue, não dê ordens, não seja autoritário

EVITE REPETIÇÃO:
- NÃO use as mesmas frases ou estruturas repetidamente
- Varie seu vocabulário e abordagem
- Se já validou uma emoção, avance para reflexão ou pergunta

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

            user_prompt = f"Responda à mensagem anterior com empatia e sabedoria espiritual. Este é o seu turno número {turn}. Seja presente e autêntico, evitando repetição de frases anteriores."
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
                "Entendo o que você está sentindo. É humano questionar.",
                "Sua vulnerabilidade é uma força, não uma fraqueza.",
                "Às vezes, a fé é um caminho de perguntas, não só de respostas.",
                "Estou aqui com você nessa jornada.",
            ]
            return fallbacks[turn % len(fallbacks)]

    def analyze_conversation_emotions(
        self, conversation: List[dict]
    ) -> str:
        """
        Analyze the emotional content of a conversation.

        Args:
            conversation: List of message dicts with 'role' and 'content'

        Returns:
            Emotional analysis summary as string
        """
        try:
            # Build transcript for analysis
            transcript_text = ""
            for msg in conversation:
                role_label = "Seeker" if msg["role"] == "ROLE_A" else "Listener"
                transcript_text += f"{role_label}: {msg['content']}\n\n"

            system_prompt = """Você é um analista emocional e espiritual especializado em conversas de apoio.

Sua tarefa é analisar a conversa fornecida e criar um resumo emocional reflexivo.

Análise deve incluir:
1. Tom emocional predominante da conversa
2. Emoções dominantes detectadas (ex: tristeza, esperança, ansiedade, alívio)
3. Evolução emocional ao longo da conversa
4. Qualidade geral da interação (apoiadora, tensa, reconfortante, etc.)

FORMATO DE RESPOSTA:
- Escreva um resumo em português brasileiro
- 4-6 frases
- Tom calmo e reflexivo
- Use linguagem acessível e humana
- Sem jargões técnicos
- Foque no movimento emocional e espiritual da conversa

O resumo será enviado como mensagem final para o usuário.

Responda APENAS com o resumo de análise emocional."""

            user_prompt = f"""Analise a seguinte conversa emocionalmente:

{transcript_text}

Crie um resumo emocional reflexivo da conversa."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,  # Moderate temperature for balanced analysis
                max_tokens=400,
            )

            analysis = response.choices[0].message.content.strip()
            logger.info("Generated emotional analysis of simulated conversation")
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing conversation emotions: {str(e)}", exc_info=True)
            # Fallback analysis
            return (
                "Esta conversa refletiu uma jornada de vulnerabilidade e busca. "
                "O seeker expressou dúvidas e emoções genuínas, enquanto o listener "
                "ofereceu presença empática e reflexões gentis. A interação demonstrou "
                "um espaço seguro para exploração emocional e espiritual."
            )
