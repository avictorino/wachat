"""
Service for simulating realistic human messages in conversations.

This module uses AI to generate emotionally-driven, imperfect human messages
that reflect real spiritual and life struggles.
"""

import logging
import random
from typing import List

from groq import Groq

logger = logging.getLogger(__name__)


class HumanSimulator:
    """
    Simulates a human user with emotional and spiritual needs.

    Uses AI to generate realistic, imperfect messages that reflect genuine
    human struggles with faith, doubt, relationships, and life challenges.
    """

    def __init__(self, api_key: str, name: str, domain: str = "spiritual"):
        """
        Initialize the human simulator.

        Args:
            api_key: Groq API key
            name: The simulated person's name
            domain: The domain of conversation (e.g., 'spiritual', 'relationship', 'grief')
        """
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
        self.name = name
        self.domain = domain
        self.conversation_state = {
            "turn": 0,
            "emotional_state": "seeking",  # seeking, opening_up, vulnerable, reflective
            "topics_mentioned": [],
        }

    def generate_message(self, conversation_history: List[dict], turn_number: int) -> str:
        """
        Generate a realistic human message based on conversation context.

        Args:
            conversation_history: List of previous messages with 'role' and 'content'
            turn_number: Current turn number in the conversation

        Returns:
            A realistic human message as string
        """
        try:
            self.conversation_state["turn"] = turn_number

            # Build context for the AI
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(conversation_history, turn_number)

            # Generate message
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.9,  # High temperature for more variability
                max_tokens=200,
            )

            message = response.choices[0].message.content.strip()

            # Update emotional state based on turn
            self._update_emotional_state(turn_number)

            logger.info(
                f"Generated simulated human message (turn {turn_number}): {message[:50]}..."
            )
            return message

        except Exception as e:
            logger.error(f"Error generating simulated human message: {str(e)}", exc_info=True)
            # Fallback to a basic message
            return self._get_fallback_message(turn_number)

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the AI."""
        emotional_state = self.conversation_state["emotional_state"]

        base_prompt = f"""Você está simulando uma pessoa brasileira real chamada {self.name}
que está em busca de conforto espiritual e emocional. Esta pessoa:

- Escreve mensagens CURTAS (1-3 frases no máximo)
- Usa linguagem informal e imperfeita (pode ter erros de digitação ocasionais)
- Mostra emoções reais: dúvida, medo, esperança, vulnerabilidade
- Fala sobre {self.domain} de forma genuína e pessoal
- Não é eloquente ou poética - é uma pessoa comum com problemas reais
- Às vezes interrompe o pensamento ou muda de assunto
- Pode ser repetitiva quando está ansiosa
- Responde aos pontos do assistente, mas com suas próprias preocupações

Estado emocional atual: {emotional_state}
"""

        # Add state-specific instructions
        if emotional_state == "seeking":
            base_prompt += "\n- Está tentando se abrir, mas ainda hesitante"
        elif emotional_state == "opening_up":
            base_prompt += "\n- Está começando a compartilhar mais detalhes"
        elif emotional_state == "vulnerable":
            base_prompt += "\n- Está mais aberto e vulnerável, compartilhando mais profundamente"
        elif emotional_state == "reflective":
            base_prompt += "\n- Está refletindo sobre as conversas anteriores e processando"

        base_prompt += "\n\nResponda APENAS com a mensagem que esta pessoa enviaria. Não adicione explicações, aspas ou contexto."

        return base_prompt

    def _build_user_prompt(self, conversation_history: List[dict], turn_number: int) -> str:
        """Build the user prompt with conversation context."""
        if turn_number == 1:
            # First message - initiate conversation
            prompts = [
                f"Envie a primeira mensagem iniciando uma conversa sobre {self.domain}. "
                "Seja breve, vulnerável e real.",
                f"Comece uma conversa expressando uma preocupação ou dúvida sobre {self.domain}. "
                "Mantenha curto e humano.",
            ]
            return random.choice(prompts)

        # Subsequent messages - respond to conversation
        context = "Histórico da conversa:\n"
        for msg in conversation_history[-4:]:  # Last 4 messages for context
            role_label = "Eu" if msg["role"] == "user" else "Assistente"
            context += f"{role_label}: {msg['content']}\n"

        context += f"\nEscreva a próxima mensagem que {self.name} enviaria. "
        context += "Responda ao que o assistente disse, mas mantenha suas próprias preocupações e emoções. "
        context += "Seja breve (1-3 frases)."

        return context

    def _update_emotional_state(self, turn_number: int):
        """Update emotional state based on conversation progress."""
        if turn_number <= 2:
            self.conversation_state["emotional_state"] = "seeking"
        elif turn_number <= 4:
            self.conversation_state["emotional_state"] = "opening_up"
        elif turn_number <= 6:
            self.conversation_state["emotional_state"] = "vulnerable"
        else:
            self.conversation_state["emotional_state"] = "reflective"

    def _get_fallback_message(self, turn_number: int) -> str:
        """Get a fallback message if AI generation fails."""
        fallback_messages = [
            "Tô confuso sobre algumas coisas...",
            "Será que isso faz sentido?",
            "Às vezes me sinto perdido sabe",
            "Obrigado por me ouvir",
            "Isso me ajuda a pensar melhor",
        ]

        if turn_number == 1:
            return "Oi, preciso conversar sobre umas coisas que tão me incomodando"

        return random.choice(fallback_messages)
