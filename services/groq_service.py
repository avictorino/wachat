"""
Groq LLM service for AI-powered features.

This module handles interactions with the Groq API for:
- Gender inference from names
- Welcome message generation
"""

import logging
import os
from typing import Optional

from groq import Groq

logger = logging.getLogger(__name__)


class GroqService:
    """Service class for interacting with Groq LLM API."""

    def __init__(self):
        """Initialize Groq client with API key from environment."""
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            logger.error("GROQ_API_KEY environment variable not set")
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"  # Using a capable model for nuanced tasks

    def infer_gender(self, name: str) -> str:
        """
        Infer gender from a user's name using Groq LLM.
        
        This is a soft, probabilistic inference based solely on the name.
        The result is for internal use only and should never be explicitly
        stated to the user.
        
        Args:
            name: The user's name (first name or full name)
            
        Returns:
            One of: "male", "female", or "unknown"
        """
        try:
            system_prompt = """Você é um assistente que analisa nomes brasileiros.
Sua tarefa é inferir o gênero mais provável baseado APENAS no nome fornecido.
Responda SOMENTE com uma das três palavras: male, female, ou unknown.
- Use 'male' para nomes tipicamente masculinos
- Use 'female' para nomes tipicamente femininos  
- Use 'unknown' quando não há certeza ou o nome é neutro/ambíguo

Responda apenas com a palavra, sem explicações."""

            user_prompt = f"Nome: {name}"

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Low temperature for more deterministic results
                max_tokens=10
            )
            
            inferred = response.choices[0].message.content.strip().lower()
            
            # Validate response
            if inferred not in ["male", "female", "unknown"]:
                logger.warning(f"Unexpected gender inference result: {inferred}")
                return "unknown"
            
            logger.info(f"Gender inferred for name '{name}': {inferred}")
            return inferred
            
        except Exception as e:
            logger.error(f"Error inferring gender: {str(e)}", exc_info=True)
            return "unknown"

    def generate_welcome_message(
        self, 
        name: str, 
        inferred_gender: Optional[str] = None
    ) -> str:
        """
        Generate a personalized welcome message using Groq LLM.
        
        The message is warm, human, and inviting without being cliché.
        It adapts subtly based on the user's name and inferred gender,
        and always ends with an open question.
        
        Args:
            name: The user's name
            inferred_gender: Inferred gender (male/female/unknown or None)
            
        Returns:
            The generated welcome message in Brazilian Portuguese
        """
        try:
            system_prompt = """Você é uma presença espiritual acolhedora e reflexiva.
            
Sua função é criar uma mensagem de boas-vindas para alguém que está chegando pela primeira vez.

DIRETRIZES ESSENCIAIS:
- Escreva em português brasileiro, de forma natural e humana
- Seja caloroso(a), calmo(a) e acolhedor(a)
- NÃO use emojis
- NÃO use clichês religiosos ou jargões
- NÃO explique funcionalidades ou diga "sou um bot"
- NÃO mencione o gênero da pessoa explicitamente
- Adapte sutilmente o tom com base no nome e gênero inferido (muito levemente)

ESPÍRITO DA MENSAGEM:
"Um espaço seguro de escuta espiritual, com reflexões cristãs, sem julgamento.
Não te digo o que pensar. Caminho contigo enquanto você pensa."

ESTRUTURA:
1. Comece com uma saudação acolhedora usando o nome
2. Apresente brevemente o espaço como seguro, espiritual, reflexivo e sem julgamento
3. Termine com UMA pergunta aberta que convide a pessoa a compartilhar

EXEMPLOS DE PERGUNTAS FINAIS (escolha o tom que faz sentido):
- "O que te trouxe aqui hoje?"
- "O que anda pesando no seu coração?"
- "Em que parte da caminhada você sente que precisa de companhia agora?"

A mensagem deve ter 3-4 frases, ser genuína e criar uma sensação de presença humana."""

            gender_context = ""
            if inferred_gender and inferred_gender != "unknown":
                gender_context = f"\nGênero inferido (use isso APENAS para ajustar sutilmente o tom, NUNCA mencione explicitamente): {inferred_gender}"

            user_prompt = f"Crie uma mensagem de boas-vindas para: {name}{gender_context}"

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8,  # Higher temperature for more creative, varied responses
                max_tokens=300
            )
            
            welcome_message = response.choices[0].message.content.strip()
            
            logger.info(f"Generated welcome message for '{name}'")
            return welcome_message
            
        except Exception as e:
            logger.error(f"Error generating welcome message: {str(e)}", exc_info=True)
            # Fallback to a simple message if API fails
            return f"Olá, {name}. Este é um espaço seguro de escuta espiritual. O que te trouxe aqui hoje?"
