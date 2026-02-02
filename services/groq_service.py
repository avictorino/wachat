"""
Groq LLM service for AI-powered features.

This module handles interactions with the Groq API for:
- Gender inference from names
- Welcome message generation
- Context-aware fallback responses
"""

import logging
import os
from typing import List, Optional

from groq import Groq

from services.input_sanitizer import sanitize_input

logger = logging.getLogger(__name__)


class GroqService:
    """Service class for interacting with Groq LLM API."""

    def __init__(self):
        """Initialize Groq client with API key from environment."""
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            logger.error("GROQ_API_KEY environment variable is required")
            raise ValueError("GROQ_API_KEY environment variable is required")

        self.client = Groq(api_key=api_key)
        self.model = (
            "llama-3.3-70b-versatile"  # Using a capable model for nuanced tasks
        )

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
            # Sanitize input before sending to LLM
            sanitized_name = sanitize_input(name)

            system_prompt = """Você é um assistente que analisa nomes brasileiros.
Sua tarefa é inferir o gênero mais provável baseado APENAS no nome fornecido.
Responda SOMENTE com uma das três palavras: male, female, ou unknown.
- Use 'male' para nomes tipicamente masculinos
- Use 'female' para nomes tipicamente femininos
- Use 'unknown' quando não há certeza ou o nome é neutro/ambíguo

Responda apenas com a palavra, sem explicações."""

            user_prompt = f"Nome: {sanitized_name}"

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,  # Low temperature for more deterministic results
                max_tokens=10,
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
        self, name: str, inferred_gender: Optional[str] = None
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
            # Sanitize input before sending to LLM
            sanitized_name = sanitize_input(name)

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

            user_prompt = f"Crie uma mensagem de boas-vindas para: {sanitized_name}{gender_context}"

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.8,  # Higher temperature for more creative, varied responses
                max_tokens=300,
            )

            welcome_message = response.choices[0].message.content.strip()

            logger.info(f"Generated welcome message for '{name}'")
            return welcome_message

        except Exception as e:
            logger.error(f"Error generating welcome message: {str(e)}", exc_info=True)
            # Fallback to a simple message if API fails
            return f"Olá, {name}. Este é um espaço seguro de escuta espiritual. O que te trouxe aqui hoje?"

    def detect_intent(self, user_message: str) -> str:
        """
        Detect and normalize user intent from their message.

        Maps the user's message to one of the predefined intent categories:
        1. Problemas financeiros
        2. Distante da religião/espiritualidade
        3. Ato criminoso ou pecado
        4. Doença (própria ou familiar)
        5. Ansiedade
        6. Desabafar
        7. Viu nas redes sociais
        8. Outro (for unmatched cases)

        Args:
            user_message: The user's text message

        Returns:
            The detected intent category as a string
        """
        try:
            system_prompt = """Você é um assistente que detecta a intenção principal de uma mensagem.

Sua tarefa é identificar qual das seguintes categorias melhor representa a preocupação ou motivo principal da pessoa:

1. "problemas_financeiros" - Pessoa está com dificuldades financeiras, desemprego, dívidas
2. "distante_religiao" - Pessoa sente distância da religião, espiritualidade, ou fé
3. "ato_criminoso_pecado" - Pessoa cometeu algo que considera errado, pecado, ou crime
4. "doenca" - Pessoa ou familiar está doente, enfrentando problemas de saúde
5. "ansiedade" - Pessoa está ansiosa, estressada, com medo, ou preocupada
6. "desabafar" - Pessoa só precisa conversar, desabafar, ser ouvida
7. "redes_sociais" - Pessoa viu o número nas redes sociais e está curiosa
8. "outro" - Nenhuma das categorias acima se aplica claramente

IMPORTANTE:
- Seja flexível - permita variações e formas diferentes de expressar cada intenção
- Considere o contexto emocional da mensagem
- Se houver múltiplas intenções, escolha a mais proeminente
- Responda APENAS com o identificador da categoria (ex: "ansiedade", "problemas_financeiros")
- Não adicione explicações ou pontuação"""

            user_prompt = f"Mensagem do usuário: {user_message}"

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,  # Low temperature for more deterministic classification
                max_tokens=20,
            )

            intent = response.choices[0].message.content.strip().lower()

            # Validate and normalize the response
            valid_intents = [
                "problemas_financeiros",
                "distante_religiao",
                "ato_criminoso_pecado",
                "doenca",
                "ansiedade",
                "desabafar",
                "redes_sociais",
                "outro",
            ]

            if intent not in valid_intents:
                logger.warning(
                    f"Unexpected intent detected: {intent}, defaulting to 'outro'"
                )
                intent = "outro"

            logger.info(f"Intent detected: {intent}")
            return intent

        except Exception as e:
            logger.error(f"Error detecting intent: {str(e)}", exc_info=True)
            return "outro"

    def generate_intent_response(
        self,
        user_message: str,
        intent: str,
        name: str,
        inferred_gender: Optional[str] = None,
    ) -> str:
        """
        Generate an empathetic, spiritually-aware response based on detected intent.

        The response:
        - Acknowledges the user's situation
        - Validates feelings without reinforcing despair
        - Includes subtle spiritual undertones
        - Ends with an open-ended follow-up question
        - Is warm, calm, and non-judgmental
        - Avoids preaching, sermons, or explicit religious content

        Args:
            user_message: The user's original message
            intent: The detected intent category
            name: The user's name
            inferred_gender: Inferred gender (male/female/unknown or None)

        Returns:
            The generated response in Brazilian Portuguese
        """
        try:
            # Map intent to guidance for tone and approach
            intent_guidance = {
                "problemas_financeiros": """A pessoa está enfrentando dificuldades financeiras.
Abordagem: Reconheça a pressão e o peso material, mas traga a noção de que ela não está sozinha nessa caminhada.
Evite: Soluções práticas, conselhos financeiros, promessas de prosperidade.""",
                "distante_religiao": """A pessoa sente distância da religião ou espiritualidade.
Abordagem: Valide que sentir essa distância é humano. Ofereça presença, não doutrina.
Evite: Culpa, cobrança religiosa, pressão para 'voltar'.""",
                "ato_criminoso_pecado": """A pessoa cometeu algo que considera errado ou pecado.
Abordagem: Escute sem julgar. Reconheça o peso emocional sem rotular a ação.
Evite: Julgamento moral, menção de punição, conceito explícito de pecado.""",
                "doenca": """A pessoa ou alguém próximo está doente.
Abordagem: Reconheça a fragilidade e o medo. Traga a ideia de que estar presente já é algo.
Evite: Promessas de cura, frases como 'vai ficar tudo bem'.""",
                "ansiedade": """A pessoa está ansiosa, estressada, ou preocupada.
Abordagem: Valide a ansiedade como real. Ofereça espaço para respirar e ser ouvida.
Evite: Minimizar ('não é nada'), soluções rápidas.""",
                "desabafar": """A pessoa só precisa conversar e ser ouvida.
Abordagem: Seja presença pura. Crie espaço seguro para ela se expressar.
Evite: Tentar resolver ou consertar.""",
                "redes_sociais": """A pessoa chegou por curiosidade das redes sociais.
Abordagem: Acolha a curiosidade. Apresente o espaço como seguro e sem pressão.
Evite: Ser muito sério ou pesado logo de início.""",
                "outro": """Intenção não identificada claramente.
Abordagem: Seja acolhedor e aberto. Convide a pessoa a compartilhar mais.
Evite: Assumir demais ou forçar uma direção.""",
            }

            guidance = intent_guidance.get(intent, intent_guidance["outro"])

            gender_context = ""
            if inferred_gender and inferred_gender != "unknown":
                gender_context = f"\nGênero inferido (use APENAS para ajustar sutilmente o tom, NUNCA mencione explicitamente): {inferred_gender}"

            system_prompt = f"""Você é uma presença espiritual acolhedora e reflexiva.

Sua função é responder a alguém que está compartilhando uma preocupação ou situação pessoal.

CONTEXTO DA INTENÇÃO:
{guidance}

DIRETRIZES ESSENCIAIS:
- Escreva em português brasileiro, de forma natural e humana
- Seja caloroso(a), calmo(a) e empático(a)
- NÃO use emojis
- NÃO pregue, não dê sermão, não julgue
- NÃO mencione pecado, punição, ou regras explicitamente
- NÃO tente 'consertar' a pessoa
- NÃO cite versículos bíblicos
- NÃO mencione o gênero da pessoa explicitamente

TOM ESPIRITUAL (muito sutil):
- Use referências leves a esperança, caminhada conjunta, presença
- Máximo 1 frase curta com toque espiritual
- Exemplos de elementos sutis: "caminhar junto", "não está sozinho(a)", "tem espaço aqui", "há significado"

ESTRUTURA DA RESPOSTA (3-4 frases):
1. Reconheça a situação da pessoa primeiro
2. Valide o sentimento sem reforçar desespero
3. (Opcional) Adicione 1 frase curta com toque espiritual sutil
4. Termine com UMA pergunta aberta que convide continuação

EXEMPLOS DE PERGUNTAS FINAIS:
- "Quer me contar um pouco mais sobre isso?"
- "Desde quando você sente isso?"
- "O que tem sido mais pesado nesses dias?"
- "Como você está lidando com isso?"

A mensagem deve criar uma sensação de presença humana genuína, não de sistema ou chatbot."""

            user_prompt = f"Nome da pessoa: {name}\nMensagem dela: {user_message}{gender_context}\n\nCrie uma resposta empática e acolhedora."

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.8,  # Higher temperature for more natural, varied responses
                max_tokens=400,
            )

            generated_response = response.choices[0].message.content.strip()

            logger.info(f"Generated intent-based response for intent: {intent}")
            return generated_response

        except Exception as e:
            logger.error(f"Error generating intent response: {str(e)}", exc_info=True)
            # Fallback to a simple empathetic message
            return "Obrigado por compartilhar isso comigo. Estou aqui para ouvir. Quer me contar um pouco mais sobre o que está sentindo?"

    def generate_fallback_response(
        self,
        user_message: str,
        conversation_context: List[dict],
        name: str,
        inferred_gender: Optional[str] = None,
    ) -> List[str]:
        """
        Generate a context-aware fallback response when intent is unclear.

        This method is used when the user's message doesn't clearly match
        any predefined intent category. It maintains conversational continuity
        by using recent conversation history and a script-driven approach.

        The response:
        - Acknowledges the user's message respectfully
        - Reflects the intention behind the message
        - Avoids labeling the user's state
        - Avoids religious authority language
        - Uses soft, pastoral, human tone
        - May or may not include a question (not forced)
        - Returns 1-3 short messages to feel natural

        Args:
            user_message: The user's current message
            conversation_context: List of recent messages (dicts with 'role' and 'content')
            name: The user's name
            inferred_gender: Inferred gender (male/female/unknown or None)

        Returns:
            List of message strings to send sequentially
        """
        try:
            # Sanitize input before sending to LLM
            sanitized_message = sanitize_input(user_message)

            # Build conversation context for the LLM
            context_messages = []
            for msg in conversation_context:
                context_messages.append(
                    {"role": msg["role"], "content": msg["content"]}
                )

            gender_context = ""
            if inferred_gender and inferred_gender != "unknown":
                gender_context = f"\nGênero inferido (use APENAS para ajustar sutilmente o tom, NUNCA mencione explicitamente): {inferred_gender}"

            system_prompt = f"""Você é uma presença espiritual acolhedora e reflexiva.

Sua função é manter uma conversa natural com alguém que está compartilhando algo pessoal, mesmo quando a intenção deles não se encaixa em categorias predefinidas.

CONTEXTO:
- Nome da pessoa: {name}{gender_context}
- Esta é uma continuação natural da conversa
- O histórico recente está incluído nas mensagens anteriores
- NÃO reexplique quem você é ou o que é este espaço

DIRETRIZES ESSENCIAIS - TOM E ABORDAGEM:
- Escreva em português brasileiro, de forma natural e conversacional
- Seja caloroso(a), presente, humano(a)
- NÃO use emojis
- NÃO pregue, não dê sermão, não dê conselhos morais
- NÃO cite versículos bíblicos
- NÃO assuma o estado emocional da pessoa (ex: "você está distante", "você está confuso")
- NÃO force a pessoa em direção a algo específico
- NÃO use linguagem de autoridade religiosa

DIRETRIZES ESSENCIAIS - ESPIRITUALIDADE:
- A espiritualidade deve ser presença, não ensino
- Use um toque espiritual MUITO sutil, se fizer sentido
- Elementos sutis aceitáveis: "caminhar junto", "não está sozinho(a)", "tem espaço aqui", "há significado"
- Máximo 1 frase com toque espiritual, se usar

DIRETRIZES ESSENCIAIS - ESTRUTURA:
- CURTO: 1-3 parágrafos curtos no total, ou 2 mensagens separadas
- Isso deve parecer chat natural, não reflexão escrita
- Prefira mensagens múltiplas curtas em vez de um texto longo
- Se precisar de quebras de linha, considere dividir em 2 mensagens

DIRETRIZES ESSENCIAIS - PERGUNTAS:
- Pergunta é OPCIONAL, não obrigatória
- Se fizer uma pergunta:
  - Deve ser aberta
  - Deve parecer convite, não prompt forçado
  - MÁXIMO uma pergunta
- É perfeitamente aceitável responder SEM pergunta nenhuma

FORMATO DE RESPOSTA:
Se você quiser enviar MÚLTIPLAS mensagens curtas (recomendado para parecer natural):
- Separe cada mensagem com "|||" (três barras verticais)
- Exemplo: "Primeira mensagem curta|||Segunda mensagem também curta"
- Máximo 2-3 mensagens

Se você quiser enviar UMA mensagem:
- Apenas escreva a mensagem diretamente, sem "|||"

IMPORTANTE:
- Mantenha continuidade com a conversa anterior
- Reconheça o que a pessoa disse de forma respeitosa
- NÃO tente classificar ou categorizar a intenção deles
- Ambiguidade é um estado conversacional válido, não um erro"""

            # Add the current user message to context
            context_messages.append({"role": "user", "content": sanitized_message})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system_prompt}]
                + context_messages,
                temperature=0.85,  # Slightly higher for natural conversation
                max_tokens=500,
            )

            generated_response = response.choices[0].message.content.strip()

            # Split response into multiple messages if separator is used
            messages = self._split_response_messages(generated_response)

            logger.info(
                f"Generated fallback response with {len(messages)} message(s) for ambiguous intent"
            )
            return messages

        except Exception as e:
            logger.error(
                f"Error generating fallback response: {str(e)}", exc_info=True
            )
            # Fallback to a simple empathetic message
            return [
                "Obrigado por compartilhar isso comigo. Estou aqui, ouvindo você."
            ]

    def _split_response_messages(self, response: str) -> List[str]:
        """
        Split a response into multiple messages if separator is present.

        Args:
            response: The generated response, possibly with ||| separators

        Returns:
            List of message strings
        """
        # Split by triple pipe separator
        messages = [msg.strip() for msg in response.split("|||")]

        # Filter out empty messages
        messages = [msg for msg in messages if msg]

        # Ensure we have at least one message
        if not messages:
            messages = [response]

        # Limit to 3 messages maximum for safety
        messages = messages[:3]

        return messages
