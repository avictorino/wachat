"""
Ollama LLM service for AI-powered features.

This module handles interactions with a local Ollama server for:
- Gender inference from names
- Welcome message generation
- Context-aware fallback responses with RAG integration
"""

import logging
import os
from typing import Any, Dict, List, Optional

import requests

from services.input_sanitizer import sanitize_input
from services.llm_service_interface import LLMServiceInterface
from services.rag_service import get_rag_context

logger = logging.getLogger(__name__)


# Helper constant for gender context in Portuguese
# This instruction is in Portuguese because it's part of the system prompt
# sent to the LLM, which operates in Brazilian Portuguese
GENDER_CONTEXT_INSTRUCTION = (
    "Gênero inferido (use APENAS para ajustar sutilmente o tom, "
    "NUNCA mencione explicitamente): {gender}"
)


class OllamaService(LLMServiceInterface):
    """Service class for interacting with local Ollama LLM API."""

    def __init__(self):
        """Initialize Ollama client with configuration from environment."""
        self.base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.environ.get("OLLAMA_MODEL", "llama3.1")
        self.api_url = f"{self.base_url}/api/chat"

        logger.info(
            f"Initialized OllamaService with base_url={self.base_url}, model={self.model}"
        )

    def _make_chat_request(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """
        Make a chat completion request to Ollama API.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate (optional, no enforcement)
            **kwargs: Additional Ollama options (top_p, repeat_penalty, etc.)

        Returns:
            The assistant's response text

        Raises:
            requests.exceptions.RequestException: On connection or API errors
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                **kwargs,
            },
        }

        # Add max_tokens if provided (Ollama calls it num_predict)
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=60,  # 60 second timeout for local requests
            )
            response.raise_for_status()

            response_data = response.json()
            content = response_data.get("message", {}).get("content", "").strip()

            if not content:
                logger.warning("Ollama returned empty content")
                raise ValueError("Empty response from Ollama")

            return content

        except requests.exceptions.ConnectionError as e:
            logger.error(f"Failed to connect to Ollama at {self.base_url}: {str(e)}")
            raise
        except requests.exceptions.Timeout as e:
            logger.error(f"Ollama request timed out: {str(e)}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {str(e)}")
            raise
        except (KeyError, ValueError) as e:
            logger.error(f"Failed to parse Ollama response: {str(e)}")
            raise

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

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response_text = self._make_chat_request(
                messages, temperature=0.3, max_tokens=10
            )

            inferred = response_text.strip().lower()

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
        Generate a personalized welcome message using Ollama LLM.

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

ESPÍRITO DO ESPAÇO:
"Um espaço seguro de escuta espiritual, com reflexões cristãs, sem julgamento.
Não te digo o que pensar. Caminho contigo enquanto você pensa."

DIRETRIZES:
- Português brasileiro, natural e humano
- Caloroso, calmo, acolhedor
- NÃO use emojis
- NÃO use clichês religiosos ou jargões
- NÃO explique funcionalidades ou diga "sou um bot"
- NÃO mencione gênero explicitamente
- Adapte sutilmente o tom baseado no nome (muito levemente)

ESTRUTURA (3-4 frases):
1. Saudação acolhedora usando o nome
2. Apresente o espaço (seguro, espiritual, reflexivo, sem julgamento)
3. UMA pergunta aberta que convide a compartilhar

EXEMPLOS DE PERGUNTAS (escolha tom apropriado):
- "O que te trouxe aqui hoje?"
- "O que anda pesando no seu coração?"
- "Em que parte da caminhada você sente que precisa de companhia agora?"

Crie sensação de presença humana genuína."""

            gender_context = ""
            if inferred_gender and inferred_gender != "unknown":
                gender_context = f"\nGênero inferido (use isso APENAS para ajustar sutilmente o tom, NUNCA mencione explicitamente): {inferred_gender}"

            user_prompt = (
                f"Crie uma mensagem de boas-vindas para: {sanitized_name}{gender_context}"
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            welcome_message = self._make_chat_request(
                messages, temperature=0.8, max_tokens=300
            )

            logger.info(f"Generated welcome message for '{name}'")
            return welcome_message

        except Exception as e:
            logger.error(f"Error generating welcome message: {str(e)}", exc_info=True)
            # Fallback to a simple message if API fails
            return f"Olá, {name}. Este é um espaço seguro de escuta espiritual. O que te trouxe aqui hoje?"

    def detect_intent(self, user_message: str) -> str:
        """
        Detect and normalize user intent from their message.

        Maps the user's message to one of the predefined intent categories.

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
8. "drogas" - Pessoa está lutando com uso de drogas, substâncias, dependência química
9. "alcool" - Pessoa está lutando com uso de álcool, bebida, dependência alcoólica
10. "sexo" - Pessoa está lutando com compulsão sexual, comportamento sexual compulsivo
11. "cigarro" - Pessoa está lutando com cigarro, tabagismo, nicotina
12. "outro" - Nenhuma das categorias acima se aplica claramente
13. "luto" – Pessoa perdeu alguém importante (morte de familiar, amigo, cônjuge)
14. "separacao_divorcio" – Término de relacionamento, divórcio ou crise conjugal
15. "solidao" – Pessoa se sente sozinha, sem apoio emocional ou social
16. "culpa_vergonha" – Sentimento intenso de culpa, vergonha ou arrependimento
17. "sentido_da_vida" – Busca por propósito, significado ou direção para a vida
18. "medo_do_futuro" – Insegurança com o futuro, decisões importantes ou mudanças
19. "crise_existencial" – Questionamentos profundos sobre existência, morte, fé ou Deus
20. "familia" – Conflitos familiares (pais, filhos, irmãos)
21. "filhos" – Dificuldades na criação dos filhos, culpa parental, medo de errar
22. "casamento" – Problemas conjugais, rotina, traição, esfriamento emocional
23. "trabalho" – Burnout, pressão profissional, conflitos no trabalho, falta de sentido na carreira
24. "depressao" – Tristeza persistente, desânimo, sensação de vazio
25. "perda_material" – Perda de bens, falência, prejuízo financeiro relevante
26. "trauma" – Experiências traumáticas passadas (violência, abuso, acidentes)
27. "busca_de_perdao" – Desejo de perdão divino ou de perdoar alguém
28. "agradecimento" – Pessoa quer agradecer por algo que deu certo
29. "milagre_intervencao" – Busca por ajuda sobrenatural, milagre ou intervenção divina
30. "rotina_devocional" – Pessoa já é religiosa e busca oração, leitura ou reflexão diária
31. "curiosidade_espiritual" – Interesse intelectual ou cultural sobre fé e espiritualidade
32. "conversao" – Interesse em se aproximar ou retornar à religião
33. "pressao_social_familiar" – Influência de família, amigos ou comunidade religiosa
34. "valores_morais" – Busca por orientação ética, certo e errado
35. "esperanca" – Necessidade de esperança em um momento difícil
36. "medo_da_morte" – Medo de morrer ou de perder alguém
37. "agravamento_crise" – Vários problemas acumulados ao mesmo tempo
38. "orientacao_decisao" – Busca por direção antes de uma decisão importante
39. "paz_interior" – Desejo de calma, equilíbrio emocional e espiritual
40. "outro" – Motivo não identificado ou combinação complexa de fatores

IMPORTANTE:
- Seja flexível - permita variações e formas diferentes de expressar cada intenção
- Considere o contexto emocional da mensagem
- Se houver múltiplas intenções, escolha a mais proeminente
- ATENÇÃO: Trate drogas, alcool, sexo, cigarro como condições reais e sérias, não como escolhas ou fraquezas
- Responda APENAS com o identificador da categoria (ex: "ansiedade", "problemas_financeiros", "drogas")
- Não adicione explicações ou pontuação"""

            user_prompt = f"Mensagem do usuário: {user_message}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response_text = self._make_chat_request(
                messages, temperature=0.3, max_tokens=20
            )

            intent = response_text.strip().lower()

            logger.info(f"Intent detected: {intent}")
            return intent

        except Exception as e:
            logger.error(f"Error detecting intent: {str(e)}", exc_info=True)
            return "outro"

    def approximate_theme(self, user_input: str) -> str:
        """
        Approximate user input to one of the predefined theme categories using LLM.

        This method takes a user's theme input (which may be in natural language,
        synonyms, or variations) and maps it to the closest predefined theme category.

        Args:
            user_input: The user's theme input (e.g., "enfermidade", "pecado")

        Returns:
            The approximated theme category as a string (one of the valid themes)
        """
        try:
            # Sanitize input before sending to LLM
            sanitized_input = sanitize_input(user_input)

            system_prompt = """Você é um assistente que mapeia palavras-chave para categorias de temas predefinidas.

Sua tarefa é identificar qual das seguintes categorias melhor representa a palavra ou frase fornecida:

1. "problemas_financeiros" - Relacionado a dificuldades financeiras, dinheiro, desemprego, dívidas
2. "distante_religiao" - Relacionado a distância da religião, espiritualidade, fé, afastamento espiritual
3. "ato_criminoso_pecado" - Relacionado a atos errados, pecados, crimes, culpa, arrependimento
4. "doenca" - Relacionado a doenças, saúde, enfermidades, mal-estar, problemas de saúde
5. "ansiedade" - Relacionado a ansiedade, estresse, medo, nervosismo, preocupação
6. "desabafar" - Relacionado a necessidade de conversar, desabafar, ser ouvido, solidão
7. "redes_sociais" - Relacionado a redes sociais, curiosidade vinda das redes
8. "drogas" - Relacionado a uso de drogas, substâncias, dependência química, entorpecentes
9. "alcool" - Relacionado a uso de álcool, bebida, dependência alcoólica, alcoolismo
10. "sexo" - Relacionado a compulsão sexual, vício sexual, comportamento sexual compulsivo
11. "cigarro" - Relacionado a cigarro, fumo, tabagismo, nicotina, vício em tabaco
12. "outro" - Nenhuma das categorias acima se aplica claramente

IMPORTANTE:
- Seja flexível e considere sinônimos e variações
- Palavras como "enfermidade", "mal", "doente" devem mapear para "doenca"
- Palavras como "pecado", "erro", "culpa" devem mapear para "ato_criminoso_pecado"
- Palavras como "dinheiro", "financeiro", "desemprego" devem mapear para "problemas_financeiros"
- Palavras como "religião", "fé", "distante" devem mapear para "distante_religiao"
- Palavras como "solidão", "conversar", "sozinho" devem mapear para "desabafar"
- Palavras como "cocaína", "maconha", "crack", "vício", "dependência química" devem mapear para "drogas"
- Palavras como "bebida", "beber", "álcool", "alcoolismo", "bêbado" devem mapear para "alcool"
- Palavras como "pornografia", "compulsão sexual", "vício sexual" devem mapear para "sexo"
- Palavras como "fumo", "tabaco", "fumar", "tabagismo" devem mapear para "cigarro"
- Responda APENAS com o identificador da categoria (ex: "doenca", "problemas_financeiros", "drogas")
- Não adicione explicações ou pontuação"""

            user_prompt = f"Palavra ou frase: {sanitized_input}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response_text = self._make_chat_request(
                messages, temperature=0.2, max_tokens=20
            )

            theme = response_text.strip().lower()

            # Validate and normalize the response
            valid_themes = [
                "problemas_financeiros",
                "distante_religiao",
                "ato_criminoso_pecado",
                "doenca",
                "ansiedade",
                "desabafar",
                "redes_sociais",
                "drogas",
                "alcool",
                "sexo",
                "cigarro",
                "outro",
            ]

            if theme not in valid_themes:
                logger.warning(
                    f"Unexpected theme approximated: {theme}, defaulting to 'outro'"
                )
                theme = "outro"

            logger.info(f"Theme approximated: '{user_input}' -> '{theme}'")
            return theme

        except Exception as e:
            logger.error(f"Error approximating theme: {str(e)}", exc_info=True)
            # Default to "outro" on error
            return "outro"

    def generate_intent_response(
        self,
        user_message: str,
        intent: str,
        name: str,
        inferred_gender: Optional[str] = None,
        theme_id: Optional[str] = None,
        conversation_context: Optional[List[dict]] = None,
    ) -> List[str]:
        """
        Generate an empathetic, spiritually-aware response based on detected intent.

        The system prompt is assumed to be defined in the Ollama Modelfile.
        RAG context is injected as silent background context.

        Args:
            user_message: The user's original message
            intent: The detected intent category
            name: The user's name
            inferred_gender: Inferred gender (male/female/unknown or None)
            theme_id: Optional theme identifier
            conversation_context: Optional list of recent messages (dicts with 'role' and 'content')

        Returns:
            List of message strings to send sequentially
        """
        try:
            # Sanitize input before sending to LLM
            sanitized_message = sanitize_input(user_message)

            # Get RAG context (silent injection)
            rag_texts = get_rag_context(sanitized_message, limit=5)
            
            # Build system message with RAG context if available
            system_parts = []
            
            if rag_texts:
                # Inject RAG as silent context
                # Portuguese instruction because it's part of the system prompt for the LLM
                system_parts.append("CONTEXTO DE REFERÊNCIA (use de forma natural e implícita):")
                system_parts.append("\n\n".join(rag_texts))
                system_parts.append("\n---\n")
            
            # Add user context
            system_parts.append(f"Nome da pessoa: {name}")
            
            if inferred_gender and inferred_gender != "unknown":
                system_parts.append(GENDER_CONTEXT_INSTRUCTION.format(gender=inferred_gender))
            
            system_prompt = "\n".join(system_parts)

            # Build messages list with conversation context if provided
            messages = [{"role": "system", "content": system_prompt}]

            # Add conversation context if provided
            if conversation_context:
                for msg in conversation_context:
                    messages.append({"role": msg["role"], "content": msg["content"]})

            # Add the current user message
            messages.append({"role": "user", "content": sanitized_message})

            response_text = self._make_chat_request(
                messages, temperature=0.85, max_tokens=250
            )

            # Split response into multiple messages if separator is used
            messages = self._split_response_messages(response_text)

            logger.info(
                f"Generated intent-based response with {len(messages)} message(s) "
                f"for intent: {intent} (RAG chunks: {len(rag_texts)})"
            )
            return messages

        except Exception as e:
            logger.error(f"Error generating intent response: {str(e)}", exc_info=True)
            # Fallback to a simple empathetic message with a follow-up question
            return [
                "Obrigado por compartilhar isso comigo. O que mais te incomoda agora?"
            ]

    def generate_fallback_response(
        self,
        user_message: str,
        conversation_context: List[dict],
        name: str,
        inferred_gender: Optional[str] = None,
        theme_id: Optional[str] = None,
    ) -> List[str]:
        """
        Generate a context-aware fallback response when intent is unclear.

        The system prompt is assumed to be defined in the Ollama Modelfile.
        RAG context is injected as silent background context.

        Args:
            user_message: The user's current message
            conversation_context: List of recent messages (dicts with 'role' and 'content')
            name: The user's name
            inferred_gender: Inferred gender (male/female/unknown or None)
            theme_id: Optional theme identifier

        Returns:
            List of message strings to send sequentially
        """
        try:
            # Sanitize input before sending to LLM
            sanitized_message = sanitize_input(user_message)

            # Get RAG context (silent injection)
            rag_texts = get_rag_context(sanitized_message, limit=5)
            
            # Build system message with RAG context if available
            system_parts = []
            
            if rag_texts:
                # Inject RAG as silent context
                # Portuguese instruction because it's part of the system prompt for the LLM
                system_parts.append("CONTEXTO DE REFERÊNCIA (use de forma natural e implícita):")
                system_parts.append("\n\n".join(rag_texts))
                system_parts.append("\n---\n")
            
            # Add user context
            system_parts.append(f"Nome da pessoa: {name}")
            
            if inferred_gender and inferred_gender != "unknown":
                system_parts.append(GENDER_CONTEXT_INSTRUCTION.format(gender=inferred_gender))
            
            system_parts.append("Esta é uma continuação natural da conversa.")
            
            system_prompt = "\n".join(system_parts)

            # Build conversation context for the LLM
            context_messages = []
            for msg in conversation_context:
                context_messages.append({"role": msg["role"], "content": msg["content"]})

            # Add the current user message to context
            context_messages.append({"role": "user", "content": sanitized_message})

            # Prepend system message
            all_messages = [{"role": "system", "content": system_prompt}] + context_messages

            response_text = self._make_chat_request(
                all_messages, temperature=0.85, max_tokens=350
            )

            # Split response into multiple messages if separator is used
            messages = self._split_response_messages(response_text)

            logger.info(
                f"Generated fallback response with {len(messages)} message(s) "
                f"(RAG chunks: {len(rag_texts)})"
            )
            return messages

        except Exception as e:
            logger.error(f"Error generating fallback response: {str(e)}", exc_info=True)
            # Fallback to a simple empathetic message with a follow-up question
            return [
                "Obrigado por compartilhar isso comigo. Como você está se sentindo agora?"
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
