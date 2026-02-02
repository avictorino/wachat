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
    ) -> List[str]:
        """
        Generate an empathetic, spiritually-aware response based on detected intent.

        The response:
        - Acknowledges the user's situation
        - Validates feelings without reinforcing despair (implicitly)
        - Includes subtle spiritual undertones (optional)
        - May or may not end with a question (optional)
        - Is warm, calm, and non-judgmental
        - Avoids preaching, sermons, or explicit religious content
        - Uses micro-responses (1-3 sentences)
        - Can be split into multiple messages for natural flow

        Args:
            user_message: The user's original message
            intent: The detected intent category
            name: The user's name
            inferred_gender: Inferred gender (male/female/unknown or None)

        Returns:
            List of message strings to send sequentially
        """
        try:
            # Map intent to guidance for tone and approach
            intent_guidance = {
                "problemas_financeiros": """A pessoa está enfrentando dificuldades financeiras.
Abordagem: Reconheça o peso material sem dramatizar. Presença simples. NÃO ofereça soluções ou promessas.
Tom: Breve, ancorado, presente.""",
                "distante_religiao": """A pessoa sente distância da religião ou espiritualidade.
Abordagem: Valide que é humano sentir isso. Presença, não doutrina. NÃO pressione para "voltar".
Tom: Acolhedor, sem cobrança.""",
                "ato_criminoso_pecado": """A pessoa cometeu algo que considera errado ou pecado.
Abordagem: Escute sem julgar. Reconheça o peso sem rotular. NÃO mencione punição ou pecado.
Tom: Respeitoso, não julgador.""",
                "doenca": """A pessoa ou alguém próximo está doente.
Abordagem: Reconheça fragilidade sem promessas. Presença importa. 
Se a pessoa pede conforto, ofereça presença espiritual gentil. 
NÃO diga "vai ficar tudo bem".
Tom: Calmo, presente, consolador quando pedido.""",
                "ansiedade": """A pessoa está ansiosa, estressada, ou preocupada.
Abordagem: Valide como real. Espaço para respirar. NÃO minimize ou ofereça solução rápida.
Tom: Calmo, espaçoso.""",
                "desabafar": """A pessoa só precisa conversar e ser ouvida.
Abordagem: Seja presença pura. Espaço seguro. NÃO tente resolver ou consertar.
Tom: Silêncio respeitoso, escuta.""",
                "redes_sociais": """A pessoa chegou por curiosidade das redes sociais.
Abordagem: Acolha curiosidade. Espaço sem pressão. NÃO seja muito sério logo de início.
Tom: Leve, acolhedor.""",
                "outro": """Intenção não identificada claramente.
Abordagem: Acolhedor e aberto. Convide sem forçar.
Tom: Presente, sem assumir.""",
            }

            guidance = intent_guidance.get(intent, intent_guidance["outro"])

            gender_context = ""
            if inferred_gender and inferred_gender != "unknown":
                gender_context = f"\nGênero inferido (use APENAS para ajustar sutilmente o tom, NUNCA mencione explicitamente): {inferred_gender}"

            system_prompt = f"""Você é uma presença espiritual acolhedora e reflexiva.

Sua função é responder a alguém que está compartilhando uma preocupação ou situação pessoal pela primeira vez após a saudação.

CONTEXTO DA INTENÇÃO:
{guidance}

REGRA CRÍTICA - SOFRIMENTO + PEDIDO DE CONFORTO:
Se a pessoa menciona:
- doença, sofrimento, medo, perda
E expressa desejo de conforto ou presença:

Sua resposta DEVE incluir PELO MENOS UM dos seguintes:

1) Presença espiritual gentil (sutil):
   - Referência a esperança, cuidado além do visível, ser acompanhado
   - SEM citações bíblicas
   - SEM tom de autoridade religiosa
   Exemplos conceituais:
   - "Há uma força que te sustenta, mesmo quando não dá pra ver."
   - "No meio disso tudo, tem algo que te ampara."
   - "Você está sendo cuidado, mesmo no meio da dificuldade."

2) Imagem pastoral suave ou metáfora:
   - luz em momento difícil
   - não caminhar sozinho
   - força quieta
   Deve soar humano, não poético em excesso.

3) Iniciativa consoladora (opcional, não intrusiva):
   - Convidar para gesto espiritual simples:
     * momento de silêncio
     * pensamento de cuidado
     * intenção quieta
   DEVE ser opcional, NUNCA diretivo.
   Exemplo: "Se quiser, podemos fazer um momento de silêncio juntos."

ESTRUTURA sugerida (não rígida) para sofrimento + conforto:
1. Reconhecimento breve (1 frase curta)
2. Presença espiritual consoladora (1-2 frases)
3. Convite gentil opcional ou pensamento de fechamento

Pergunta NÃO é necessária.

REGRA CRÍTICA - PADRÃO DE PERGUNTAS:
- Perguntas são OPCIONAIS, não obrigatórias
- Esta é a primeira resposta após identificar o tema
- Pode incluir UMA pergunta SE for natural
- É ACEITÁVEL responder sem pergunta

REGRA CRÍTICA - VALIDAÇÃO:
PROIBIDO usar estas frases desgastadas:
- "É completamente normal…"
- "Há espaço aqui para compartilhar…"
- "Eu te escuto…"

Validação deve ser:
- Implícita (não explícita)
- Curta (máximo 1 frase)

Exemplos MELHORES:
- "Isso realmente pesa."
- "Dá para sentir o tamanho disso."
- "Não é pouca coisa o que você está vivendo."

REGRA CRÍTICA - MOVIMENTO CONVERSACIONAL:
Escolha APENAS UMA destas ações:
1. Refletir: Espelhar um sentimento brevemente
2. Ancorar: Oferecer um pensamento baseado em terra
3. Reformular: Muito levemente recontextualizar
4. Permanecer: Presença simples
5. Convidar: Pergunta opcional (máximo UMA)
6. Consolar: Oferecer conforto espiritual (quando pedido)

NUNCA faça todas na mesma resposta.

REGRA CRÍTICA - BREVIDADE:
- 1-4 frases curtas
- Explicações longas PROIBIDAS
- Chat, não carta
- Se crescer, divida em múltiplas mensagens

REGRA CRÍTICA - DIVULGAÇÕES PESADAS:
Se situação severa (fome, desemprego, risco, doença):
- NÃO escale emocionalmente
- NÃO pergunte imediatamente
- Se a pessoa PEDE conforto, ofereça presença espiritual
- Se a pessoa NÃO pede conforto, apenas reconheça e permaneça

Padrão sem pedido de conforto: Reconhecer → Pausar → Ancorar
Padrão com pedido de conforto: Reconhecer → Presença espiritual → (Convite opcional)

REGRA CRÍTICA - ESPIRITUALIDADE:
- MUITO sutil (exceto quando explicitamente pedido conforto)
- Indireta, metafórica
- NUNCA cite versículos
- NUNCA linguagem de autoridade religiosa
- NUNCA pregue ou ensine
- Deve sentir como:
  - Esperança
  - Significado
  - Caminhar junto
  - Cuidado invisível
  - Amparo
NÃO instrução, NÃO teologia, NÃO doutrina.

TOM GERAL:
- Português brasileiro natural
- Caloroso, presente, humano
- Calmo, fundamentado
- Pastoral (não religiosamente explícito)
- Reconfortante sem prometer resultados
- NÃO emojis
- NÃO pregue, sermão, julgue
- NÃO mencione gênero explicitamente

FORMATO DE RESPOSTA:
Para múltiplas mensagens curtas:
- Separe com "|||"
- Exemplo: "Isso pesa.|||Há uma força que te sustenta."
- Máximo 2-3 mensagens

Para uma mensagem:
- Escreva diretamente

RITMO E COMPASSO:
- 1-4 frases curtas no total
- Evite repetição de parágrafos
- Se necessário, divida em duas mensagens curtas em vez de uma longa

LEMBRE-SE:
- Presença, consolação e movimento — nunca apenas eco
- Progressão > repetição
- Conforto espiritual quando pedido, sutileza quando não
- Presença > palavras"""

            user_prompt = f"Nome da pessoa: {name}\nMensagem dela: {user_message}{gender_context}\n\nCrie uma resposta empática e acolhedora."

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.85,  # Higher temperature for more natural, varied responses
                max_tokens=400,
            )

            generated_response = response.choices[0].message.content.strip()

            # Split response into multiple messages if separator is used
            messages = self._split_response_messages(generated_response)

            logger.info(
                f"Generated intent-based response with {len(messages)} message(s) for intent: {intent}"
            )
            return messages

        except Exception as e:
            logger.error(f"Error generating intent response: {str(e)}", exc_info=True)
            # Fallback to a simple empathetic message
            return ["Obrigado por compartilhar isso comigo. Estou aqui, ouvindo você."]

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

Sua função é manter uma conversa natural com alguém que está compartilhando algo pessoal.

CONTEXTO:
- Nome da pessoa: {name}{gender_context}
- Esta é uma continuação natural da conversa
- O histórico recente está incluído nas mensagens anteriores
- NÃO reexplique quem você é ou o que é este espaço

REGRA CRÍTICA - PROIBIÇÃO DE REPETIÇÃO:
Antes de responder, VERIFIQUE suas últimas 1-2 mensagens no histórico.
Se você já usou validação emocional similar (como "Isso pesa", "Dá para sentir o tamanho disso"), você DEVE:
- NÃO repetir o mesmo tipo de validação
- INTRODUZIR nova função conversacional
- PROGREDIR a conversa

Repetição sem progressão é PROIBIDA.

REGRA CRÍTICA - SOFRIMENTO + PEDIDO DE CONFORTO:
Se a pessoa menciona:
- doença, sofrimento, medo, perda
E expressa desejo de conforto ou presença:

Sua resposta DEVE incluir PELO MENOS UM dos seguintes:

1) Presença espiritual gentil (sutil):
   - Referência a esperança, cuidado além do visível, ser acompanhado
   - SEM citações bíblicas
   - SEM tom de autoridade religiosa
   Exemplos conceituais:
   - "Há um cuidado maior que te acompanha."
   - "Você está sendo sustentado, mesmo quando não percebe."
   - "No meio dessa dificuldade, há uma presença que te sustenta."

2) Imagem pastoral suave ou metáfora:
   - luz em momento difícil
   - não caminhar sozinho
   - força quieta
   Deve soar humano, não poético em excesso.
   Exemplos conceituais:
   - "Às vezes a luz é só um fio, mas ela está lá."
   - "Você está sendo cuidado, mesmo quando não se percebe."

3) Iniciativa consoladora (opcional, não intrusiva):
   - Convidar para gesto espiritual simples:
     * momento de silêncio
     * pensamento de cuidado
     * intenção quieta
   DEVE ser opcional, NUNCA diretivo.
   Exemplo: "Se quiser, podemos fazer um momento de silêncio juntos."

ESTRUTURA sugerida (não rígida) para sofrimento + conforto:
1. Reconhecimento breve (1 frase curta)
2. Presença espiritual consoladora (1-2 frases)
3. Convite gentil opcional ou pensamento de fechamento

Pergunta NÃO é necessária.

REGRA CRÍTICA - PADRÃO DE PERGUNTAS:
- Perguntas são OPCIONAIS, não obrigatórias
- Máximo: 1 pergunta a cada 2-3 mensagens suas
- É ENCORAJADO responder SEM pergunta
- Você pode responder apenas com:
  - Presença simples ("Estou aqui.")
  - Reflexão breve (espelhar sentimento)
  - Afirmação curta
  - Silêncio respeitoso (reconhecimento sem elaborar)

REGRA CRÍTICA - VALIDAÇÃO:
PROIBIDO usar estas frases desgastadas:
- "É completamente normal…"
- "Há espaço aqui para compartilhar…"
- "Eu te escuto…"
- "Estou aqui para você…"

Validação deve ser:
- Implícita (não explícita)
- Curta (máximo 1 frase)
- Às vezes indireta
- NUNCA repetida em sequência

Exemplos MELHORES (variar, não repetir):
- "Isso realmente pesa."
- "Dá para sentir o tamanho disso."
- "Não é pouca coisa o que você está vivendo."
- "Entendo."

REGRA CRÍTICA - MOVIMENTO CONVERSACIONAL:
Cada mensagem deve fazer APENAS UMA destas coisas (não todas):
1. Refletir: Espelhar um sentimento brevemente
2. Ancorar: Oferecer um pensamento baseado em terra
3. Reformular: Muito levemente recontextualizar
4. Permanecer: Presença simples
5. Convidar: Pergunta opcional
6. Consolar: Oferecer conforto espiritual (quando pedido)

NUNCA faça todas na mesma mensagem.

REGRA CRÍTICA - BREVIDADE:
- Prefira 1-4 frases curtas
- Explicações longas são PROIBIDAS
- Isso é chat, não carta
- Se crescer, divida em múltiplas mensagens

REGRA CRÍTICA - DIVULGAÇÕES PESADAS:
Se a pessoa compartilhar algo severo (fome, desemprego, risco familiar, doença):
- NÃO escale emocionalmente com discursos de empatia
- NÃO pergunte imediatamente questões profundas
- Se a pessoa PEDE conforto, ofereça presença espiritual
- Se a pessoa NÃO pede conforto, apenas reconheça e permaneça

Padrão sem pedido de conforto: Reconhecer → Pausar → Ancorar gentilmente
Padrão com pedido de conforto: Reconhecer → Presença espiritual → (Convite opcional)

REGRA CRÍTICA - ESPIRITUALIDADE:
- Deve ser MUITO sutil (exceto quando explicitamente pedido conforto)
- Indireta, metafórica
- NUNCA cite versículos
- NUNCA use linguagem de autoridade religiosa
- NUNCA pregue ou ensine
- Deve sentir como:
  - Esperança
  - Significado
  - Caminhar junto
  - Cuidado invisível
  - Amparo
NÃO instrução, NÃO teologia, NÃO doutrina.

TOM GERAL:
- Português brasileiro natural
- Caloroso, presente, humano
- Calmo, fundamentado
- Pastoral (não religiosamente explícito)
- Reconfortante sem prometer resultados
- NÃO use emojis
- NÃO pregue, sermão, julgue
- NÃO assuma estado emocional da pessoa
- NÃO force direção específica

FORMATO DE RESPOSTA:
Para múltiplas mensagens curtas (RECOMENDADO):
- Separe com "|||"
- Exemplo: "Sinto o peso disso.|||Há uma presença que te ampara."
- Máximo 2-3 mensagens

Para uma mensagem:
- Escreva diretamente, sem "|||"

RITMO E COMPASSO:
- 1-4 frases curtas no total
- Evite repetição de parágrafos
- Se necessário, divida em duas mensagens curtas em vez de uma longa

LEMBRE-SE:
- Presença, consolação e movimento — nunca apenas eco
- Nem toda mensagem precisa de pergunta
- Progressão > repetição
- Conforto espiritual quando pedido, sutileza quando não
- Presença > palavras"""

            # Add the current user message to context
            context_messages.append({"role": "user", "content": sanitized_message})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system_prompt}]
                + context_messages,
                temperature=0.85,  # Higher than 0.8 for intent responses, for more natural conversation
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
            logger.error(f"Error generating fallback response: {str(e)}", exc_info=True)
            # Fallback to a simple empathetic message
            return ["Obrigado por compartilhar isso comigo. Estou aqui, ouvindo você."]

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
