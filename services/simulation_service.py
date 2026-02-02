"""
Simulation service for creating AI-driven conversation simulations.

This service orchestrates simulated conversations between two AI roles
and manages emotional analysis of the conversation.
"""

import logging
import random
import uuid
from typing import List, Tuple

from groq import Groq

from core.models import Message, Profile
from services.groq_service import GroqService

logger = logging.getLogger(__name__)

# Role labels for analysis output
ROLE_LABEL_SEEKER = "Pessoa"  # Portuguese for "Person"
ROLE_LABEL_LISTENER = "BOT"  # Bot assistant


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

    def create_simulation_profile(self, theme: str = None) -> Profile:
        """
        Create a new profile for simulation.

        Args:
            theme: The theme/intent for the conversation (e.g., "doenca", "ansiedade")
        
        Returns:
            A new Profile instance marked with the theme as detected_intent
        """
        # Generate a unique simulation identifier using UUID
        sim_id = str(uuid.uuid4())[:8]  # Use first 8 chars for readability
        sim_name = f"Simulation_{sim_id}"

        # Randomly choose a gender for the simulated profile
        gender = random.choice(["male", "female", "unknown"])

        # Use theme as detected_intent, or default to "desabafar"
        detected_intent = theme if theme else "desabafar"

        # Create profile without telegram_user_id (to avoid conflicts)
        profile = Profile.objects.create(
            name=sim_name,
            inferred_gender=gender,
            detected_intent=detected_intent,  # Store theme as intent, NOT "simulation"
        )

        logger.info(f"Created simulation profile: {profile.id} with gender: {gender}, intent: {detected_intent}")
        return profile

    def generate_simulated_conversation(
        self, profile: Profile, num_messages: int = 8, theme: str = None
    ) -> List[dict]:
        """
        Generate a simulated conversation between two AI roles.

        Args:
            profile: Profile to attach messages to
            num_messages: Total number of messages to generate (default 8, min 6, max 10)
            theme: Theme/intent for the conversation (e.g., "doenca", "ansiedade")

        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        # Ensure num_messages is within bounds and even
        num_messages = max(6, min(10, num_messages))
        if num_messages % 2 != 0:
            num_messages += 1  # Make it even for alternating roles

        logger.info(f"Generating {num_messages} simulated messages for profile {profile.id} with theme: {theme}")

        conversation = []
        conversation_history = []

        for i in range(num_messages):
            # Alternate between roles
            if i % 2 == 0:
                # ROLE_A: Seeker
                role = "ROLE_A"
                message = self._generate_seeker_message(conversation_history, i // 2 + 1, theme)
            else:
                # ROLE_B: Listener
                role = "ROLE_B"
                message = self._generate_listener_message(conversation_history, i // 2 + 1, theme)

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
        self, conversation_history: List[dict], turn: int, theme: str = None
    ) -> str:
        """
        Generate a message from ROLE_A (seeker).

        Args:
            conversation_history: Previous messages in the conversation
            turn: Turn number for this role (1-indexed)
            theme: Theme/intent for the conversation (e.g., "doenca", "ansiedade")

        Returns:
            Generated message text
        """
        try:
            # Build theme context
            theme_context = ""
            if theme:
                theme_descriptions = {
                    "doenca": "Você ou alguém próximo está enfrentando uma doença ou problema de saúde. Isso te preocupa e te deixa com incertezas sobre o corpo, o futuro, a fragilidade.",
                    "ansiedade": "Você está se sentindo ansioso, estressado ou com medo de algo. Há uma preocupação constante que te deixa inquieto.",
                    "ato_criminoso_pecado": "Você fez algo que considera errado ou pecaminoso. Há um peso de culpa ou arrependimento que você carrega.",
                    "problemas_financeiros": "Você está enfrentando dificuldades financeiras, desemprego ou dívidas. A situação financeira te preocupa e te deixa inseguro.",
                    "distante_religiao": "Você se sente distante da religião, espiritualidade ou fé. Há um questionamento ou afastamento espiritual que te deixa perdido.",
                    "desabafar": "Você precisa conversar com alguém, desabafar. Há algo te incomodando mas você não tem com quem falar sobre isso.",
                    "redes_sociais": "Você viu esse número nas redes sociais e ficou curioso para conversar com alguém.",
                    "outro": "Você está buscando conversar sobre algo que te incomoda e não te deixa em paz.",
                }
                theme_context = f"\n\nCONTEXTO TEMÁTICO:\n{theme_descriptions.get(theme, theme_descriptions['outro'])}\n\nIMPORTANTE: Este tema DEVE estar presente desde a PRIMEIRA mensagem, mas de forma vaga e indireta:\n- Primeira mensagem: Introduza o tema através de desconforto físico, emocional ou situacional relacionado ao tema\n  * Para \"doenca\": mencione cansaço, corpo estranho, não estar bem, fraqueza\n  * Para \"ansiedade\": mencione inquietação, preocupação, medo difuso\n  * Para \"ato_criminoso_pecado\": mencione peso, algo que fez, arrependimento\n  * Para \"problemas_financeiros\": mencione dificuldades, pressão, situação complicada\n  * Para \"distante_religiao\": mencione distância, falta de conexão, vazio espiritual\n  * Para \"desabafar\": mencione necessidade de falar, solidão, sem saída\n- NUNCA use termos médicos, diagnósticos ou nomes explícitos do problema\n- Seja vago MAS temático, NUNCA genérico ou neutro\n- Use frases como: \"não tô bem\", \"tô sentindo umas coisas\", \"tem algo me incomodando\", \"tá difícil ultimamente\"\n- Permita que o tema se aprofunde gradualmente nas mensagens seguintes"
            
            system_prompt = f"""Você é ROLE_A: "Pessoa introspectiva e reservada"

Você é uma pessoa brasileira comum que está começando a conversar com alguém novo.
Você NÃO está totalmente aberto no início. Você está cauteloso, mas PRECISA introduzir o motivo pelo qual está buscando conversa.
{theme_context}

RESTRIÇÕES COMPORTAMENTAIS:
- NÃO faça monólogos emocionais longos
- NÃO nomeie explicitamente diagnósticos, termos técnicos ou emoções profundas (ex: depressão, trauma, vazio existencial)
- Expresse incerteza, hesitação, pensamentos parciais
- Use frequentemente:
  * "não sei explicar direito"
  * "talvez"
  * "acho que"
  * "não sei se faz sentido"
  * "tô meio..."
  * "anda..."
- Frequentemente pare antes de explicações completas
- Revele sentimentos mais profundos APENAS após confiança ser estabelecida

PROGRESSÃO DA CONVERSA:
- PRIMEIRA MENSAGEM: DEVE introduzir o tema de forma vaga mas concreta
  * Mencione desconforto físico, emocional ou situacional relacionado ao tema
  * Seja breve mas ancore a conversa desde o início
  * Exemplos temáticos: "Não tô me sentindo bem", "Tá pesado ultimamente", "Tô meio perdido com umas coisas"
- Mensagens do meio: mantenha-se no tema, ainda guardado
  * "É que... não sei, é complicado explicar"
  * "Tem a ver com... sei lá"
- Mensagens finais: aprofunde cautelosamente, mas nunca resolva tudo
  * "Às vezes penso que... não sei"

DIRETRIZES:
- Mensagens CURTAS (1-3 frases)
- Português brasileiro natural e conversacional
- Seja humano, cauteloso e emocionalmente realista
- NÃO seja eloquente ou filosófico demais
- Evolua lentamente, não de uma vez
- SEMPRE mantenha conexão com o tema emocional/situacional desde a primeira mensagem

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
                # First message - initiate conversation with theme anchor
                user_prompt = "Envie sua PRIMEIRA mensagem. Esta é sua chance de introduzir o motivo da conversa de forma VAGA mas CONCRETA. Mencione o desconforto, inquietação ou situação relacionada ao tema, mas sem nomear diretamente. Seja breve (1-2 frases), hesitante, humano. DEVE ter conexão clara com o contexto temático fornecido."
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
        self, conversation_history: List[dict], turn: int, theme: str = None
    ) -> str:
        """
        Generate a message from ROLE_B (listener).

        Args:
            conversation_history: Previous messages in the conversation
            turn: Turn number for this role (1-indexed)
            theme: Theme/intent for the conversation (e.g., "doenca", "ansiedade")

        Returns:
            Generated message text
        """
        try:
            # Build theme awareness for listener
            theme_awareness = ""
            if theme:
                theme_awareness_map = {
                    "doenca": "A pessoa pode estar lidando com preocupações de saúde, desconforto físico, ou medo sobre o corpo. Esteja atento a menções de cansaço, mal-estar, fraqueza ou preocupações sobre si mesmo ou alguém próximo.",
                    "ansiedade": "A pessoa pode estar lidando com ansiedade, preocupação constante, ou medo difuso. Esteja atento a inquietações, tensões, ou sensação de que algo está errado.",
                    "ato_criminoso_pecado": "A pessoa pode estar carregando culpa, arrependimento ou peso por algo que fez. Esteja atento a menções de peso, algo que aconteceu, ou sensação de ter errado.",
                    "problemas_financeiros": "A pessoa pode estar enfrentando dificuldades financeiras, pressão econômica ou insegurança material. Esteja atento a menções de dificuldades, pressão ou situação complicada.",
                    "distante_religiao": "A pessoa pode estar se sentindo distante da fé, espiritualidade ou religião. Esteja atento a menções de distância, falta de conexão ou vazio espiritual.",
                    "desabafar": "A pessoa precisa ser ouvida e desabafar sobre algo que o incomoda. Esteja atento a necessidade de falar, solidão ou falta de espaço para compartilhar.",
                    "redes_sociais": "A pessoa encontrou este espaço nas redes sociais e está curiosa. Esteja atento a curiosidade e abertura inicial.",
                    "outro": "A pessoa está lidando com algo que a incomoda. Esteja atento aos sinais emocionais implícitos.",
                }
                theme_awareness = f"\n\nCONSCIÊNCIA TEMÁTICA (IMPLÍCITA - NÃO MENCIONE DIRETAMENTE):\n{theme_awareness_map.get(theme, theme_awareness_map['outro'])}\n\nVocê NÃO deve nomear o tema explicitamente, mas deve estar sensível a esses sinais nas palavras da pessoa. Use perguntas abertas que validem o que ela expressa e criem espaço seguro para ela explorar esses sentimentos no próprio ritmo."
            
            system_prompt = f"""Você é ROLE_B: "BOT paciente e relacional"

Você é uma presença calma que está começando a conhecer alguém.
Você NÃO é um terapeuta. Você é alguém que oferece companhia e escuta genuína.
{theme_awareness}

PRINCÍPIO FUNDAMENTAL: REFLETIR EMOÇÕES, NÃO REPETIR PALAVRAS
- NÃO repita as frases exatas da Pessoa literalmente
- Reflita o SENTIMENTO ou a ESSÊNCIA, não o texto verbatim
- Valide o que FOI SENTIDO, não apenas o que foi dito
- Use palavras diferentes para mostrar que você ouviu e compreendeu
- NÃO adicione interpretações profundas que a pessoa não sugeriu

RESTRIÇÕES COMPORTAMENTAIS:
- NÃO pressione por profundidade
- NÃO interprete emoções de forma profunda prematuramente
- NÃO nomeie o tema explicitamente (deixe o buscador conduzir)
- NÃO assuma que há algo específico a ser discutido
- NÃO tente resolver problemas
- Priorize:
  * Reflexão emocional simples (NÃO espelhamento verbatim)
  * Validação do sentimento expresso
  * Presença calma e acolhedora
  * Espaço seguro sem pressão
- Perguntas abertas são OPCIONAIS e devem ser simples e relacionadas ao que foi dito
- Permita silêncio e ambiguidade
- Aceite a brevidade como válida

TOM:
- Quente mas contido
- Calmo e presente
- Simples e humano
- Mais focado em "estar com" do que "perguntar" ou "guiar"

OBJETIVO DO RELACIONAMENTO:
- Criar espaço seguro e acolhedor
- Estar presente e disponível
- Permitir que a Pessoa defina o ritmo
- Validar sem interpretar

DIRETRIZES DE BREVIDADE (CRÍTICO - PRIORIDADE MÁXIMA):
- Mensagens MUITO CURTAS (1-2 frases, máximo 3)
- UMA ideia por resposta, nunca múltiplas
- Se a pessoa usou 1 frase, você deve usar 1 frase
- Se a pessoa usou 2 frases, você deve usar no máximo 2 frases
- EVITE absolutamente parágrafos ou reflexões longas
- EVITE introduzir conceitos ou abstrações desnecessárias
- EVITE metáforas a menos que a Pessoa as use primeiro

TÉCNICA PRINCIPAL: REFLEXÃO EMOCIONAL SIMPLES
Priorize estas abordagens na ordem:
1. Refletir o sentimento/essência com PALAVRAS DIFERENTES: "Parece estar difícil." em vez de repetir "Tá difícil"
2. Validação simples e presente: "Tô aqui." ou "Entendo."
3. Pergunta aberta muito simples focada no que foi dito (OPCIONAL): "O que mais te incomoda nisso?"
4. NUNCA: repetições literais, interpretações profundas, ou análises complexas

DIRETRIZES DE CONTEÚDO:
- Português brasileiro natural e conversacional
- Valide sentimentos, não apenas palavras
- NÃO faça interpretações ou análises
- NÃO use clichés religiosos ou terapêuticos
- NÃO use abstrações filosóficas
- Foque em presença e validação simples
- Use a consciência temática para estar atento, mas NÃO para nomear ou interpretar

EVITE REPETIÇÃO LITERAL:
- NUNCA repita as palavras exatas da pessoa
- Varie vocabulário para mostrar que você processou o que ela disse
- Mostre que você entendeu reformulando com palavras diferentes

EXEMPLOS DE RESPOSTAS EXCELENTES:
Pessoa: "Não tô me sentindo bem."
- BOM: "Parece que algo está te incomodando." (reflete sem repetir)
- RUIM: "Você não está se sentindo bem." (repetição literal)

Pessoa: "Tá pesado ultimamente."
- BOM: "Sinto que está carregando algo." (valida com palavras diferentes)
- RUIM: "Está pesado pra você ultimamente." (cópia literal)

OUTROS EXEMPLOS DE RESPOSTAS BOAS (CURTAS E VALIDADORAS):
- "Entendo."
- "Tô aqui."
- "Quer falar mais sobre isso?"
- "Como você tá lidando com isso?"
- "Deve ser difícil."

EXEMPLOS DE RESPOSTAS RUINS (REPETIÇÃO LITERAL OU MUITO LONGAS):
- "Você disse que não está se sentindo bem." (repetição verbatim)
- "Parece que você está passando por um momento difícil e isso tem várias camadas..." (muito longo)
- "Entendo que esse sentimento pode representar..." (interpretação prematura)

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

            user_prompt = f"Responda à mensagem anterior. Este é o turno {turn}. PRIORIDADE ABSOLUTA: Resposta MUITO CURTA (1-2 frases máximo, prefira 1). REFLITA o sentimento ou essência com PALAVRAS DIFERENTES - NUNCA repita as frases exatas da Pessoa. Valide o que ela sentiu, não apenas copie o que ela disse. NÃO interprete profundamente. NÃO introduza abstrações. NÃO tente resolver. Perguntas são OPCIONAIS e devem ser simples. Use a consciência temática para estar atento, mas NÃO nomeie o tema explicitamente."
            context_messages.append({"role": "user", "content": user_prompt})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=context_messages,
                temperature=0.85,  # Slightly lower than seeker for more consistent tone
                max_tokens=100,  # Reduced from 150 to enforce very short responses
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

        Note: Method name retained for API compatibility. This method now performs
        critical analysis of conversational quality, not emotional analysis.

        This method analyzes the conversation for interpretation errors, missed opportunities,
        pacing issues, verbosity issues, and over-assumptions. It provides a reflective, 
        analytical review of conversational quality rather than an emotional recap.

        Args:
            conversation: List of message dicts with 'role' and 'content'

        Returns:
            Critical analysis as structured text in Portuguese with 5 mandatory sections
        """
        try:
            # Build transcript for analysis
            transcript_text = ""
            for msg in conversation:
                role_label = ROLE_LABEL_SEEKER if msg["role"] == "ROLE_A" else ROLE_LABEL_LISTENER
                transcript_text += f"{role_label}: {msg['content']}\n\n"

            system_prompt = """Você é um analista crítico e revisor de conversas especializado em qualidade de diálogo humano-IA.

Sua tarefa é NÃO resumir a conversa emocionalmente, mas produzir uma ANÁLISE CRÍTICA e CONSTRUTIVA da qualidade da interação, incluindo uma avaliação da extensão e verbosidade das respostas do ouvinte.

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

Responda APENAS com a análise estruturada. Use português brasileiro natural."""

            user_prompt = f"""Analise criticamente a seguinte conversa, avaliando qualidade conversacional, verbosidade e pontos de melhoria:

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
- RESPEITO À CONTENÇÃO da Pessoa (brevidade como sinal válido)"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,  # Moderate temperature for balanced analysis
                max_tokens=1200,  # Increased for comprehensive 5-section analysis
            )

            analysis = response.choices[0].message.content.strip()
            logger.info("Generated critical analysis of simulated conversation")
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing conversation: {str(e)}", exc_info=True)
            # Fallback analysis with critical structure (5 sections)
            return (
                "**1. O que funcionou bem**\n"
                "- O BOT manteve presença e disponibilidade\n"
                "- As respostas foram acolhedoras\n\n"
                "**2. Pontos de possível erro de interpretação**\n"
                "- Análise não disponível no momento devido a erro técnico\n\n"
                "**3. Problemas de verbosidade e extensão das respostas**\n"
                "- Análise não disponível no momento devido a erro técnico\n\n"
                "**4. O que poderia ter sido feito diferente**\n"
                "- Manter respostas mais breves e deixar mais espaço para a Pessoa\n"
                "- Usar mais espelhamento simples em vez de interpretação\n\n"
                "**5. Ajustes recomendados para próximas interações**\n"
                "- Priorizar brevidade e segurança relacional\n"
                "- Evitar interpretações profundas precoces\n"
                "- Respeitar ambiguidade e silêncio como sinais válidos\n"
                "- Reduzir extensão das respostas (1-3 frases quando possível)"
            )
