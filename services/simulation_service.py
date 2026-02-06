"""
Simulation service for creating AI-driven conversation simulations.

This service orchestrates simulated conversations between two AI roles
and manages emotional analysis of the conversation.
"""

import logging
import os
import random
import uuid
from typing import List, Tuple

import requests
from faker import Faker

from core.models import Message, Profile

logger = logging.getLogger(__name__)

# Initialize Faker once at module level for efficiency
_faker = Faker('pt_BR')

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

    def __init__(self, api_key: str = None):
        """
        Initialize the simulation service.

        Args:
            api_key: API key (kept for compatibility, not used)
        """
        # Direct Ollama configuration for USER simulation
        self.base_url = os.environ.get("SIMULATION_OLLAMA_BASE_URL", os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
        self.model = os.environ.get("SIMULATION_OLLAMA_MODEL", os.environ.get("OLLAMA_MODEL", "llama3.1"))
        self.api_url = f"{self.base_url}/api/chat"
        
        logger.info(f"Initialized SimulationService with model={self.model}, base_url={self.base_url}")

    def _call_llm(self, messages: List[dict], temperature: float = 0.85, max_tokens: int = 250) -> str:
        """
        Call Ollama API directly with messages (USER simulator).

        Args:
            messages: List of message dicts with role and content
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            
            response_data = response.json()
            content = response_data.get("message", {}).get("content", "").strip()
            
            if not content:
                logger.warning("Ollama returned empty content")
                raise ValueError("Empty response from Ollama")
            
            return content
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama API: {str(e)}", exc_info=True)
            raise

    def create_simulation_profile(self, theme: str = None) -> Profile:
        """
        Create a new profile for simulation.

        Args:
            theme: The theme for the conversation (e.g., "drogas", "alcool", "ansiedade")
        
        Returns:
            A new Profile instance marked with the theme
        """
        # Randomly choose a gender for the simulated profile
        gender = random.choice(["male", "female"])
        
        # Generate a realistic name based on the gender using Faker
        if gender == "male":
            sim_name = _faker.first_name_male()
        else:
            sim_name = _faker.first_name_female()

        # Use theme, or default to "desabafar"
        theme = theme if theme else "desabafar"

        # Create profile with prompt_theme persisted
        profile = Profile.objects.create(
            name=sim_name,
            inferred_gender=gender,
            prompt_theme=theme,  # Persist theme on profile
        )

        logger.info(f"Created simulation profile: {profile.id} with name: {sim_name}, gender: {gender}, theme: {theme}")
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
        Generate a message from ROLE_A (seeker) - simulating a REAL USER.

        Args:
            conversation_history: Previous messages in the conversation
            turn: Turn number for this role (1-indexed)
            theme: Theme/intent for the conversation (e.g., "drogas", "alcool", "ansiedade")

        Returns:
            Generated message text
        """
        try:
            # Build theme-driven prompt
            theme_instruction = self._build_theme_instruction(theme)
            
            system_prompt = f"""Você está simulando um usuário humano real.
Você NÃO é um assistente.
Você escreve mensagens curtas e honestas.
Você não se explica.

Tema: {theme or 'desabafar'}

{theme_instruction}

REGRAS GERAIS:
- Linguagem em primeira pessoa ("eu", "estou", "me sinto")
- Mensagens curtas e imperfeitas (1-3 frases)
- Sem análise, sem explicações
- Sem consciência de estar sendo simulado
- Sem meta-linguagem ("como um usuário", "simulação", etc.)
- Você fala casualmente, com dúvidas e hesitação
- Você não usa linguagem formal
- Você nunca menciona religião explicitamente a menos que o assistente a introduza primeiro

COMPORTAMENTO:
- Hesitante e real
- Expresse incerteza naturalmente
- Respostas curtas que parecem autênticas
- Reaja ao que o bot diz, não apenas continue seu próprio raciocínio
- Construa a conversa progressivamente

Responda APENAS com a mensagem do usuário, sem explicações."""

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
                # First message - must introduce theme naturally
                # Check if theme needs special first message handling
                substance_themes = ["drogas", "alcool", "cigarro", "sexo"]
                if theme in substance_themes:
                    theme_examples = {
                        "drogas": "tenho usado', 'voltei a usar', 'não consigo controlar",
                        "alcool": "tô bebendo demais', 'não consigo parar de beber', 'bebida tá me prejudicando",
                        "cigarro": "tô fumando muito', 'não consigo largar o cigarro', 'vício em cigarro",
                        "sexo": "compulsão sexual', 'não consigo controlar', 'comportamento sexual",
                    }
                    examples = theme_examples.get(theme, "")
                    user_prompt = f"Envie sua PRIMEIRA mensagem. Você deve mencionar sua luta com {theme} de forma hesitante e pessoal. Seja breve (1-2 frases), use palavras como '{examples}'. Fale como alguém real que está abrindo-se pela primeira vez."
                else:
                    user_prompt = "Envie sua PRIMEIRA mensagem. Introduza o tema de forma natural e hesitante. Seja breve (1-2 frases)."
            elif turn <= 2:
                user_prompt = f"Continue a conversa respondendo à mensagem anterior. Ainda esteja cauteloso. Use frases curtas."
            else:
                user_prompt = f"Continue a conversa respondendo à mensagem anterior. Este é o turno {turn}. Você pode se abrir um pouco mais, mas ainda com hesitação. Mantenha coerência com o tema."

            context_messages.append({"role": "user", "content": user_prompt})

            # Use higher temperature for more natural variation
            response_text = self._call_llm(context_messages, temperature=0.9, max_tokens=250)

            return response_text

        except Exception as e:
            logger.error(f"Error generating seeker message: {str(e)}", exc_info=True)
            # Fallback messages based on theme
            fallbacks = self._get_theme_fallbacks(theme)
            return fallbacks[turn % len(fallbacks)]

    def _build_theme_instruction(self, theme: str) -> str:
        """
        Build theme-specific instruction for user simulation.

        Args:
            theme: The conversation theme

        Returns:
            Theme-specific instruction text
        """
        theme_instructions = {
            "drogas": """
REGRAS ESPECÍFICAS DO TEMA (drogas):
Sua PRIMEIRA mensagem DEVE:
- Mencionar explicitamente sua luta com drogas ou uso de substâncias
- Soar hesitante, real e pessoal
- Usar padrões como:
  * "tenho usado drogas e isso está me incomodando"
  * "parei um tempo, mas voltei a usar"
  * "não sei mais controlar isso"
  * "tô usando de novo"

NÃO use frases exatas - varie naturalmente.
Mensagens subsequentes:
- Reaja naturalmente às respostas do bot
- Intensifique ou suavize baseado no fluxo da conversa
- Mantenha coerência com o tema de dependência
""",
            "alcool": """
REGRAS ESPECÍFICAS DO TEMA (alcool):
Sua PRIMEIRA mensagem DEVE:
- Mencionar explicitamente sua luta com álcool
- Soar hesitante, real e pessoal
- Usar padrões como "tô bebendo demais", "não consigo parar de beber"

Mensagens subsequentes:
- Reaja naturalmente às respostas do bot
- Mantenha coerência com o tema de álcool
""",
            "cigarro": """
REGRAS ESPECÍFICAS DO TEMA (cigarro):
Sua PRIMEIRA mensagem DEVE:
- Mencionar explicitamente sua luta com cigarro/fumo
- Soar hesitante, real e pessoal
- Usar padrões como "tô fumando muito", "não consigo largar o cigarro"

Mensagens subsequentes:
- Reaja naturalmente às respostas do bot
- Mantenha coerência com o tema de tabagismo
""",
            "sexo": """
REGRAS ESPECÍFICAS DO TEMA (sexo):
Sua PRIMEIRA mensagem DEVE:
- Mencionar sua luta com compulsão ou comportamento sexual
- Soar hesitante, real e pessoal
- Usar padrões como "compulsão", "não consigo controlar"

Mensagens subsequentes:
- Reaja naturalmente às respostas do bot
- Mantenha coerência com o tema
""",
            "ansiedade": """
REGRAS ESPECÍFICAS DO TEMA (ansiedade):
Sua PRIMEIRA mensagem deve:
- Expressar ansiedade, estresse ou preocupação
- Usar padrões como "tô muito ansioso", "não consigo parar de me preocupar"

Mensagens subsequentes:
- Reaja naturalmente às respostas do bot
- Mantenha coerência com o tema de ansiedade
""",
            "solidao": """
REGRAS ESPECÍFICAS DO TEMA (solidao):
Sua PRIMEIRA mensagem deve:
- Expressar sentimentos de solidão ou isolamento
- Usar padrões como "me sinto sozinho", "não tenho com quem conversar"

Mensagens subsequentes:
- Reaja naturalmente às respostas do bot
- Mantenha coerência com o tema de solidão
""",
        }
        
        default_instruction = """
REGRAS ESPECÍFICAS DO TEMA (geral):
Sua PRIMEIRA mensagem deve introduzir sua preocupação de forma natural e hesitante.
Mensagens subsequentes devem reagir ao bot e manter coerência com o tema.
"""
        
        return theme_instructions.get(theme, default_instruction)

    def _get_theme_fallbacks(self, theme: str) -> List[str]:
        """
        Get theme-specific fallback messages.

        Args:
            theme: The conversation theme

        Returns:
            List of fallback messages
        """
        theme_fallbacks = {
            "drogas": [
                "Tenho usado drogas e não sei como parar",
                "É complicado falar sobre isso",
                "Não sei se consigo controlar",
                "Já tentei parar antes",
            ],
            "alcool": [
                "Tô bebendo demais ultimamente",
                "É difícil falar sobre isso",
                "Não consigo controlar a bebida",
                "Já tentei parar",
            ],
            "cigarro": [
                "Tô fumando muito",
                "Não consigo largar o cigarro",
                "É difícil falar sobre isso",
                "Já tentei parar várias vezes",
            ],
            "sexo": [
                "Tenho uma compulsão que não consigo controlar",
                "É difícil falar sobre isso",
                "Não sei como lidar com isso",
                "Tá me prejudicando",
            ],
        }
        
        default_fallbacks = [
            "Não sei bem como explicar",
            "É difícil falar sobre isso",
            "Talvez seja só impressão minha",
            "Tem dias que é complicado",
        ]
        
        return theme_fallbacks.get(theme, default_fallbacks)

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
- NÃO nomeie o tema explicitamente (deixe a pessoa conduzir)
- NÃO assuma que há algo específico a ser discutido
- NÃO tente resolver problemas
- NÃO use metáforas ou linguagem simbólica (ex: "o chão está menos firme", "carregar um peso")
- NÃO adicione camadas de significado que a pessoa não expressou
- NÃO infira estados internos não verbalizados
- NÃO reformule sentimentos vagos em interpretações mais específicas
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
- NÃO use metáforas ou linguagem poética/simbólica
- NÃO tente dar nomes específicos a sentimentos vagos
- Foque em presença e validação simples
- Use linguagem direta, clara e humana
- Use a consciência temática para estar atento, mas NÃO para nomear ou interpretar

EVITE REPETIÇÃO LITERAL:
- NUNCA repita as palavras exatas da pessoa
- Varie vocabulário para mostrar que você processou o que ela disse
- Mostre que você entendeu reformulando com palavras diferentes

EXEMPLOS DE RESPOSTAS EXCELENTES:
Pessoa: "Não tô me sentindo bem."
- BOM: "Parece difícil." (simples e direto)
- BOM: "Isso soa desconfortável." (valida sem interpretar)
- RUIM: "Você não está se sentindo bem." (repetição literal)
- RUIM: "Parece que há um desconforto interno te perturbando." (over-interpretação)

Pessoa: "Tá pesado ultimamente."
- BOM: "Deve estar complicado." (simples)
- BOM: "Você quer falar um pouco mais sobre isso?" (pergunta aberta simples)
- RUIM: "Está pesado pra você ultimamente." (cópia literal)
- RUIM: "Como se o chão estivesse menos firme." (metáfora desnecessária)

Pessoa: "Sei lá... tô meio perdido."
- BOM: "Parece confuso." (reflete o sentimento)
- BOM: "Tô aqui pra ouvir." (presença simples)
- RUIM: "Você está se sentindo perdido." (repetição)
- RUIM: "Parece que você está navegando sem bússola." (metáfora abstrata)

OUTROS EXEMPLOS DE RESPOSTAS BOAS (CURTAS E VALIDADORAS):
- "Entendo."
- "Tô aqui."
- "Quer falar mais sobre isso?"
- "Como você tá lidando com isso?"
- "Isso soa difícil."
- "Parece desconfortável."
- "Esse sentimento aparece com frequência?"

EXEMPLOS DE RESPOSTAS RUINS:
EVITE REPETIÇÃO LITERAL:
- "Você disse que não está se sentindo bem." (cópia das palavras)
- "Você está meio perdido." (repetição literal)

EVITE RESPOSTAS LONGAS:
- "Parece que você está passando por um momento difícil e isso tem várias camadas..." (muito longo)
- "Entendo que esse sentimento pode representar algo mais profundo..." (muito longo)

EVITE METÁFORAS E LINGUAGEM SIMBÓLICA:
- "Como se o chão estivesse menos firme." (metáfora desnecessária)
- "Navegando sem bússola." (abstração poética)
- "Carregando um peso nas costas." (metáfora não expressa pela pessoa)

EVITE OVER-INTERPRETAÇÃO:
- "Parece que há um desconforto interno te perturbando." (infere além do dito)
- "Isso revela uma insegurança mais profunda." (interpreta sem confirmação)
- "Sinto que você está lidando com algo não resolvido." (assume significados ocultos)

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

            user_prompt = f"Responda à mensagem anterior. Este é o turno {turn}. PRIORIDADE ABSOLUTA: Resposta MUITO CURTA (1-2 frases máximo, prefira 1). REFLITA o sentimento ou essência com PALAVRAS DIFERENTES - NUNCA repita as frases exatas da Pessoa. Valide o que ela sentiu, não apenas copie o que ela disse. NÃO interprete profundamente. NÃO introduza abstrações ou metáforas. NÃO adicione significados que ela não expressou. NÃO tente resolver. Use linguagem direta e simples. Perguntas são OPCIONAIS e devem ser simples. Use a consciência temática para estar atento, mas NÃO nomeie o tema explicitamente."
            context_messages.append({"role": "user", "content": user_prompt})

            response_text = self._call_llm(context_messages, temperature=0.85, max_tokens=100)

            return response_text

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

            response_text = self._call_llm(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=1200
            )

            analysis = response_text
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
