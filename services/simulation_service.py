"""
Simulation service for creating AI-driven conversation simulations.

This service orchestrates simulated conversations between two AI roles
and manages emotional analysis of the conversation.
"""

import logging
import os
import random
from typing import List

import requests
from faker import Faker

from core.models import Message, Profile

logger = logging.getLogger(__name__)

# Initialize Faker once at module level for efficiency
_faker = Faker("pt_BR")

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
        self.base_url = os.environ.get("OLLAMA_BASE_URL")
        self.model = os.environ.get("OLLAMA_MODEL_EMBEDDING")
        self.api_url = f"{self.base_url}/api/chat"

        logger.info(
            f"Initialized SimulationService with model={self.model}, base_url={self.base_url}"
        )

    def _call_llm(
        self, messages: List[dict], temperature: float = 0.85, max_tokens: int = 250
    ) -> str:
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

        logger.info(
            f"Created simulation profile: {profile.id} with name: {sim_name}, gender: {gender}, theme: {theme}"
        )
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

        logger.info(
            f"Generating {num_messages} simulated messages for profile {profile.id} with theme: {theme}"
        )

        conversation = []
        conversation_history = []

        for i in range(num_messages):
            # Alternate between roles
            if i % 2 == 0:
                # ROLE_A: Seeker
                role = "ROLE_A"
                message = self._generate_seeker_message(
                    conversation_history, i // 2 + 1, theme
                )
            else:
                # ROLE_B: Listener
                role = "ROLE_B"
                message = self._generate_listener_message(
                    conversation_history, i // 2 + 1, theme
                )

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
            # Determine emotional state based on turn and random variation
            emotional_state = self._get_emotional_state(turn)
            
            # Build theme-driven prompt
            theme_instruction = self._build_theme_instruction(theme)
            
            # Build emotional state instruction
            state_instruction = self._build_emotional_state_instruction(emotional_state)

            system_prompt = f"""Você está simulando um usuário humano real com problemas reais.
Você NÃO é um assistente.
Você é uma pessoa imperfeita, confusa, cheia de contradições.
Você NÃO sabe que está sendo simulado.

Tema: {theme or 'desabafar'}

{theme_instruction}

ESTADO EMOCIONAL ATUAL: {emotional_state}
{state_instruction}

REGRAS DE MENSAGENS HUMANAS E IMPERFEITAS:
- Linguagem em primeira pessoa ("eu", "estou", "me sinto")
- Mensagens curtas (1-3 frases), mas VARIADAS em estrutura
- Misture reflexão, confusão, resistência e vulnerabilidade
- MOSTRE CONTRADIÇÃO INTERNA - você pode dizer uma coisa e depois outra que contradiz
- Às vezes hesite, se corrija, ou interrompa o pensamento
- Pode repetir IDEIAS, mas NUNCA repita a MESMA estrutura de frase
- Seja ambivalente - "eu sei que... mas..."
- Expresse frustração consigo mesmo
- NÃO seja terapêutico ou analítico
- NÃO soe como um prompt escrito para LLM
- Fale como uma pessoa comum em dificuldade

COMPORTAMENTO NATURAL:
- Reaja emocionalmente ao que o assistente disse
- Não explique demais
- Deixe coisas implícitas
- Use linguagem informal do dia a dia
- Pode ser defensivo, depois vulnerável
- Pode minimizar, depois admitir
- Pode expressar raiva, cansaço, ou confusão
- Às vezes só desabafa sem pedir ajuda

VARIAÇÃO DE ESTRUTURA (CRÍTICO):
NUNCA use a mesma estrutura duas vezes seguidas:
- "É como se..." (só use uma vez em toda a conversa)
- "Parece que..." (só use uma vez em toda a conversa)
- "Tem dias que..." (varie com "às vezes", "quando", "toda vez que")
- Misture afirmações diretas com frases complexas
- Use diferentes conectores: mas, e, porque, quando, aí

EXEMPLOS DO QUE FAZER:
❌ NÃO: "É como se eu não tivesse escolha."
✅ SIM: "Eu sei que isso tá me destruindo, mas quando a vontade vem eu simplesmente vou."

❌ NÃO: "Parece que não consigo parar."
✅ SIM: "Tem dias que eu penso que consigo parar, mas aí alguma coisa acontece e eu volto."

❌ NÃO: "Isso soa difícil."
✅ SIM: "Eu fico com raiva de mim mesmo depois, mas na hora parece que nada mais importa."

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
                # First message - must introduce theme naturally with emotional complexity
                substance_themes = ["drogas", "alcool", "cigarro", "sexo"]
                if theme in substance_themes:
                    theme_examples = {
                        "drogas": "tenho usado e tá foda', 'voltei e não era pra ter voltado', 'quando vem a vontade eu vou",
                        "alcool": "tô bebendo demais de novo', 'não era pra estar assim mas tô', 'a bebida voltou e eu deixei",
                        "cigarro": "tô fumando pra caralho', 'não consigo ficar sem', 'o cigarro voltou com tudo",
                        "sexo": "não consigo controlar isso', 'tá me consumindo', 'é compulsivo e eu sei",
                    }
                    examples = theme_examples.get(theme, "")
                    user_prompt = f"Envie sua PRIMEIRA mensagem. Mencione sua luta com {theme} de forma emocional, imperfeita e pessoal. Use contraste ou contradição (ex: 'eu sei que...mas...', 'devia parar...só que...'). Seja breve (1-3 frases) mas HUMANO. Exemplos de frases: '{examples}'. NÃO use estruturas como 'É como se' ou 'Parece que'."
                else:
                    user_prompt = "Envie sua PRIMEIRA mensagem. Introduza o tema com contradição ou ambivalência. Seja breve (1-3 frases) mas mostre conflito interno. NÃO use 'É como se' ou 'Parece que'."
            elif turn == 2:
                user_prompt = f"Responda ao assistente. Estado: {emotional_state}. Reaja emocionalmente. Pode ser defensivo ou minimizar. Seja breve mas VARIE a estrutura da mensagem anterior. NÃO repita padrões como 'É como se' ou 'Parece que'."
            elif turn == 3:
                user_prompt = f"Estado: {emotional_state}. Pode começar a se abrir mais ou mostrar mais vulnerabilidade/resistência. VARIE completamente a estrutura. Pode admitir algo difícil ou contradizer o que disse antes. 1-3 frases."
            elif turn == 4:
                user_prompt = f"Estado: {emotional_state}. Aprofunde o conflito interno. Pode expressar frustração consigo mesmo ou cansaço. NUNCA repita estruturas anteriores. Use linguagem direta e crua."
            else:
                user_prompt = f"Estado: {emotional_state}. Continue desenvolvendo. Pode mostrar mudança de perspectiva, exaustão, ou momento de lucidez dolorosa. SEMPRE varie a estrutura. Seja genuinamente humano."

            context_messages.append({"role": "user", "content": user_prompt})

            # Use higher temperature for more natural variation
            response_text = self._call_llm(
                context_messages, temperature=0.95, max_tokens=250
            )

            return response_text

        except Exception as e:
            logger.error(f"Error generating seeker message: {str(e)}", exc_info=True)
            # Fallback messages based on theme and emotional state
            fallbacks = self._get_theme_fallbacks(theme)
            return fallbacks[turn % len(fallbacks)]
    
    def _get_emotional_state(self, turn: int) -> str:
        """
        Determine emotional state based on turn number with some randomness.
        
        States: CONFUSION, LOSS_OF_CONTROL, RESISTANCE, SHAME, EXHAUSTION, AMBIVALENCE
        
        Args:
            turn: Current turn number
            
        Returns:
            Emotional state string
        """
        # Add some randomness but with progression
        base_states = {
            1: ["CONFUSION", "AMBIVALENCE"],
            2: ["LOSS_OF_CONTROL", "RESISTANCE", "AMBIVALENCE"],
            3: ["RESISTANCE", "SHAME", "LOSS_OF_CONTROL"],
            4: ["SHAME", "EXHAUSTION", "AMBIVALENCE"],
            5: ["EXHAUSTION", "CONFUSION", "AMBIVALENCE"]
        }
        
        # Get possible states for this turn, with fallback for turns > 5
        possible_states = base_states.get(turn, ["AMBIVALENCE", "EXHAUSTION", "CONFUSION"])
        
        # Add 30% chance to pick from any state for variety
        if random.random() < 0.3:
            all_states = ["CONFUSION", "LOSS_OF_CONTROL", "RESISTANCE", "SHAME", "EXHAUSTION", "AMBIVALENCE"]
            return random.choice(all_states)
        
        return random.choice(possible_states)
    
    def _build_emotional_state_instruction(self, state: str) -> str:
        """
        Build instruction for specific emotional state.
        
        Args:
            state: Emotional state name
            
        Returns:
            Instruction text for this state
        """
        state_instructions = {
            "CONFUSION": """
Você está confuso sobre o que sente ou o que fazer.
- Expresse incerteza: "não sei se...", "será que...", "às vezes eu acho que..."
- Contradiga-se: diga uma coisa, depois questione
- Mostre que as coisas não fazem sentido pra você
- Pode questionar suas próprias percepções
""",
            "LOSS_OF_CONTROL": """
Você sente que perdeu o controle sobre a situação.
- Expresse impotência: "quando vem a vontade eu vou", "não consigo parar"
- Mostre que há uma força maior que sua vontade
- Pode soar derrotado mas ainda lutando
- Use palavras que mostram automático: "simplesmente acontece", "antes que eu perceba"
""",
            "RESISTANCE": """
Você está resistindo - à mudança, à ajuda, à realidade.
- Pode minimizar: "não é tão grave assim", "todo mundo faz"
- Pode ser defensivo: "eu sei controlar", "não é sempre"
- Depois pode contradizer sua própria defesa
- Mostre que você quer e não quer ao mesmo tempo
""",
            "SHAME": """
Você sente vergonha do que faz ou de si mesmo.
- Expresse auto-julgamento: "eu sou fraco", "que merda que eu sou"
- Mostre frustração consigo mesmo: "eu fico com raiva de mim"
- Pode ser duro consigo mesmo
- Depois do auto-julgamento pode voltar a justificar
""",
            "EXHAUSTION": """
Você está cansado de lutar, de tentar, de tudo.
- Expresse cansaço: "tô cansado disso", "não aguento mais"
- Mostre desistência momentânea: "foda-se", "tanto faz"
- Pode soar sem energia ou esperança
- Não busca soluções, só expressa
""",
            "AMBIVALENCE": """
Você quer duas coisas opostas ao mesmo tempo.
- Use "mas" constantemente: "eu quero parar, mas..."
- Mostre o conflito interno explicitamente
- Reconheça os dois lados: "eu sei que é ruim, mas me ajuda"
- Não resolve a contradição, apenas a expõe
"""
        }
        
        return state_instructions.get(state, state_instructions["AMBIVALENCE"])

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
- Mostrar CONTRADIÇÃO ou AMBIVALÊNCIA
- Usar padrões de conflito interno:
  * "eu sei que tá me destruindo, mas..."
  * "voltei a usar, eu jurei que não ia..."
  * "quando vem a vontade eu simplesmente vou"
  * "eu fico com raiva de mim depois, mas na hora..."

NÃO use frases genéricas como "tenho usado drogas".
Mensagens subsequentes:
- VARIE completamente a estrutura a cada mensagem
- Mostre escalada ou mudança de perspectiva
- Expresse frustração, vergonha, exaustão, ou confusão
- Mantenha contradição e ambivalência
""",
            "alcool": """
REGRAS ESPECÍFICAS DO TEMA (alcool):
Sua PRIMEIRA mensagem DEVE:
- Mencionar explicitamente sua luta com álcool
- Mostrar CONTRADIÇÃO ou conflito interno
- Usar padrões como:
  * "eu sei que tô bebendo demais, mas é o que me acalma"
  * "toda vez que digo que vou parar, aí chega o fim de semana..."
  * "fico me prometendo que vai ser a última vez, mas nunca é"

Mensagens subsequentes:
- VARIE estruturas - nunca repita o mesmo padrão
- Mostre momentos de lucidez e recaída
- Expresse cansaço, raiva de si, ou resignação
""",
            "cigarro": """
REGRAS ESPECÍFICAS DO TEMA (cigarro):
Sua PRIMEIRA mensagem DEVE:
- Mencionar explicitamente sua luta com cigarro/fumo
- Mostrar frustração com recaídas ou impotência
- Usar padrões como:
  * "larguei por [tempo], aí voltei pior que antes"
  * "eu sei que tô me matando aos poucos, mas..."
  * "falo que vou parar amanhã faz tipo um ano já"

Mensagens subsequentes:
- VARIE completamente - use diferentes estruturas
- Mostre tentativas fracassadas de parar
- Expresse vergonha, raiva, ou aceitação resignada
""",
            "sexo": """
REGRAS ESPECÍFICAS DO TEMA (sexo):
Sua PRIMEIRA mensagem DEVE:
- Mencionar luta com compulsão ou comportamento sexual
- Mostrar vergonha MAS também perda de controle
- Usar padrões como:
  * "eu sei que é compulsivo, mas quando vem..."
  * "depois eu me odeio, mas na hora parece que não tenho escolha"
  * "tá me consumindo e eu sei disso, mas não consigo parar"

Mensagens subsequentes:
- VARIE estruturas drasticamente
- Alterne entre vergonha profunda e admissão de impotência
- Pode ser cru e direto quando apropriado
""",
            "ansiedade": """
REGRAS ESPECÍFICAS DO TEMA (ansiedade):
Sua PRIMEIRA mensagem deve:
- Expressar ansiedade com detalhes físicos ou mentais
- Mostrar como afeta seu dia a dia
- Usar padrões como:
  * "não consigo parar de me preocupar com tudo"
  * "meu peito aperta e eu não sei porquê"
  * "fico ansioso até com coisa boba, é automático"

Mensagens subsequentes:
- Varie estruturas constantemente
- Expresse frustração com a própria mente
- Pode mostrar exaustão mental ou confusão
""",
            "solidao": """
REGRAS ESPECÍFICAS DO TEMA (solidao):
Sua PRIMEIRA mensagem deve:
- Expressar solidão de forma vulnerável mas não melodramática
- Usar padrões como:
  * "não tenho com quem conversar de verdade"
  * "tô rodeado de gente mas me sinto sozinho pra caralho"
  * "faz tempo que eu não falo com alguém assim"

Mensagens subsequentes:
- Varie estruturas - nunca repita padrões
- Pode mostrar vergonha da própria solidão
- Expresse necessidade mas também resistência
""",
        }

        default_instruction = """
REGRAS ESPECÍFICAS DO TEMA (geral):
Sua PRIMEIRA mensagem deve introduzir sua preocupação com contradição interna.
Use "eu sei que... mas...", "devia... só que...", ou similar.
Mostre conflito entre o que você sabe e o que você faz/sente.
Mensagens subsequentes devem VARIAR completamente em estrutura e mostrar evolução emocional.
"""

        return theme_instructions.get(theme, default_instruction)

    def _get_theme_fallbacks(self, theme: str) -> List[str]:
        """
        Get theme-specific fallback messages with emotional complexity.

        Args:
            theme: The conversation theme

        Returns:
            List of fallback messages
        """
        theme_fallbacks = {
            "drogas": [
                "Eu sei que tá me destruindo, mas quando a vontade vem eu simplesmente vou",
                "Voltei a usar, eu jurei que não ia mas voltei",
                "Eu fico com raiva de mim depois, mas na hora nada mais importa",
                "Tem dias que eu acho que consigo parar, aí alguma coisa acontece e eu volto",
                "Não era pra eu estar usando de novo, mas tô",
            ],
            "alcool": [
                "Eu sei que tô bebendo demais, mas é o que me acalma agora",
                "Toda vez que digo que vou parar, aí chega o fim de semana e foda-se",
                "Fico me prometendo que vai ser a última vez, mas nunca é",
                "Tô bebendo todo dia agora, não era pra ser assim",
                "Eu sei que preciso parar, só que quando bate a ansiedade eu vou",
            ],
            "cigarro": [
                "Larguei por dois meses, aí voltei pior que antes",
                "Eu sei que tô me matando aos poucos, mas não consigo ficar sem",
                "Falo que vou parar amanhã faz tipo um ano já",
                "Tô fumando um maço por dia agora, antes era menos",
                "Tento parar mas quando bate o stress eu já acendo",
            ],
            "sexo": [
                "Eu sei que é compulsivo, mas quando vem eu não consigo controlar",
                "Depois eu me odeio, mas na hora parece que eu não tenho escolha",
                "Tá me consumindo e eu sei disso, mas não consigo parar",
                "Eu prometo pra mim mesmo que não vou mais, mas aí acontece de novo",
                "É automático, eu nem penso direito",
            ],
        }

        default_fallbacks = [
            "Eu sei que devia fazer diferente, mas quando chega a hora eu não faço",
            "Tem dias que eu acho que consigo, tem dias que eu desisto",
            "Eu fico com raiva de mim depois, mas isso não muda nada",
            "Não sei se faz sentido o que tô falando",
            "Tô cansado de tentar e falhar",
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

            response_text = self._call_llm(
                context_messages, temperature=0.85, max_tokens=100
            )

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

    def analyze_conversation_emotions(self, conversation: List[dict]) -> str:
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
                role_label = (
                    ROLE_LABEL_SEEKER
                    if msg["role"] == "ROLE_A"
                    else ROLE_LABEL_LISTENER
                )
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
                max_tokens=1200,
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
