# WaChat

✨ O que você é

Um espaço seguro de escuta espiritual, com reflexões cristãs e respostas personalizadas via Telegram, sem julgamento.

Frase-guia interna

“Não te digo o que pensar. Caminho contigo enquanto você pensa.”



postgres hosted at: https://supabase.com/dashboard/project/

WaChat é um projeto Django para aplicação de chat via Telegram.

cloudflared tunnel --url http://localhost:9000

## 📋 Requisitos

- Python 3.8+
- PostgreSQL (ou SQLite para desenvolvimento)

## 🚀 Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/avictorino/wachat.git
cd wachat
```

### 2. Crie e ative um ambiente virtual

```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Configure as variáveis de ambiente

Copie o arquivo `.env.example` para `.env` e ajuste as configurações:

```bash
cp .env.example .env
```

Edite o arquivo `.env` com suas configurações:

```env
# Django Settings
SECRET_KEY=sua-chave-secreta-aqui
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

# Database Configuration
# Para PostgreSQL:
DATABASE_URL=postgresql://usuario:senha@localhost:5432/nome_do_banco

# Para SQLite (desenvolvimento):
# DATABASE_URL=sqlite:///db.sqlite3
```

**Importante:** O arquivo `.env` não será versionado (já está no `.gitignore`) para proteger suas credenciais.

### 5. Execute as migrações

```bash
python manage.py migrate
```

### 6. Crie um superusuário (opcional)

```bash
python manage.py createsuperuser
```

### 7. Inicie o servidor de desenvolvimento

```bash
python manage.py runserver
```

Acesse a aplicação em: http://localhost:8000

## 🗄️ Configuração do Banco de Dados

Este projeto utiliza `dj-database-url` para simplificar a configuração do banco de dados através de uma URL string.

### PostgreSQL (Produção/Desenvolvimento)

No arquivo `.env`, configure a variável `DATABASE_URL`:

```env
DATABASE_URL=postgresql://usuario:senha@host:porta/nome_do_banco
```

Exemplos:
- Local: `postgresql://postgres:senha123@localhost:5432/wachat`
- Heroku: `postgresql://user:pass@ec2-xxx.compute.amazonaws.com:5432/dbname`

### SQLite (Desenvolvimento Local)

Se preferir usar SQLite para desenvolvimento local, deixe a variável `DATABASE_URL` vazia ou use:

```env
DATABASE_URL=sqlite:///db.sqlite3
```

## 📦 Dependências

- **Django 4.2.27**: Framework web principal
- **dj-database-url**: Configuração de banco de dados via URL
- **python-decouple**: Gerenciamento de variáveis de ambiente
- **django-dotenv**: Carregamento automático de variáveis do arquivo .env
- **psycopg2-binary**: Driver PostgreSQL

## 🛠️ Desenvolvimento

### Estrutura do Projeto

```
wachat/
├── core/               # App principal
├── config/            # Configurações do projeto
│   ├── settings.py    # Configurações Django
│   ├── urls.py        # URLs principais
│   └── wsgi.py        # WSGI config
├── manage.py          # Utilitário Django
├── requirements.txt   # Dependências Python
├── .env.example       # Exemplo de variáveis de ambiente
└── README.md         # Este arquivo
```

### Comandos Úteis

```bash
# Criar migrações
python manage.py makemigrations

# Aplicar migrações
python manage.py migrate

# Rodar servidor
python manage.py runserver

# Criar superusuário
python manage.py createsuperuser

# Rodar testes
python manage.py test

# Coletar arquivos estáticos
python manage.py collectstatic

# Simular conversa realista entre humano e bot
python manage.py simulate_conversation --turns 5 --domain spiritual

# Simular conversa entre dois agentes de IA (buscador e ouvinte)
python manage.py simulate --num-messages 8
```

## 🤖 Simulação de Conversas

O projeto inclui dois comandos de gerenciamento para simular conversas:

### `simulate_conversation` - Simulação Realista com Humano

Simula uma conversa realista entre um usuário humano (simulado por IA) e o bot. Isso é útil para:

- Testar o fluxo completo de conversação
- Validar progressão do funil e gerenciamento de estado
- Gerar dados de teste para desenvolvimento
- Demonstrar capacidades conversacionais do bot

```bash
# Simulação básica com 5 turnos
python manage.py simulate_conversation

# Simulação personalizada
python manage.py simulate_conversation --turns 10 --domain grief --name "Ana Costa"

# Modo de teste (sem chamadas reais de API)
python manage.py simulate_conversation --mock-telegram --turns 3
```

Para documentação completa, veja [docs/SIMULATE_CONVERSATION.md](docs/SIMULATE_CONVERSATION.md).

### `simulate` - Simulação entre Dois Agentes de IA

Simula uma conversa entre dois agentes de IA (buscador e ouvinte) e fornece análise crítica. Útil para:

- Testar a qualidade do diálogo do bot
- Avaliar a empatia e resposta do ouvinte
- Analisar verbosidade e interpretação das respostas
- Gerar exemplos de conversas para treinamento

```bash
# Simulação básica com 8 mensagens
python manage.py simulate

# Simulação com número personalizado de mensagens (6-10)
python manage.py simulate --num-messages 10

# Modo silencioso (apenas a conversa e análise)
python manage.py simulate --quiet
```

O comando gera uma conversa alternada entre:
- 🧑‍💬 **Pessoa** (ROLE_A): pessoa em busca espiritual, vulnerável e cautelosa
- 🌿 **BOT** (ROLE_B): assistente empático e não-julgador

Ao final, exibe uma análise crítica em 5 seções:
1. O que funcionou bem
2. Pontos de possível erro de interpretação
3. Problemas de verbosidade e extensão das respostas
4. O que poderia ter sido feito diferente
5. Ajustes recomendados para próximas interações

## 💬 Comandos do Telegram Bot

Os seguintes comandos estão disponíveis no bot do Telegram:

### `/start`
Inicia uma nova conversa com o bot. Cria um perfil de usuário, infere gênero a partir do nome e envia uma mensagem de boas-vindas personalizada.

### `/reset`
Inicia o processo de exclusão de dados do usuário. Solicita confirmação antes de deletar permanentemente o perfil, conversas e mensagens. O usuário deve responder com "CONFIRM" dentro de 5 minutos.

### `/simulate [número]`
**Novo!** Executa uma simulação completa de conversa entre dois papéis de IA:
- 🧑‍💬 **Pessoa**: Uma pessoa em busca espiritual, vulnerável e questionadora
- 🌿 **BOT**: Um assistente espiritual empático e não-julgador

**Uso:**
- `/simulate` - Gera 8 mensagens (padrão)
- `/simulate 6` - Gera 6 mensagens (mínimo)
- `/simulate 10` - Gera 10 mensagens (máximo)

O comando gera o número especificado de mensagens alternadas (6-10, padrão 8), persiste tudo no banco de dados, e retorna:
1. Cada mensagem da conversa simulada com identificação de papel
2. Uma análise crítica final da conversa, incluindo:
   - O que funcionou bem
   - Pontos de possível erro de interpretação
   - Problemas de verbosidade e extensão das respostas
   - O que poderia ter sido feito diferente
   - Ajustes recomendados para próximas interações

**Útil para:**
- Demonstrar as capacidades do bot
- Testar o fluxo conversacional
- Visualizar análise crítica em ação
- Gerar exemplos de conversas

## 📝 Variáveis de Ambiente

| Variável | Descrição | Padrão | Obrigatória |
|----------|-----------|--------|-------------|
| `SECRET_KEY` | Chave secreta do Django | - | Sim (produção) |
| `DEBUG` | Modo de debug | `True` | Não |
| `ALLOWED_HOSTS` | Hosts permitidos (separados por vírgula) | - | Sim (produção) |
| `DATABASE_URL` | URL de conexão com o banco de dados | SQLite local | Não |
| `LLM_PROVIDER` | Provedor de LLM (`groq` ou `ollama`) | `groq` | Não |
| `GROQ_API_KEY` | Chave da API Groq | - | Sim (se LLM_PROVIDER=groq) |
| `OLLAMA_BASE_URL` | URL base do servidor Ollama local | `http://localhost:11434` | Não |
| `OLLAMA_MODEL` | Modelo Ollama a ser usado | `llama3.1` | Não |

## 🤖 Configuração do Provedor de LLM

O WaChat suporta dois provedores de LLM (Large Language Model):

### 1. Groq (Padrão - Cloud API)

O Groq é o provedor padrão e utiliza a API cloud da Groq.

**Configuração:**
```env
LLM_PROVIDER=groq
GROQ_API_KEY=sua-chave-api-groq
```

**Prós:**
- Setup simples (apenas API key)
- Alta performance
- Sem necessidade de hardware local

**Contras:**
- Requer chave de API
- Custos por uso (dependendo do plano)
- Requer conexão com internet

### 2. Ollama (Local)

O Ollama permite executar modelos LLM localmente, sem dependência de APIs externas.

**Configuração:**

1. **Instale o Ollama:**
   ```bash
   # Linux/macOS
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Ou visite: https://ollama.com/download
   ```

2. **Baixe um modelo:**
   ```bash
   # Recomendado: llama3.1 (modelo padrão)
   ollama pull llama3.1
   
   # Ou outros modelos:
   # ollama pull llama3
   # ollama pull mistral
   # ollama pull codellama
   ```

3. **Crie um modelo customizado a partir do Modelfile (Opcional):**
   
   O WaChat inclui um `Modelfile` na raiz do projeto que define o comportamento
   conversacional base do assistente. Para usar o Ollama com este comportamento
   customizado, crie um modelo do Ollama a partir do Modelfile:
   
   ```bash
   # Na raiz do projeto wachat
   ollama create wachat -f Modelfile
   
   # Configure o modelo no .env
   OLLAMA_MODEL=wachat
   ```
   
   **Nota:** Esta etapa é **recomendada** para melhor experiência com Ollama.
   O Modelfile define o comportamento conversacional completo do assistente, incluindo
   tom, regras de conversação e postura espiritual. O código da aplicação envia apenas
   instruções dinâmicas e contextuais (temas e modos de resposta).

4. **Inicie o servidor Ollama:**
   ```bash
   ollama serve
   # O servidor será iniciado em http://localhost:11434
   ```

5. **Configure as variáveis de ambiente:**
   ```env
   LLM_PROVIDER=ollama
   OLLAMA_BASE_URL=http://localhost:11434  # Padrão, pode ser omitido
   OLLAMA_MODEL=llama3.1                   # Padrão, pode ser omitido (ou use 'wachat' se criou o modelo customizado)
   ```

6. **Inicie o WaChat:**
   ```bash
   python manage.py runserver
   ```

**Prós:**
- Totalmente local (sem custos de API)
- Privacidade completa dos dados
- Sem limitações de tokens
- Funciona offline

**Contras:**
- Requer hardware adequado (GPU recomendada)
- Setup inicial mais complexo
- Pode ser mais lento que APIs cloud

**Modelos Recomendados:**
- `llama3.1` (padrão) - Bom equilíbrio entre qualidade e performance
- `llama3` - Alternativa mais leve
- `mistral` - Outra opção de qualidade
- `gemma` - Modelo do Google, também eficiente

**Exemplo de uso com modelo customizado:**
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral  # Usando Mistral em vez do padrão
```

### Alternando entre provedores

Você pode facilmente alternar entre provedores mudando a variável `LLM_PROVIDER`:

```bash
# Usar Groq
export LLM_PROVIDER=groq
export GROQ_API_KEY=sua-chave

# Ou usar Ollama
export LLM_PROVIDER=ollama
```

A aplicação detectará automaticamente o provedor configurado e utilizará o serviço apropriado sem necessidade de mudanças no código.

## 🔒 Segurança

- **NUNCA** commite o arquivo `.env` no repositório
- Mantenha o `SECRET_KEY` seguro e único por ambiente
- Em produção, sempre configure `DEBUG=False`
- Configure `ALLOWED_HOSTS` apropriadamente em produção

## 🚀 Deploy para Heroku

Para instruções completas de deployment no Heroku, consulte o [Guia de Deploy para Heroku](HEROKU_DEPLOYMENT.md).

O deploy no Heroku inclui:
- Configuração automática de PostgreSQL
- Execução automática de migrations durante o deploy
- Sincronização com o branch `main` do GitHub
- Python buildpack configurado
- Gunicorn como servidor WSGI

## 📄 Licença

[Especifique a licença do projeto aqui]

## 👥 Contribuição

[Instruções para contribuir com o projeto]

## 📧 Contato

[Informações de contato]
