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
- 🧑‍💬 **Buscador** (ROLE_A): pessoa em busca espiritual, vulnerável e cautelosa
- 🌿 **Ouvinte** (ROLE_B): assistente empático e não-julgador

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

### `/simulate`
**Novo!** Executa uma simulação completa de conversa entre dois papéis de IA:
- 🧑‍💬 **Buscador**: Uma pessoa em busca espiritual, vulnerável e questionadora
- 🌿 **Ouvinte**: Um assistente espiritual empático e não-julgador

O comando gera 6-10 mensagens alternadas, persiste tudo no banco de dados, e retorna:
1. Cada mensagem da conversa simulada com identificação de papel
2. Uma análise emocional final da conversa, incluindo:
   - Tom emocional predominante
   - Emoções dominantes detectadas
   - Evolução emocional ao longo da conversa
   - Qualidade geral da interação

**Útil para:**
- Demonstrar as capacidades do bot
- Testar o fluxo conversacional
- Visualizar análise emocional em ação
- Gerar exemplos de conversas

## 📝 Variáveis de Ambiente

| Variável | Descrição | Padrão | Obrigatória |
|----------|-----------|--------|-------------|
| `SECRET_KEY` | Chave secreta do Django | - | Sim (produção) |
| `DEBUG` | Modo de debug | `True` | Não |
| `ALLOWED_HOSTS` | Hosts permitidos (separados por vírgula) | - | Sim (produção) |
| `DATABASE_URL` | URL de conexão com o banco de dados | SQLite local | Não |

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
