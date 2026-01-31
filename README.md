# WaChat


postgres hosted at: https://supabase.com/dashboard/project/jratzhwgcwawfefwuqye

WaChat é um projeto Django para aplicação de chat.

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
```

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
