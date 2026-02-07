# RAG (Retrieval-Augmented Generation) System

## Overview

This system implements a Django-native RAG pipeline integrated with Ollama for semantic search and context-aware response generation.

## Architecture

### Components

1. **RagChunk Model** (`core/models.py`)
   - Django model for storing text chunks with embeddings
   - Fields: id, source, page, chunk_index, text, embedding (768-dim vector), type
   - Indexes: HNSW on embedding (cosine distance), composite on (source, page)
   - Types: "behavior" (guidance/posture) or "content" (informational)

2. **Ingestion Script** (`core/management/commands/ingest_pdf_for_ollama.py`)
   - Processes PDFs from `model/pdfs/` directory
   - Extracts text blocks, chunks with sentence overlap
   - Classifies chunks as behavior or content based on keywords
   - Generates embeddings using Ollama (nomic-embed-text)
   - Stores in database using Django ORM bulk_create

3. **RAG Service** (`services/rag_service.py`)
   - `get_rag_context(user_input, limit=5)`: Retrieves relevant chunks
   - Generates query embedding
   - Searches using cosine similarity
   - Prefers behavior chunks over content chunks
   - Returns only raw text strings (no metadata)

4. **Response Generation** (`services/ollama_service.py`)
   - Integrates RAG context into system prompts
   - RAG content is injected as "silent context"
   - Model behavior comes from Ollama Modelfile
   - No explicit mention of RAG to the user

## Usage

### 1. Ingest PDFs

```bash
# Ingest PDFs without embeddings (faster, for testing structure)
python manage.py ingest_pdf_for_ollama

# Ingest PDFs with embeddings (production)
python manage.py ingest_pdf_for_ollama --embed --ollama-url http://localhost:11434

# Custom chunk size
python manage.py ingest_pdf_for_ollama --embed --chunk-size 900
```

### 2. Query RAG Context

```python
from services.rag_service import get_rag_context

# Get relevant chunks for a user message
context_texts = get_rag_context("Estou me sentindo ansioso", limit=5)
# Returns: List of 5 most relevant text chunks
```

### 3. Generate Responses

Response generation automatically integrates RAG:

```python
from services.ollama_service import OllamaService

service = OllamaService()
messages = service.generate_fallback_response(
    user_message="Estou com medo",
    conversation_context=[...],
    name="João"
)
# RAG context is automatically retrieved and injected
```

## Database Schema

```sql
CREATE TABLE core_ragchunk (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    page INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding vector(768),
    type VARCHAR(20) DEFAULT 'content'
);

CREATE INDEX core_ragchunk_embedding_idx 
ON core_ragchunk USING hnsw (embedding vector_cosine_ops);

CREATE INDEX rag_source_page_idx 
ON core_ragchunk (source, page);
```

## Chunk Classification

Chunks are automatically classified based on keyword analysis:

### Behavior Keywords
- postura, tom, cuidado, relacionamento
- acolhimento, empatia, escuta, presença
- orientação, guia, direção, conselho
- apoio, sustentação, companhia
- atenção, sensibilidade, discernimento
- pastoral, ministério

**Rule**: If 2+ behavior keywords are found, chunk is classified as "behavior"

### Default
All other chunks are classified as "content"

## RAG Retrieval Strategy

1. Generate embedding for user input
2. Query database ordered by cosine distance
3. Fetch 2× limit chunks initially
4. Split into behavior and content chunks
5. Prefer behavior chunks (fill up to limit)
6. Fill remaining slots with content chunks
7. Return only text (no metadata)

## Integration with Ollama

### System Prompt Structure

```
CONTEXTO DE REFERÊNCIA (use de forma natural e implícita):
[RAG chunk 1 text]

[RAG chunk 2 text]
...
---

Nome da pessoa: [name]
Gênero inferido: [gender]
Esta é uma continuação natural da conversa.
```

### Key Principles

1. **Modelfile First**: Behavioral rules come from Ollama Modelfile
2. **RAG as Context**: Retrieved chunks provide silent background knowledge
3. **No Explicit RAG**: User is never told about RAG system
4. **Behavior Priority**: Behavioral guidance preferred over factual content

## Migration from Old System

### What Changed

1. ✅ Raw psycopg → Django ORM
2. ✅ Standalone table → Django model with migrations
3. ✅ Manual SQL → bulk_create/bulk_update
4. ✅ BASE_PROMPT_PTBR → Modelfile + RAG (removed)
5. ✅ PromptComposer → Direct system messages (removed)

### Removed

- `services/prompts/` directory (PromptComposer, themes, and all related code)
- `services/theme_selector.py` (theme selection logic)
- Direct usage of BASE_PROMPT in production code

### What to Keep

- Modelfile defines base behavior
- RAG provides contextual guidance
- System remains simple and explicit

## Testing

### Manual Test Flow

1. **Test Model Creation**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

2. **Test Ingestion**
   ```bash
   # Place a PDF in model/pdfs/
   python manage.py ingest_pdf_for_ollama --embed
   ```

3. **Test RAG Retrieval**
   ```python
   from services.rag_service import get_rag_context
   chunks = get_rag_context("teste", limit=3)
   print(f"Retrieved {len(chunks)} chunks")
   ```

4. **Test Response Generation**
   ```python
   from services.ollama_service import OllamaService
   service = OllamaService()
   response = service.generate_fallback_response(
       user_message="Olá",
       conversation_context=[],
       name="Teste"
   )
   print(response)
   ```

## Environment Variables

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
OLLAMA_EMBED_MODEL=nomic-embed-text
DATABASE_URL=postgresql://...
```

## Performance Considerations

1. **Embedding Generation**: ~100ms per chunk (Ollama)
2. **Batch Size**: 500 chunks per bulk_create
3. **Vector Search**: O(log n) with HNSW index
4. **Query Limit**: 5 chunks default (balance relevance vs context size)

## Future Enhancements

- [ ] Add chunk importance scoring
- [ ] Implement semantic deduplication
- [ ] Add chunk update timestamps
- [ ] Support multiple embedding models
- [ ] Add RAG context quality metrics
