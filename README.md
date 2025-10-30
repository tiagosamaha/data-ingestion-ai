# Desafio MBA Engenharia de Software com IA - Full Cycle

Sistema RAG (Retrieval-Augmented Generation) para ingestão e consulta de documentos PDF utilizando LangChain, PostgreSQL com pgvector e OpenAI.

## Arquitetura

O sistema é composto por três componentes principais:

1. **Ingestão** (`src/ingest.py`): Carrega PDFs, divide em chunks, gera embeddings e armazena no pgvector
2. **Busca** (`src/search.py`): Realiza busca por similaridade e constrói prompts com contexto
3. **Chat** (`src/chat.py`): Interface principal que orquestra a busca e interação com o LLM

## Pré-requisitos

- Docker e Docker Compose
- Chave de API da OpenAI

## Configuração

### 1. Configurar Variáveis de Ambiente

Copie o arquivo de exemplo e configure suas credenciais:

```bash
cp .env.example .env
```

Edite o arquivo `.env` com suas configurações:

```env
# API Key da OpenAI
OPENAI_API_KEY=sua-chave-openai-aqui
```

## Executando o Sistema

### Passo 1: Fazer Ingestão do PDF

O arquivo PDF já está no diretório do projeto (na imagem docker). Execute o comando para realizar a ingestão:

```bash
docker-compose run --rm --entrypoint python chat src/ingest.py
```

### Passo 2: Executar o Chat

```bash
docker-compose run --rm chat
```

As variáveis de ambiente do arquivo `.env` serão automaticamente passadas para o container.

## Usando o Chat

Após iniciar o chat, você verá a seguinte interface:

```
============================================================
Sistema RAG - Chat com Documentos
============================================================
Digite 'sair', 'quit' ou 'exit' para encerrar o chat.

Você:
```

Digite suas perguntas e o sistema responderá com base no conteúdo dos documentos ingeridos.

**Importante**: O sistema segue princípios rigorosos de RAG - as respostas são baseadas exclusivamente no contexto fornecido pelos documentos. Se a informação não estiver disponível no contexto, o sistema responderá: "Não tenho informações necessárias para responder sua pergunta."

## Tecnologias Utilizadas

- **Python 3.13**
- **LangChain 0.3.27**: Framework para aplicações com LLMs
- **PostgreSQL 17 + pgvector**: Banco de dados vetorial
- **OpenAI**: Modelos de linguagem e embeddings
- **pypdf**: Processamento de PDFs
- **Docker**: Containerização

## Estrutura do Projeto

```
.
├── src/
│   ├── chat.py       # Interface de chat
│   ├── ingest.py     # Ingestão de documentos
│   └── search.py     # Busca e construção de prompts
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

## Troubleshooting

### Erro de conexão com o banco de dados

Verifique se o PostgreSQL está rodando:

```bash
docker-compose ps
```

### Erro de API Key

Certifique-se de que configurou corretamente a chave de API da OpenAI no arquivo `.env` e que ela é válida.

### Chat não responde ou responde "Não tenho informações"

Certifique-se de que você executou a ingestão antes de iniciar o chat e que o PDF foi processado com sucesso.