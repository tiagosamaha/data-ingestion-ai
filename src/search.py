from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector

from src import config


PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def search_prompt(question: str) -> str:
    embeddings = OpenAIEmbeddings(model=config.OPENAI_EMBEDDING_MODEL, api_key=config.OPENAI_API_KEY)

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=config.PG_VECTOR_COLLECTION_NAME,
        connection=config.DATABASE_URL,
        use_jsonb=True,
    )

    documents_with_scores = vector_store.similarity_search_with_score(
        question, k=config.K_RETRIEVAL_RESULTS
    )

    filtered_documents = [
        (doc, score)
        for doc, score in documents_with_scores
        if score <= config.SIMILARITY_SCORE_THRESHOLD
    ]

    if not filtered_documents:
        return "Não tenho informações necessárias para responder sua pergunta."

    context = "\n".join([doc.page_content for doc, score in filtered_documents])

    prompt = PROMPT_TEMPLATE.format(contexto=context, pergunta=question)

    llm = ChatOpenAI(
        model=config.OPENAI_LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        api_key=config.OPENAI_API_KEY,
    )

    response = llm.invoke(prompt)

    return response.content
