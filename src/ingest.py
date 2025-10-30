import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_embeddings_model():
    logger.info("Inicializando modelo de embeddings...")
    embeddings = OpenAIEmbeddings(model=config.OPENAI_EMBEDDING_MODEL, api_key=config.OPENAI_API_KEY)
    return embeddings


def check_if_processed(embeddings, pdf_path):
    logger.info("Verificando se o arquivo já foi processado...")
    try:
        store = PGVector(
            embeddings=embeddings,
            collection_name=config.PG_VECTOR_COLLECTION_NAME,
            connection=config.DATABASE_URL,
        )

        existing = store.similarity_search(
            query="",
            k=1,
            filter={"source": pdf_path}
        )

        if existing:
            logger.warning(f"Arquivo {pdf_path} já foi processado anteriormente.")
            logger.warning("Para reprocessar, limpe a collection primeiro.")
            return True

    except Exception as e:
        logger.error(f"Collection não existe ainda ou erro ao verificar: {e}")

    return False


def load_pdf_documents(pdf_path):
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")

    logger.info(f"Carregando PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    logger.info(f"PDF carregado: {len(documents)} páginas")
    return documents


def split_documents_into_chunks(documents):
    logger.info("Dividindo documentos em chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Documentos divididos em {len(chunks)} chunks")
    return chunks


def store_chunks_in_vector_db(chunks, embeddings):
    logger.info("Gerando embeddings e salvando no pgvector...")
    PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.PG_VECTOR_COLLECTION_NAME,
        connection=config.DATABASE_URL,
    )
    logger.info(f"Ingestão concluída! {len(chunks)} chunks armazenados no pgvector.")


def ingest_pdf():
    try:
        embeddings = create_embeddings_model()

        if check_if_processed(embeddings, config.PDF_PATH):
            return

        documents = load_pdf_documents(config.PDF_PATH)
        chunks = split_documents_into_chunks(documents)
        store_chunks_in_vector_db(chunks, embeddings)

    except FileNotFoundError as e:
        logger.error(f"Arquivo não encontrado: {e}")
        raise
    except ValueError as e:
        logger.error(f"Erro de validação: {e}")
        raise
    except Exception as e:
        logger.error(f"Erro inesperado durante ingestão: {e}")
        raise


if __name__ == "__main__":
    try:
        ingest_pdf()
    except Exception as e:
        logger.error(f"Falha na ingestão: {e}")
        exit(1)
