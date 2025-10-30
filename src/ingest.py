import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def ingest_pdf():
    logger.info("Inicializando modelo de embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=config.GOOGLE_EMBEDDING_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
    )

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
            filter={"source": config.PDF_PATH}
        )

        if existing:
            logger.warning(f"Arquivo {config.PDF_PATH} já foi processado anteriormente.")
            logger.warning("Para reprocessar, limpe a collection primeiro.")
            return
    except Exception as e:
        logger.info(f"Collection não existe ainda ou erro ao verificar: {e}")
        logger.info("Prosseguindo com a ingestão...")

    logger.info(f"Carregando PDF: {config.PDF_PATH}")
    loader = PyPDFLoader(config.PDF_PATH)
    documents = loader.load()
    logger.info(f"PDF carregado: {len(documents)} páginas")

    logger.info("Dividindo documentos em chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Documentos divididos em {len(chunks)} chunks")

    logger.info("Gerando embeddings e salvando no pgvector...")
    PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.PG_VECTOR_COLLECTION_NAME,
        connection=config.DATABASE_URL,
    )
    logger.info(f"Ingestão concluída! {len(chunks)} chunks armazenados no pgvector.")


if __name__ == "__main__":
    ingest_pdf()
