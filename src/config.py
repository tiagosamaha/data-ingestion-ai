from decouple import config


# Database and Storage
PDF_PATH = config("PDF_PATH")
DATABASE_URL = config(
    "DATABASE_URL",
    default="postgresql://postgres:postgres@localhost:5432/rag"
)
PG_VECTOR_COLLECTION_NAME = config("PG_VECTOR_COLLECTION_NAME", default="companies")

# Google AI
GOOGLE_API_KEY = config("GOOGLE_API_KEY", default="")
GOOGLE_EMBEDDING_MODEL = config("GOOGLE_EMBEDDING_MODEL", default="models/embedding-001")

# OpenAI (optional)
OPENAI_API_KEY = config("OPENAI_API_KEY", default="")
OPENAI_EMBEDDING_MODEL = config("OPENAI_EMBEDDING_MODEL", default="text-embedding-3-small")

# Document Processing
CHUNK_SIZE = config("CHUNK_SIZE", default=1000, cast=int)
CHUNK_OVERLAP = config("CHUNK_OVERLAP", default=150, cast=int)

# LLM Configuration
K_RETRIEVAL_RESULTS = config("K_RETRIEVAL_RESULTS", default=10, cast=int)
SIMILARITY_SCORE_THRESHOLD = config("SIMILARITY_SCORE_THRESHOLD", default=0.8, cast=float)
GOOGLE_LLM_MODEL = config("GOOGLE_LLM_MODEL", default="gemini-2.5-flash-lite")
LLM_TEMPERATURE = config("LLM_TEMPERATURE", default=0.2, cast=float)
