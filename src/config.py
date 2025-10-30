from decouple import config


# Database and Storage
PDF_PATH = config("PDF_PATH")
DATABASE_URL = config("DATABASE_URL")
PG_VECTOR_COLLECTION_NAME = config("PG_VECTOR_COLLECTION_NAME")

# Google AI
GOOGLE_API_KEY = config("GOOGLE_API_KEY", default="")
GOOGLE_EMBEDDING_MODEL = config("GOOGLE_EMBEDDING_MODEL", default="models/embedding-001")

# OpenAI (optional)
OPENAI_API_KEY = config("OPENAI_API_KEY", default="")
OPENAI_EMBEDDING_MODEL = config("OPENAI_EMBEDDING_MODEL", default="text-embedding-3-small")

# Document Processing
CHUNK_SIZE = config("CHUNK_SIZE", default=1000, cast=int)
CHUNK_OVERLAP = config("CHUNK_OVERLAP", default=150, cast=int)
