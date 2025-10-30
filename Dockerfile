# Build stage
FROM python:3.13-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.13-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY src/ ./src/
COPY document.pdf ./document.pdf

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Set Python to run in unbuffered mode for better logging
ENV PYTHONUNBUFFERED=1

# Default environment variables (can be overridden at runtime)
ENV DATABASE_URL=postgresql://postgres:postgres@postgres:5432/rag
ENV PG_VECTOR_COLLECTION_NAME=embeddings
ENV GOOGLE_EMBEDDING_MODEL=models/embedding-001
ENV OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Entrypoint is the chat application
ENTRYPOINT ["python", "src/chat.py"]
