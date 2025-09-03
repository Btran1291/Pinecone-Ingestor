# ---- Build Stage ----
FROM python:3.11-slim-bookworm AS builder

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install build dependencies and system libraries for document processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libffi-dev \
        libssl-dev \
        python3-dev \
        git \
        libmagic-dev \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-eng \
        libreoffice \
        pandoc \
        libgl1-mesa-glx \
    rm -rf /var/lib/apt/lists/*

RUN pip install "poetry==2.1.4"
ENV POETRY_VIRTUALENVS_PATH=/app/.venv

COPY pyproject.toml poetry.lock ./

# Install Python dependencies into Poetry's virtual environment
RUN poetry install --only main --no-root --no-interaction

RUN ls -l /app/.venv/bin/python


# ---- Production Stage ----
FROM python:3.11-slim-bookworm AS production

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Install runtime system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libmagic1 \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-eng \
        libreoffice \
        pandoc \
        libgl1-mesa-glx \
    rm -rf /var/lib/apt/lists/*

# Download SpaCy model directly in production for robustness
RUN python -m spacy download en_core_web_sm

# Copy application code and run script
COPY app.py run.sh ./
RUN chmod +x run.sh

EXPOSE 8501

# Use wrapper script for proper signal handling

ENTRYPOINT ["./run.sh"]
