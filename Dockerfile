# ---- Build Stage ----
FROM python:3.10-slim-bookworm AS builder

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
        libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

RUN pip install "poetry==2.1.4"

# Copy project definition files for dependency resolution
COPY pyproject.toml poetry.lock ./

# Install Python dependencies into Poetry's virtual environment
RUN poetry install --only main --no-root --no-interaction


# ---- Production Stage ----
FROM python:3.10-slim-bookworm AS production

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libmagic-dev \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-eng \
        libreoffice \
        pandoc \
        libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder stage
COPY --from=builder /root/.cache/pypoetry/virtualenvs /root/.cache/pypoetry/virtualenvs
COPY pyproject.toml poetry.lock ./

# Configure Poetry to use the copied virtual environment
RUN pip install "poetry==2.1.4" && \
    poetry config virtualenvs.create false && \
    poetry env use python3.10

# Download SpaCy model directly in production for robustness
RUN poetry run python -m spacy download en_core_web_sm

# Copy application code and run script
COPY app.py run.sh ./
RUN chmod +x run.sh

EXPOSE 8501

# Use wrapper script for proper signal handling
ENTRYPOINT ["./run.sh"]