FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=2.1.4 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONPATH=/usr/local/src/app \
    HOME=/app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential gcc libffi-dev libssl-dev python3-dev \
        libmagic1 poppler-utils tesseract-ocr tesseract-ocr-eng \
        libreoffice pandoc libgl1-mesa-glx libxext6 libxrender1 libsm6 git curl \
        && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

WORKDIR /usr/local/src/app
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root --no-interaction

RUN python -m spacy download en_core_web_sm && \
    python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"

COPY . .
RUN sed -i 's/\r$//' run.sh && chmod +x run.sh

WORKDIR /app
RUN chmod 777 /app

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8501/ || exit 1

ENTRYPOINT ["/usr/local/src/app/run.sh"]
