# Pinecone Ingestor

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/downloads/)  
[![Streamlit](https://img.shields.io/badge/streamlit-v1.XX-green)](https://streamlit.io/)  
[![Docker](https://img.shields.io/badge/docker-compatible-blue)](https://www.docker.com/)

---

## Overview

**Pinecone Ingestor** is a Streamlit web application designed to help you build and manage a Retrieval-Augmented Generation (RAG) knowledge base using Pinecone vector database. It enables ingestion of diverse document types into a Pinecone vector database by leveraging LangChain's `UnstructuredLoader` and advanced document processing pipelines.

---

## Installation

### Prerequisites

- Docker (version 20.10+ recommended) installed and running  
  [Get Docker](https://docs.docker.com/get-docker/)  
- Pinecone account with API key  
  [Sign up at Pinecone](https://www.pinecone.io/start/)  
- OpenAI API key (or compatible embedding API key)  
  [Get OpenAI API key](https://platform.openai.com/account/api-keys)  

---

### Clone the Repository

```bash
git clone https://github.com/Btran1291/Pinecone-Ingestor.git
cd Pinecone-Ingestor
```

---

### Configuration

The application supports configuring parameters via a `.env` file. This file should contain your API keys and other settings. The app creates and updates this file automatically when you save configuration through the UI.

You can also create the file manually in the project root. If you choose to create the `.env` file manually, here is an example template. **Replace the placeholder values with your own:**

```env
PINECONE_API_KEY="your-pinecone-api-key"
EMBEDDING_API_KEY="your-openai-api-key"
PINECONE_INDEX_NAME="your-index-name"
PINECONE_CLOUD_REGION="aws-us-east-1"
EMBEDDING_MODEL_NAME="text-embedding-3-small"
EMBEDDING_DIMENSION="1536"
METRIC_TYPE="cosine"
NAMESPACE=""
CHUNK_SIZE="1000"
CHUNK_OVERLAP="200"
OVERWRITE_EXISTING_DOCS="False"
ENABLE_FILTERING="True"
WHITELISTED_KEYWORDS=""
MIN_GENERIC_CONTENT_LENGTH="50"
ENABLE_NER_FILTERING="True"
UNSTRUCTURED_STRATEGY="hi_res"
LOGGING_LEVEL="INFO"
```

---

### Build Docker Image

Build the Docker image using the provided `Dockerfile`:

```bash
docker build -t pinecone-ingestor .
```

> **Note:** The Docker image is quite large (~14GB) due to the inclusion of multiple system dependencies (Poppler, Tesseract, LibreOffice, Pandoc, etc.) required for comprehensive document parsing. As a result, building the image may take awhile depending on your machine and network speed.

---

### Run the Docker Container

Run the container, mapping port 8501 and passing environment variables:

```bash
docker run -p 8501:8501 --env-file .env pinecone-ingestor
```

---

### Access the Application

Open your browser and navigate to:

```
http://localhost:8501
```

You should see the Pinecone Ingestor UI.

---

## Usage

### 1. Configuration

- Enter your Pinecone API key, OpenAI API key, and other parameters in the **Configuration** section.
- Choose your Pinecone index name, cloud region, embedding model, and vector dimension.
- Configure document processing parameters such as chunk size, chunk overlap, and filtering options.
- Optionally upload document-specific metadata (CSV or JSON format).
- Save or reset your configuration as needed.
- Use the **Test API Connections** button to verify your keys.

### 2. Upload Documents

- Upload one or more documents in supported formats (e.g., PDF, DOCX, PPTX, TXT, CSV).
- Click **Process, Embed & Upsert to Pinecone** to start ingestion.
- Monitor progress and logs in real-time.

### 3. Manage Documents

- View Pinecone index status and vector counts.
- Load and search document names within the current namespace.
- View metadata for individual documents.
- Delete specific documents or perform bulk deletion within the namespace.

### 4. Application Logs

- Access detailed logs for debugging and monitoring.
- Clear logs as needed.

---

## Supported Document Types

Thanks to LangChain’s `UnstructuredLoader` and the `unstructured[all-docs]` package, this tool supports a wide range of document formats, including but not limited to:

- PDF
- Microsoft Word (.docx)
- Microsoft PowerPoint (.pptx)
- Microsoft Excel (.xlsx)
- EPUB
- Markdown (.md)
- HTML/XML
- CSV/TSV
- Plain text (.txt)
- Rich Text Format (.rtf)
- Email (.eml)
- Org-mode (.org)
- ReStructuredText (.rst)

---

## System Dependencies

The Docker image includes all necessary system dependencies for document parsing and OCR:

- Poppler utilities (`poppler-utils`)
- Tesseract OCR (`tesseract-ocr` and language packs)
- LibreOffice (for MS Office formats)
- Pandoc (for EPUB and other conversions)
- libmagic (file type detection)
- Mesa GL (`libgl1-mesa-glx`) for image processing

---

## Development

### Install Poetry Dependencies Locally

If you want to run the app locally without Docker:

```bash
poetry install
poetry run streamlit run app.py
```

Make sure to install system dependencies like Poppler, Tesseract, LibreOffice, and Pandoc on your local machine to ensure full document support.

---

## Contributing

Contributions, issues, and feature requests are welcome! Please open an issue or submit a pull request.

---

## License

This project is licensed under the MIT License.
