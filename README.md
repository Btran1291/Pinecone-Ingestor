# Pinecone Ingestor

[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.30+-green)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/docker-ready-blue)](https://www.docker.com/)
[![Spaces](https://img.shields.io/badge/🤗-Hugging%20Face%20Spaces-yellow)](https://huggingface.co/spaces/Btran1291/pinecone-ingestor)

> 🎯 **Live Demo**: https://huggingface.co/spaces/Btran1291/pinecone-ingestor

---

## 📚 Overview

**Pinecone Ingestor** is a Streamlit application that turns documents into Pinecone vectors with zero code. It automates the complex process of cleaning, organizing, and uploading content to a Pinecone vector database for AI applications.

The app runs locally (Docker or bare Streamlit) and on Hugging Face Spaces. On Spaces, API keys remain in the user’s browser storage, never on the server.

---

## 🧠 Feature Highlights

-   **🔍 Structural Awareness:** Identify headers, paragraphs, and tables during the conversion process, ensuring the original meaning and layout of your document are preserved.
-   **🧼 Automated Text Cleaning:** Utilize a targeted cleaning engine to detect and remove commercial watermarks and software branding from within text blocks.
-   **📉 Noise Reduction:** Automatically filter out visual artifacts such as repetitive symbols or stray characters often generated during PDF-to-text conversion.
-   **🧠 Logic-Based Organizing:** Group sentences into units based on the ideas they represent rather than simple character counts. This ensures retrieved information remains coherent.
-   **🖇️ Context Preservation:** Prevent fragmented results by automatically merging short introductory sentences or fragments into the next relevant paragraph.
-   **⛓️ Sequential Mapping:** Link every piece of information to the content that preceded and followed it, allowing your AI to retrieve broader context when needed.

---

## ⚙️ Configuration Essentials

| Setting | Purpose |
|---------|---------|
| **Pinecone API Key / Index / Region** | Connects to your vector store. Serverless index auto-created if absent. |
| **OpenAI API Key / Embedding Model** | Uses `text-embedding-3-small` (1536 dims) by default. |
| **Chunk Size & Overlap** | Fallback recursive splitter parameters (used if semantic chunker can’t split). |
| **Semantic Threshold (type & amount)** | Controls how sensitive the semantic chunker is to topic shifts. |
| **Filtering Controls** | Minimum lengths, keyword whitelist, SpaCy NER toggle, low-confidence retention. |
| **Custom Metadata** | Global key/value pairs plus optional document-level metadata file (CSV/JSON with `file_name`). |
| **Overwrite Existing Docs** | If enabled, removes prior vectors for the same document before ingestion. |

All settings can be exported/imported as JSON profiles from the UI.

---

## 🚀 Running the App

### 1. Run Locally with Docker (recommended)

```bash
docker run -p 8501:8501 \
  -v $(pwd):/app \
  khoit12/pinecone-ingestor:v1
```

PowerShell equivalent:

```powershell
docker run -p 8501:8501 -v ${PWD}:/app khoit12/pinecone-ingestor:v1
```

Then visit `http://localhost:8501`.

### 2. Run Locally without Docker

```bash
git clone https://github.com/Btran1291/Pinecone-Ingestor.git
cd pinecone-ingestor
poetry install
poetry run streamlit run app.py
```

### 3. Deploy on Hugging Face Spaces

1. Visit the repository: https://huggingface.co/spaces/Btran1291/pinecone-ingestor/tree/main  
2. Click **⋯ > Duplicate this Space** (or “Use this Space as a template”).  
3. Choose **Streamlit** as the SDK and a GPU/CPU tier that meets your needs.  
4. After the duplicate finishes building, open the Space and the UI should be running.

---

## 📁 Local Artifacts

When running with `-v $(pwd):/app`, the app keeps everything in your working directory:

- `.env` – stores API keys and configuration (local mode).
- `.ingest_cache/` – per-document checkpoint files (resume embeddings).
- `pinecone_manifest.json` – tracks document IDs ↔ filenames for quick deduplication.

On Hugging Face, equivalent data is isolated per session ID in `/sessions/<uuid>`.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

---

## 📄 License

Licensed under the MIT License.
