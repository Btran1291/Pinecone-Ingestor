# Pinecone Ingestor

[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/downloads/)  
[![Streamlit](https://img.shields.io/badge/streamlit-v1.XX-green)](https://streamlit.io/)  
[![Docker](https://img.shields.io/badge/docker-compatible-blue)](https://www.docker.com/)

---
# 📚 Pinecone Document Ingestor

This application provides a professional pipeline for transforming documents into a structured, searchable database for AI applications. It automates the complex process of cleaning, organizing, and uploading content to a Pinecone vector database.

## 🛠️ Key Functionality

-   **🔍 Structural Awareness:** Identify headers, paragraphs, and tables during the conversion process, ensuring the original meaning and layout of your document are preserved.
-   **🧼 Automated Text Cleaning:** Utilize a targeted cleaning engine to detect and remove commercial watermarks and software branding from within text blocks.
-   **📉 Noise Reduction:** Automatically filter out visual artifacts such as repetitive symbols or stray characters often generated during PDF-to-text conversion.
-   **🧠 Logic-Based Organizing:** Group sentences into units based on the ideas they represent rather than simple character counts. This ensures retrieved information remains coherent.
-   **🖇️ Context Preservation:** Prevent fragmented results by automatically merging short introductory sentences or fragments into the next relevant paragraph.
-   **⛓️ Sequential Mapping:** Link every piece of information to the content that preceded and followed it, allowing your AI to retrieve broader context when needed.

***

## ⚙️ Understanding the Settings

Configure the application to match the specific needs of your documents:

### 📄 Document Processing

-   **Strategy (`fast` vs `hi_res`):**
    -   `fast`: Optimized for text-heavy documents. This is the recommended setting for standard books and reports.
    -   `hi_res`: Necessary only for documents containing complex tables or charts. This requires significantly more time as the system performs a detailed layout scan.

### ✂️ Organizing the Content

-   **Chunk Size:** Set the maximum allowable length for a single piece of information.
-   **Min Length:** Set the "floor" for information units. Anything smaller than this will be merged with its neighbor to ensure sufficient context for AI responses.
-   **Semantic Threshold:** Control the sensitivity of topic changes. A higher number forces the system to keep more sentences together unless a definitive shift in subject matter is detected.

***

## 🚀 How to Set Up and Run

The application is packaged as a container, ensuring all necessary dependencies are pre-installed and ready for immediate use.

### 1. Requirements

-   **Docker Desktop:** Must be installed and active.
-   **System Resources:** For processing large PDFs, allocate at least **4GB of RAM** to Docker in its resource settings.

### 2. Launch the Application

Open your terminal and execute the command corresponding to your operating system. These commands link the app to your current directory to securely store your settings and progress.

#### **Windows (PowerShell):**

```powershell
docker run -p 8501:8501 -v ${PWD}:/app khoit12/pinecone-ingestor:v1
```

#### **Mac / Linux / GitBash:**

```bash
docker run -p 8501:8501 -v $(pwd):/app khoit12/pinecone-ingestor:v1
```

### 3. Access the Interface

Once the container is running, open your web browser and navigate to:
`http://localhost:8501`

***

## 📦 Local Data Management

When using the commands above, the following items will be created in your local directory to support the application:

-   `.env`: Securely stores your Pinecone and OpenAI API keys.
-   `.ingest_cache/`: A temporary directory used to store progress, allowing the app to resume after an interruption.
-   `pinecone_manifest.json`: A record of documents that have been successfully indexed.

---

## Contributing

Contributions, issues, and feature requests are welcome! Please open an issue or submit a pull request.

---

## License

This project is licensed under the MIT License.
