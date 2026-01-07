import streamlit as st
import streamlit.components.v1 as components
import os
import tempfile
import shutil
import json
import hashlib
import re
import uuid
import time
import logging
import threading
from collections import deque, defaultdict
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_core.documents import Document
from langchain.storage import InMemoryByteStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import NotFoundException
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import base64
import io
from PIL import Image
import pandas as pd
import tiktoken
import spacy
import nltk

# -------- Logging Configuration (Must be defined before use) --------
class StreamlitLogHandler(logging.Handler):
    def __init__(self, max_records=500):
        super().__init__()
        self.log_records = deque(maxlen=max_records)
        self.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    def emit(self, record):
        self.log_records.append(self.format(record))

    def get_records(self):
        return list(self.log_records)

def setup_logging_once(level=logging.INFO):
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if "streamlit_handler" not in st.session_state:
        st.session_state.streamlit_handler = StreamlitLogHandler()

    logger_name = f"AppLogger_{st.session_state.session_id}"
    logger = logging.getLogger(logger_name)
    
    logger.setLevel(level)

    if not logger.handlers:
        logger.addHandler(st.session_state.streamlit_handler)
        if "SPACE_ID" not in os.environ: # Only console log if NOT on Hugging Face
            logger.addHandler(logging.StreamHandler())
        
        st.session_state.app_logger = logger
        logger.info(f"Logger initialized for Session: {st.session_state.session_id}")
    
    return logger

def _get_safe_logger(name_suffix: str = "Generic"):
    try:
        return st.session_state.get("app_logger") or logging.getLogger(
            f"AppLogger_{st.session_state.get('session_id', name_suffix)}"
        )
    except Exception:
        return logging.getLogger(f"AppLogger_Fallback_{name_suffix}")

# -------- Start Global Initialization --------
# 1. Define HF Context
IS_ON_HF = "SPACE_ID" in os.environ

# 2. Setup Logging and capture NLTK events immediately
logger = setup_logging_once()
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
    logger.info("NLTK data downloaded successfully.")
except Exception as e:
    logger.error(f"NLTK setup error: {e}")

# 3. Define isolated workspace paths
BASE_DIR = os.getcwd()
if IS_ON_HF:
    # On HF, isolation is critical to prevent cross-user data leakage
    DATA_DIR = os.path.join(BASE_DIR, "sessions", st.session_state.session_id)
else:
    # Locally, stay in the root for easy access to manifest and .env
    DATA_DIR = BASE_DIR

os.makedirs(DATA_DIR, exist_ok=True)

# 4. Path Helpers
def get_manifest_path():
    return os.path.join(DATA_DIR, "pinecone_manifest.json")

def get_cache_path(doc_id: str):
    cache_folder = os.path.join(DATA_DIR, ".ingest_cache")
    os.makedirs(cache_folder, exist_ok=True)
    return os.path.join(cache_folder, f"cache_{doc_id}.json")

_GLOBAL_CHECKPOINT_LOCKS: dict[str, threading.Lock] = {}

def _get_checkpoint_lock(doc_id: str) -> threading.Lock:
    """
    Returns a per-document threading.Lock. Uses st.session_state when available;
    falls back to a module-level dict otherwise.
    """
    try:
        lock_map = st.session_state.setdefault("_checkpoint_locks", {})
    except Exception:
        lock_map = _GLOBAL_CHECKPOINT_LOCKS

    if doc_id not in lock_map:
        lock_map[doc_id] = threading.Lock()
    return lock_map[doc_id]

# -------- Application Defaults --------
DEFAULT_SETTINGS = {
    "PINECONE_API_KEY": "",
    "EMBEDDING_API_KEY": "",
    "PINECONE_INDEX_NAME": "",
    "PINECONE_CLOUD_REGION": "aws-us-east-1",
    "EMBEDDING_MODEL_NAME": "text-embedding-3-small",
    "EMBEDDING_DIMENSION": "1536",
    "METRIC_TYPE": "cosine",
    "NAMESPACE": "",
    "CHUNK_SIZE": "3600",
    "CHUNK_OVERLAP": "540",
    "CUSTOM_METADATA": '[{"key": "", "value": ""}]',
    "OVERWRITE_EXISTING_DOCS": "False",
    "LOGGING_LEVEL": "INFO",
    "ENABLE_FILTERING": "True",
    "WHITELISTED_KEYWORDS": "",
    "MIN_GENERIC_CONTENT_LENGTH": "150",
    "ENABLE_NER_FILTERING": "True",
    "UNSTRUCTURED_STRATEGY": "fast",
    "MIN_CHILD_CHUNK_LENGTH": "100",
    "KEEP_LOW_CONFIDENCE_SNIPPETS": "False",
}

# Supported Pinecone cloud regions and their corresponding cloud/region values for ServerlessSpec
SUPPORTED_PINECONE_REGIONS = {
    "aws-us-east-1": {"cloud": "aws", "region": "us-east-1"},
    "aws-us-west-2": {"cloud": "aws", "region": "us-west-2"},
    "aws-eu-west-1": {"cloud": "aws", "region": "eu-west-1"},
    "gcp-us-central1": {"cloud": "gcp", "region": "us-central1"},
    "gcp-europe-west4": {"cloud": "gcp", "region": "europe-west4"},
    "azure-eastus2": {"cloud": "azure", "region": "eastus2"},
}

# Constants for Pinecone and OpenAI API interactions
UPSERT_BATCH_SIZE = 100
PINECONE_METADATA_MAX_BYTES = 40 * 1024  # 40 KB limit for metadata payload

# Constants for dynamic chunking to adhere to Pinecone metadata limits
SAFETY_BUFFER_BYTES = 1024
MAX_UTF8_BYTES_PER_CHAR = 4

# OpenAI embedding API limits
OPENAI_MAX_TOKENS_PER_EMBEDDING_REQUEST = 300000
OPENAI_MAX_TEXTS_PER_EMBEDDING_REQUEST = 1000

# Metadata keys reserved for internal use or special handling within Pinecone
RESERVED_METADATA_KEYS = {
    "document_id",
    "file_name",
    "text",
    "page_number",
    "page",
    "start_index",
    "category",
    "original_file_path",
}

# Recommended allowed keys and priority list (highest -> lowest)
# Added 'section_id' and 'chunk_index_in_section' for plugin compatibility
RAG_ALLOWED_KEYS = [
    "text",
    "chunk_id",
    "document_id",
    "file_name",
    "section_id",  # Added for plugin compatibility
    "section_heading",
    "heading_level",
    "chunk_index_in_section",  # Added for plugin compatibility
    "chunk_confidence",
    "chunk_index_in_parent",  # Kept for internal reference
    "total_chunks_in_section",
    "page_number",
    "languages",
    "filetype",
    "last_modified",
    "start_index",  # Added start_index to RAG_ALLOWED_KEYS for better retention
]

# Keys to always remove (sensitive or irrelevant)
RAG_DISCARD_KEYS = {
    "file_directory",
    "original_file_path",
    "source",
    "filename",
    "element_id",
    "parent_id",
    "coordinates",
    "page",
    "file_path",  # 'start_index' removed from discard
}


# -------- Cached Resource Loading --------
@st.cache_resource
def load_spacy_model(model_name="en_core_web_sm"):
    """
    Loads a SpaCy model once and caches it to avoid repeated loading.
    Provides user feedback during loading and handles potential errors.
    """
    logger = st.session_state.app_logger
    try:
        logger.info(f"Loading SpaCy model '{model_name}'...")
        with st.spinner(f"Loading SpaCy model '{model_name}' for NER filtering..."):
            nlp = spacy.load(model_name)
        logger.info(f"SpaCy model '{model_name}' loaded successfully.")
        return nlp
    except Exception as e:
        logger.error(
            f"Failed to load SpaCy model '{model_name}': {e}. NER filtering will be disabled."
        )
        st.error(
            f"Failed to load SpaCy model '{model_name}'. NER filtering will be disabled. Please ensure it's installed (`python -m spacy download {model_name}`)."
        )
        return None


# -------- Helper Functions --------
def sync_browser_storage(data=None, action="load"):
    """
    Professional bridge between Streamlit and Browser localStorage.
    Only executes when hosted on Hugging Face to ensure persistence.
    """
    if not IS_ON_HF:
        return
    
    # Unique key for this app's storage to prevent collisions
    STORAGE_KEY = "pinecone_ingestor_v1_config"
    
    if action == "save" and data:
        # Push: Save JSON to browser
        json_data = json.dumps(data).replace("'", "\\'")
        js_code = f"""
            <script>
                localStorage.setItem('{STORAGE_KEY}', '{json_data}');
                console.log('Configuration saved to browser storage.');
            </script>
        """
        components.html(js_code, height=0)
        
    elif action == "load":
        # Pull: Read from browser and send to Streamlit via query params
        # This is a 'handshake' - JS reads storage and puts it in the URL
        # so Python can read it once, then we clear the URL.
        js_code = f"""
            <script>
                const data = localStorage.getItem('{STORAGE_KEY}');
                if (data) {{
                    const url = new URL(window.location);
                    if (!url.searchParams.has('hydrated')) {{
                        url.searchParams.set('hydrated', 'true');
                        url.searchParams.set('config_payload', data);
                        window.location.href = url.href;
                    }}
                }}
            </script>
        """
        components.html(js_code, height=0)

    elif action == "clear":
        # Purge: Remove data from browser
        js_code = f"""
            <script>
                localStorage.removeItem('{STORAGE_KEY}');
                console.log('Browser storage cleared.');
                window.location.reload();
            </script>
        """
        components.html(js_code, height=0)

def save_uploaded_file_to_temp(uploaded_file):
    """Saves file to the session-specific sandbox."""
    logger = st.session_state.app_logger
    logger.debug(f"Saving uploaded file: {uploaded_file.name}")
    temp_dir = None
    try:
        # Force the temp directory into the private session sandbox
        session_temp_root = os.path.join(DATA_DIR, "temp")
        os.makedirs(session_temp_root, exist_ok=True)

        temp_dir = tempfile.mkdtemp(dir=session_temp_root)
        file_path = os.path.join(temp_dir, uploaded_file.name)
        file_bytes = uploaded_file.getvalue()
        with open(file_path, "wb") as f:
            f.write(file_bytes)
        logger.info(f"Saved {uploaded_file.name} to sandbox: {file_path}")
        return file_path, temp_dir, file_bytes
    except Exception as e:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
        logger.exception(f"Error saving {uploaded_file.name}")
        st.error(f"Error saving uploaded file: {e}")
        return None, None, None


def is_running_locally():
    """
    Checks if the Streamlit application is running in a local environment
    by looking for common cloud environment variables.
    """
    logger = st.session_state.app_logger
    cloud_env_vars = [
        "RENDER_EXTERNAL_HOSTNAME",
        "STREAMLIT_CLOUD_APP_NAME",
        "AWS_EXECUTION_ENV",
        "GCP_PROJECT",
        "AZURE_FUNCTIONS_ENVIRONMENT",
        "KUBERNETES_SERVICE_HOST",
    ]
    is_local = not any(os.environ.get(var) for var in cloud_env_vars)
    logger.debug(f"Running locally check: {is_local}")
    return is_local


def _load_manifest() -> list[dict]:
    logger = _get_safe_logger("ManifestLoad")
    path = get_manifest_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Manifest read failed: {e}")
        return []



def _save_manifest(entries: list[dict]):
    logger = _get_safe_logger("ManifestSave")
    try:
        with open(get_manifest_path(), "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to write manifest: {e}")



def load_doc_checkpoint(doc_id: str):
    try:
        logger = st.session_state.get("app_logger") or logging.getLogger(
            f"AppLogger_{st.session_state.get('session_id')}"
        )
    except Exception:
        logger = logging.getLogger("AppLogger_Fallback")
    path = get_cache_path(doc_id)
    lock = _get_checkpoint_lock(doc_id)
    with lock:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(
                    f"Corrupt checkpoint found for {doc_id}: {e}. Starting fresh."
                )
                return {}
    return {}

def save_doc_checkpoint(doc_id: str, data: dict):
    path = get_cache_path(doc_id)
    lock = _get_checkpoint_lock(doc_id)
    with lock:
        with tempfile.NamedTemporaryFile(
            "w", delete=False, dir=os.path.dirname(path)
        ) as tf:
            json.dump(data, tf)
            tempname = tf.name
        os.replace(tempname, path)

def clear_doc_checkpoint(doc_id: str):
    path = get_cache_path(doc_id)
    lock = _get_checkpoint_lock(doc_id)
    with lock:
        if os.path.exists(path):
            os.remove(path)

def _append_manifest_entry(document_id: str, file_name: str):
    logger = _get_safe_logger("ManifestAppend")
    entries = _load_manifest()
    entries = [e for e in entries if e.get("document_id") != document_id]
    entries.append({"document_id": document_id, "file_name": file_name})
    _save_manifest(entries)
    logger.debug(f"Manifest updated with document_id={document_id}, file_name={file_name}")

def _remove_manifest_entry(document_id: str):
    logger = _get_safe_logger("ManifestRemove")
    entries = _load_manifest()
    new_entries = [e for e in entries if e.get("document_id") != document_id]
    if len(new_entries) != len(entries):
        _save_manifest(new_entries)
        logger.debug(f"Removed manifest entry for document_id={document_id}")
    else:
        logger.debug(f"No manifest entry found for document_id={document_id}")

def _clear_manifest():
    logger = _get_safe_logger("ManifestClear")
    try:
        _save_manifest([])
        logger.info("Manifest cleared.")
    except Exception as e:
        logger.error(f"Failed to clear manifest: {e}")

def pinecone_has_index_cached(
    pc: Pinecone | None,
    index_name: str,
    logger: logging.Logger,
    cache_scope: str | None = None,
) -> bool:
    """
    Wraps pc.has_index with per-session caching and error handling to avoid repeated
    control-plane calls and transient exceptions.
    """
    if not pc or not index_name:
        return False

    scope = cache_scope or "default"
    cache_key = f"has_index::{scope}::{index_name}"
    cache = st.session_state.setdefault("pinecone_has_index_cache", {})

    if cache_key in cache:
        return cache[cache_key]

    try:
        exists = pc.has_index(index_name)
    except Exception as e:
        logger.warning(f"Unable to determine if index '{index_name}' exists: {e}")
        exists = False

    cache[cache_key] = exists
    return exists


def _bootstrap_manifest_from_pinecone(
    index,
    namespace: str | None,
    logger: logging.Logger,
    embedding_dimension: int,  # kept for signature compatibility (unused now)
    page_size: int = 99,
    fetch_batch_size: int = 200,
):
    """
    Rebuilds pinecone_manifest.json by walking every vector ID via the
    /vectors/list API (serverless indexes only) and fetching metadata in batches.
    """
    logger.info("Bootstrapping manifest via Pinecone list_paginated API...")
    namespace_name = namespace or None
    entries_by_doc: dict[str, str] = {}
    pagination_token: str | None = None
    total_ids_seen = 0

    while True:
        try:
            list_response = index.list_paginated(
                namespace=namespace_name,
                limit=min(max(page_size, 1), 99),
                pagination_token=pagination_token,
            )
        except Exception as e:
            logger.error(f"Vector ID listing failed: {e}. Aborting manifest rebuild.")
            break

        vectors = list_response.get("vectors") or []
        if not vectors:
            logger.info("No more vector IDs returned; finishing manifest rebuild.")
            break

        vector_ids = [vec.get("id") for vec in vectors if vec.get("id")]
        total_ids_seen += len(vector_ids)
        logger.debug(
            f"Fetched {len(vector_ids)} IDs (total so far: {total_ids_seen:,})."
        )

        for start in range(0, len(vector_ids), fetch_batch_size):
            chunk_ids = vector_ids[start : start + fetch_batch_size]
            try:
                fetch_response = index.fetch(ids=chunk_ids, namespace=namespace_name)
            except Exception as e:
                logger.warning(f"Fetch failed for {len(chunk_ids)} IDs: {e}")
                continue

            fetched_vectors = fetch_response.vectors or {}
            for vec in fetched_vectors.values():
                metadata = vec.metadata or {}
                doc_id = metadata.get("document_id")
                file_name = metadata.get("file_name")
                if doc_id and file_name and doc_id not in entries_by_doc:
                    entries_by_doc[doc_id] = file_name

        pagination_token = (list_response.get("pagination") or {}).get("next")
        if not pagination_token:
            logger.info("Reached end of pagination.")
            break

    if entries_by_doc:
        entries = [
            {"document_id": doc_id, "file_name": file_name}
            for doc_id, file_name in entries_by_doc.items()
        ]
        _save_manifest(entries)
        logger.info(f"Manifest rebuilt with {len(entries):,} unique documents.")
    else:
        logger.warning(
            "Manifest rebuild completed but no documents were discovered. "
            "Ensure the index is serverless and contains vectors with document metadata."
        )

def _list_documents_via_pinecone(
    index,
    namespace: str | None,
    logger: logging.Logger,
    page_size: int = 99,
    fetch_batch_size: int = 200,
) -> list[dict]:
    """
    Fetches a list of unique documents (document_id + file_name) directly from Pinecone
    by paginating vector IDs and inspecting their metadata. This avoids using the local
    manifest, which is useful on Hugging Face where shared filesystem state is not guaranteed.

    Returns:
        A list of dictionaries: [{"document_id": ..., "file_name": ...}, ...]
    """
    namespace_name = namespace or None
    entries_by_doc: dict[str, str] = {}
    pagination_token: str | None = None
    total_ids_seen = 0

    while True:
        try:
            list_response = index.list_paginated(
                namespace=namespace_name,
                limit=min(max(page_size, 1), 99),
                pagination_token=pagination_token,
            )
        except Exception as e:
            logger.error(f"Vector ID listing failed: {e}. Aborting listing operation.")
            break

        vectors = list_response.get("vectors") or []
        if not vectors:
            logger.info("No more vector IDs returned during listing.")
            break

        vector_ids = [vec.get("id") for vec in vectors if vec.get("id")]
        total_ids_seen += len(vector_ids)
        logger.debug(f"Fetched {len(vector_ids)} IDs (total so far: {total_ids_seen:,}).")

        for start in range(0, len(vector_ids), fetch_batch_size):
            chunk_ids = vector_ids[start : start + fetch_batch_size]
            try:
                fetch_response = index.fetch(ids=chunk_ids, namespace=namespace_name)
            except Exception as e:
                logger.warning(f"Fetch failed for {len(chunk_ids)} IDs: {e}")
                continue

            fetched_vectors = fetch_response.vectors or {}
            for vec in fetched_vectors.values():
                metadata = vec.metadata or {}
                doc_id = metadata.get("document_id")
                file_name = metadata.get("file_name")
                if doc_id and file_name and doc_id not in entries_by_doc:
                    entries_by_doc[doc_id] = file_name

        pagination_token = (list_response.get("pagination") or {}).get("next")
        if not pagination_token:
            logger.info("Reached end of listing pagination.")
            break

    entries = [
        {"document_id": doc_id, "file_name": file_name}
        for doc_id, file_name in entries_by_doc.items()
    ]
    logger.info(
        f"Listing completed via Pinecone API; discovered {len(entries):,} unique documents."
    )
    return entries

def _repair_dangling_brackets(text: str) -> str:
    """
    Identifies and closes dangling brackets or parentheses in headings.
    Ensures breadcrumbs like '[ qi-qing' become '[ qi-qing ]'.
    """
    if not text:
        return ""

    # 1. Clean basic whitespace
    cleaned = text.strip()

    # 2. Define pairs to check: {Opening: Closing}
    bracket_pairs = {"[": "]", "(": ")", "{": "}"}

    for open_char, close_char in bracket_pairs.items():
        open_count = cleaned.count(open_char)
        close_count = cleaned.count(close_char)

        # If there are more opening than closing, append the missing closes
        if open_count > close_count:
            # We add a space before the closing bracket if the last char is alphanumeric
            # for better readability (e.g. "[ chapter" -> "[ chapter ]")
            if cleaned and cleaned[-1].isalnum():
                cleaned += " "
            cleaned += close_char * (open_count - close_count)

    return cleaned


def deterministic_document_id(file_name: str, file_bytes: bytes) -> str:
    """
    Generates a deterministic document ID based on the file name and content hash.
    Ensures uniqueness and handles Pinecone's ID length limit.
    """
    logger = st.session_state.app_logger
    # Use the normalized file_name for ID generation
    doc_id = f"{file_name.lower()}_{hashlib.sha256(file_bytes).hexdigest()}"
    if len(doc_id) > 512:
        doc_id = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()
    logger.debug(f"Generated document ID for {file_name}: {doc_id}")
    return doc_id


def deterministic_chunk_id(
    document_id: str, chunk_text: str, page: str, start_index: str
) -> str:
    """
    Generates a deterministic chunk ID based on the document ID, chunk text hash,
    page number, and start index.
    """
    logger = st.session_state.app_logger
    text_hash = hashlib.sha256((chunk_text or "").encode("utf-8")).hexdigest()
    pos = f"{page}_{start_index}"
    chunk_id = hashlib.sha256(
        f"{document_id}_{text_hash}_{pos}".encode("utf-8")
    ).hexdigest()
    logger.debug(
        f"Generated chunk ID for doc {document_id}, page {page}, start {start_index}: {chunk_id}"
    )
    return chunk_id


def surgical_text_cleaner(text: str) -> str:
    """
    Stage 1: Removes non-linguistic commercial artifacts.
    Stage 2: Collapses structural noise (stars/dashes).
    Stage 3: Normalizes whitespace to prevent embedding bias.
    """
    if not text:
        return ""

    # 1. Commercial Signatures
    noise_patterns = [
        r"\*+.*Febo[qQkK]ok.*\*+",
        r"\*+.*DEMO Watermarks.*\*+",
        r"\*+.*Trial Version.*\*+",
        r"Produced by an unregistered version.*",
        r"Converted by Nitro PDF.*",
        r"Click here to buy.*",
        r"www\.abisource\.com",
        r"ABBYY FineReader.*",
        r"Evaluation Copy.*",
    ]

    logger = st.session_state.get("app_logger", logging.getLogger(f"AppLogger_{st.session_state.get('session_id')}"))
    initial_len = len(text)
    cleaned = text
    for pattern in noise_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.MULTILINE)

    cleaned = re.sub(r"[\*\-\=\_\#]{4,}", "", cleaned)
    cleaned = re.sub(r" +", " ", cleaned)
    cleaned = re.sub(r"(\n\s*){2,}", "\n\n", cleaned)
    
    final_text = cleaned.strip()
    if initial_len - len(final_text) > 0:
        logger.debug(f"Surgical Cleaner: Removed {initial_len - len(final_text)} noise characters.")
    
    return final_text

def _is_structural_noise(text: str) -> bool:
    if not text:
        return True

    cleaned = text.lower().strip(" \t\n\r*-_=<>[]")

    # 1. Explicit Targeted Filter
    NOISE_TARGETS = {
        "feboqok",
        "febokok",
        "abisource",
        "unregistered",
        "trial version",
        "evaluation copy",
        "demo watermark",
        "converted by",
        "produced by",
        "nitro pdf",
        "abbyy finereader",
        "ocr technology",
        "www.",
        ".com",
        "all rights reserved",
        "click here to buy",
        "watermarked",
    }
    if any(target in cleaned for target in NOISE_TARGETS):
        return True

    # 2. Structural/Visual Noise Gate
    # Catches things like "-----------" or "**********"
    if len(cleaned) > 4:
        alnum_chars = sum(1 for c in cleaned if c.isalnum())
        if (alnum_chars / len(cleaned)) < 0.3:
            return True

    # 3. Stray Fragment Gate
    # Removes single characters or page numbers like "Page 1" that
    # shouldn't be titles or chunks.
    if len(cleaned) < 2 or re.match(r"^(page\s?\d+|[0-9]+\s?of\s?[0-9]+)$", cleaned):
        return True

    return False


def count_tokens(text: str, model_name: str, logger: logging.Logger) -> int:
    """
    Counts tokens in a text string using tiktoken for OpenAI models.
    Falls back to a character-based estimation if the model encoding is not found.
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(text))
    except KeyError:
        logger.warning(
            f"Could not find tiktoken encoding for model '{model_name}'. Falling back to conservative character-based estimation."
        )
        return max(1, len(text) // 3)
    except Exception as e:
        logger.error(f"Error counting tokens for model '{model_name}': {e}")
        return max(1, len(text) // 3)


def split_text_by_tokens(
    text: str,
    max_tokens: int,
    overlap_tokens: int = 0,
    logger: logging.Logger | None = None,
) -> list[str]:
    """
    Splits text into pieces constrained by token count using tiktoken-aware splitting.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")

    try:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=max_tokens,
            chunk_overlap=overlap_tokens,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    except Exception as e:
        if logger:
            logger.error(f"Failed to create token-aware splitter: {e}")
        raise

    return splitter.split_text(text)


def prepend_token_overlap(
    previous_text: str,
    current_text: str,
    overlap_tokens: int,
    model_name: str,
    logger: logging.Logger,
) -> str:
    """
    Takes the tail of the previous chunk (overlap_tokens) and prepends it to the current chunk.
    Ensures semantic continuity across chunks.
    """
    if not previous_text or overlap_tokens <= 0:
        return current_text

    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except Exception as e:
        logger.warning(
            f"Falling back to char-based overlap; encoding lookup failed: {e}"
        )
        encoding = None

    if encoding:
        prev_tokens = encoding.encode(previous_text)
        tail_tokens = (
            prev_tokens[-overlap_tokens:]
            if len(prev_tokens) > overlap_tokens
            else prev_tokens
        )
        overlap_text = encoding.decode(tail_tokens)
    else:
        overlap_text = previous_text[-overlap_tokens * 3 :]  # rough char fallback

    return overlap_text + current_text


def compute_dynamic_threshold(base_amount: float, parent_tokens: int) -> float:
    """
    General-purpose logic: If a section is already relatively short (under 1500 tokens),
    we increase the threshold to prevent unnecessary splitting.
    """
    if parent_tokens <= 0:
        return base_amount

    # If the text block is under ~1500 tokens (approx 6-8 paragraphs),
    # we make it harder to split. This keeps coherent ideas together.
    if parent_tokens < 1500:
        return min(99.5, base_amount + 1.5)

    # For very large sections, we use the user's base setting.
    return base_amount


def _canonicalize_and_filter_metadata(
    raw_meta: dict,
    global_custom_md: dict,
    doc_specific_md: dict,
    logger: logging.Logger,
) -> dict:
    """
    Return a cleaned metadata dict that:
    - canonicalizes duplicates (e.g., 'filename' -> 'file_name'),
    - removes definitely irrelevant keys,
    - merges/validates custom metadata (global first, then doc-specific overrides),
    - ensures primitive types (strings/numbers/bools) or JSON-stringified values for lists/dicts.
    Note: Does NOT yet ensure byte-size under pinecone limit; see shrink_to_limit below.
    """
    cleaned = {}

    # 1) Work on a shallow copy to avoid mutating caller's meta
    meta = dict(raw_meta or {})

    # Check for 'page_number' first
    if "page_number" in meta and meta["page_number"] is not None:
        try:
            # Attempt to convert to integer
            cleaned["page_number"] = int(meta["page_number"])
        except (ValueError, TypeError):
            # If conversion fails, store as string and log a warning
            cleaned["page_number"] = str(meta["page_number"])
            logger.warning(
                f"Could not convert page_number '{meta['page_number']}' to int. Storing as string."
            )
    # If 'page_number' is not found, check for 'page'
    elif "page" in meta and meta["page"] is not None:
        try:
            # Attempt to convert 'page' to integer and store as 'page_number'
            cleaned["page_number"] = int(meta["page"])
        except (ValueError, TypeError):
            # If conversion fails, store as string and log a warning
            cleaned["page_number"] = str(meta["page"])
            logger.warning(
                f"Could not convert page '{meta['page']}' to int. Storing as string."
            )

    meta.pop("page", None)
    meta.pop("page_number", None)

    # Canonicalize: prefer 'file_name', but accept 'filename' as fallback
    if "file_name" not in meta and "filename" in meta:
        meta["file_name"] = meta.pop("filename")

    # Ensure file_name is consistently lowercase
    if "file_name" in meta and isinstance(meta["file_name"], str):
        meta["file_name"] = meta["file_name"].lower()

    # Remove discard keys early
    for k in list(meta.keys()):
        if k in RAG_DISCARD_KEYS:
            logger.debug(f"Removing discarded metadata key '{k}'.")
            meta.pop(k, None)

    # Keep keys that are probably useful; others can be considered custom if requested
    for k, v in meta.items():
        # Skip None or empty values
        if v is None:
            continue

        # Convert allowed types to safe serializable types
        if isinstance(v, (str, int, float, bool)):
            cleaned[k] = v
        elif isinstance(v, (list, dict)):
            # For lists/dicts, JSON stringify but keep small ones as structured lists if short strings
            try:
                json_repr = json.dumps(v)
                cleaned[k] = (
                    json.loads(json_repr)
                    if (isinstance(v, list) and all(isinstance(i, str) for i in v))
                    else json_repr
                )
            except Exception:
                cleaned[k] = str(v)
                logger.debug(
                    f"Converted non-serializable metadata key '{k}' to string."
                )
        else:
            # Fallback to string
            cleaned[k] = str(v)
            logger.debug(f"Converted metadata key '{k}' of type {type(v)} to string.")

    # 2) Merge global and document-specific custom metadata (document-specific overrides)
    for k, v in (global_custom_md or {}).items():
        if not k or k in RESERVED_METADATA_KEYS:  # keep reserved protection
            logger.debug(
                f"Ignoring global custom metadata key '{k}' (reserved or empty)."
            )
            continue
        # Convert types similarly
        if isinstance(v, (str, int, float, bool)):
            cleaned.setdefault(k, v)
        else:
            try:
                cleaned.setdefault(k, json.dumps(v))
            except Exception:
                cleaned.setdefault(k, str(v))

    # doc_specific overrides
    for k, v in (doc_specific_md or {}).items():
        if not k or k in RESERVED_METADATA_KEYS:
            logger.debug(
                f"Ignoring doc-specific metadata key '{k}' (reserved or empty)."
            )
            continue
        if isinstance(v, (str, int, float, bool)):
            cleaned[k] = v
        else:
            try:
                cleaned[k] = json.dumps(v)
            except Exception:
                cleaned[k] = str(v)

    # 3) Ensure canonical keys exist where relevant
    if "file_name" not in cleaned and "document_id" in cleaned:
        # no filename available; that's okay, do not invent it
        pass

    return cleaned


def _shrink_metadata_to_limit(
    metadata: dict,
    logger: logging.Logger,
    pinecone_metadata_max_bytes: int = PINECONE_METADATA_MAX_BYTES,
) -> dict:
    """
    Ensure metadata JSON byte size <= pinecone_metadata_max_bytes.
    Strategy:
      1) Keep highest priority keys in RAG_ALLOWED_KEYS. Remove keys not in allowed list while respecting allowed custom keys
         (we allow additional keys but prefer RAG_ALLOWED_KEYS first).
      2) If still too big, remove lower-priority allowed keys in reverse priority order.
      3) If still too big, truncate the 'text' field (as last resort) and mark text_truncated=True.
    Returns the pruned metadata (may be modified).
    """

    removed_keys = []

    # Quick check
    try:
        metadata_json = json.dumps(metadata)
    except TypeError:
        # Try converting all non-serializable entries to strings first
        safe_meta = {}
        for k, v in metadata.items():
            try:
                json.dumps({k: v})
                safe_meta[k] = v
            except Exception:
                safe_meta[k] = str(v)
        metadata = safe_meta
        metadata_json = json.dumps(metadata)

    size = len(metadata_json.encode("utf-8"))
    if size <= pinecone_metadata_max_bytes:
        return metadata

    logger.warning(
        f"Metadata size {size} bytes exceeds limit {pinecone_metadata_max_bytes} bytes. Beginning shrink process."
    )

    # Build prioritized keys to keep: start with RAG_ALLOWED_KEYS in order, then any remaining small custom keys
    prioritized = [k for k in RAG_ALLOWED_KEYS if k in metadata]
    # Add other keys by ascending length of their JSON repr (shorter first)
    other_keys = [k for k in metadata.keys() if k not in prioritized]
    other_keys_sorted = sorted(
        other_keys, key=lambda k: len(json.dumps({k: metadata[k]}))
    )
    prioritized.extend(other_keys_sorted)

    critical_keys = {"document_id", "file_name", "child_chunk_id", "chunk_id", "parent_chunk_id"}
    pruned = dict(metadata)
    for key in reversed(prioritized):
        if len(json.dumps(pruned).encode("utf-8")) <= pinecone_metadata_max_bytes:
            break
        if key == "text" or key in critical_keys:
            # don't remove text yet, and never remove critical identifiers
            continue
        if key in pruned:
            logger.debug(f"Removing metadata key '{key}' to reduce size.")
            removed_keys.append(key)
            pruned.pop(key, None)


    # If still too big, attempt to remove other keys not in RAG_ALLOWED_KEYS
    if len(json.dumps(pruned).encode("utf-8")) > pinecone_metadata_max_bytes:
        for key in list(pruned.keys()):
            if key not in RAG_ALLOWED_KEYS and key != "text":
                logger.debug(f"Removing additional non-priority metadata key '{key}'.")
                removed_keys.append(key)
                pruned.pop(key, None)
                if (
                    len(json.dumps(pruned).encode("utf-8"))
                    <= pinecone_metadata_max_bytes
                ):
                    break

    # If still too big, truncate the text field
    if len(json.dumps(pruned).encode("utf-8")) > pinecone_metadata_max_bytes:
        text = pruned.get("text", "")
        if text:
            # Compute allowed bytes for text after accounting for other metadata
            temp = dict(pruned)
            temp.pop("text", None)
            overhead = len(json.dumps(temp).encode("utf-8"))
            # Reserve small safety margin
            allowed_for_text = max(
                0, pinecone_metadata_max_bytes - overhead - SAFETY_BUFFER_BYTES
            )
            if allowed_for_text <= 0:
                # remove text entirely (should be last resort)
                logger.critical(
                    "No room left for 'text' in metadata; removing it as last resort. You should use external text storage."
                )
                pruned.pop("text", None)
                pruned["text_truncated"] = True
            else:
                # Determine character count conservatively (UTF-8 worst-case 4 bytes per char)
                max_chars = allowed_for_text // MAX_UTF8_BYTES_PER_CHAR
                truncated_text = text[:max_chars]
                pruned["text"] = truncated_text
                pruned["text_truncated"] = True
                logger.warning(
                    f"Truncated 'text' to {len(truncated_text)} chars to fit metadata limit ({pinecone_metadata_max_bytes} bytes)."
                )
        else:
            logger.critical(
                "Metadata too large and no 'text' field to shrink. Metadata will be pruned aggressively."
            )
            # prune additional keys until we fit
            for key in list(pruned.keys()):
                if key != "chunk_id":
                    removed_keys.append(key)
                    pruned.pop(key, None)
                if (
                    len(json.dumps(pruned).encode("utf-8"))
                    <= pinecone_metadata_max_bytes
                ):
                    break

    final_size = len(json.dumps(pruned).encode("utf-8"))
    if final_size > pinecone_metadata_max_bytes:
        logger.error(
            f"Unable to reduce metadata to {pinecone_metadata_max_bytes} bytes; final size {final_size} bytes. Upsert may fail."
        )
    else:
        logger.debug(
            f"Shrunk metadata to {final_size} bytes (limit {pinecone_metadata_max_bytes} bytes)."
        )

    if removed_keys:
        unique_keys = sorted(set(removed_keys))
        logger.info(f"Metadata size limit reached ({size} bytes). Pruned {len(unique_keys)} keys to reach {final_size} bytes.")
        logger.debug(f"Pruned metadata keys: {unique_keys}")
    
    if pruned.get("text_truncated"):
        logger.warning("CRITICAL: Metadata 'text' field was truncated to fit Pinecone limits.")

    return pruned

def _finalize_parent_section(
    parent_documents: list[Document],
    parent_doc_store: InMemoryByteStore,
    current_parent_id: str,
    current_parent_content: list[str],
    current_parent_metadata: dict,
    logger: logging.Logger,
):
    """Utility to flush the current parent buffer into a Document."""
    if not current_parent_content:
        return

    full_parent_content = "\n\n".join(current_parent_content)
    parent_metadata_copy = current_parent_metadata.copy()

    page_start = parent_metadata_copy.get("page_number_start")
    page_end = parent_metadata_copy.get("page_number_end")
    if page_start is not None and page_end is not None:
        parent_metadata_copy["page_range"] = f"{page_start}-{page_end}"
    elif page_start is not None:
        parent_metadata_copy["page_range"] = str(page_start)

    breadcrumbs = []
    file_name = parent_metadata_copy.get("file_name")
    if file_name:
        breadcrumbs.append(file_name)

    heading_hierarchy = parent_metadata_copy.get("heading_hierarchy") or []
    if heading_hierarchy:
        breadcrumbs.extend(heading_hierarchy)
    else:
        # Fallback to single section heading if hierarchy missing
        section_heading = parent_metadata_copy.get("section_heading")
        if section_heading:
            breadcrumbs.append(section_heading)

    parent_metadata_copy["breadcrumbs"] = breadcrumbs

    parent_doc = Document(
        page_content=full_parent_content, metadata=parent_metadata_copy
    )
    parent_doc.metadata["parent_chunk_id"] = current_parent_id
    parent_metadata_copy["breadcrumbs"] = breadcrumbs

    parent_documents.append(parent_doc)
    parent_doc_store.mset(
        [(current_parent_id, parent_doc.page_content.encode("utf-8"))]
    )

    logger.debug(
        f"Created parent document '{current_parent_id}' "
        f"with {len(current_parent_content)} elements and {len(full_parent_content)} chars."
    )


def _build_breadcrumbs(
    parent_meta: dict, heading_stack: list[tuple[int, str]], file_name: str | None
) -> list[str]:
    """
    Construct breadcrumb path (e.g., ["doc.pdf", "Chapter 3", "Section A"]).
    """
    breadcrumbs: list[str] = []

    if file_name:
        breadcrumbs.append(file_name)

    if heading_stack:
        breadcrumbs.extend([heading for _, heading in heading_stack if heading])
    else:
        section_heading = parent_meta.get("section_heading")
        if section_heading:
            breadcrumbs.append(section_heading)

    return breadcrumbs


# -------- Parent Document Generation Helper --------
def _create_parent_documents(
    raw_elements: list[Document],
    logger: logging.Logger,
    embedding_model_name: str,
) -> tuple[list[Document], InMemoryByteStore]:
    TOKEN_CAP_PER_PARENT = 3000
    try:
        token_encoding = tiktoken.encoding_for_model(embedding_model_name)
    except Exception as e:
        logger.warning(
            f"Falling back to cl100k_base encoding for parent segmentation: {e}"
        )
        try:
            token_encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as inner_e:
            logger.error(
                f"Could not load fallback encoding; using char counts. Error: {inner_e}"
            )
            token_encoding = None
    HEADING_LEVEL_MAP = {"Title": 1, "Header": 2, "Subheader": 3}
    heading_stack: list[tuple[int, str]] = []
    parent_documents = []
    parent_doc_store = InMemoryByteStore()

    if not raw_elements:
        logger.warning(
            "No raw elements provided for parent document creation. Returning empty."
        )
        return [], parent_doc_store

    current_parent_content = []
    current_parent_metadata = {}
    current_parent_id = None
    current_parent_tokens = 0

    # Define strong structural breaks. These categories from Unstructured.io often mark new sections.
    MAJOR_STRUCTURAL_BREAKS = {"Title", "Header"}

    # Heuristic: If a NarrativeText or ListItem is very short and follows a significant break,
    # it might be a de-facto heading or a very short, distinct point.
    SHORT_ELEMENT_AS_BREAK_THRESHOLD = 160  # Characters

    for i, element in enumerate(raw_elements):
        element_text = (element.page_content or "").strip()
        element_category = element.metadata.get("category")

        # Update heading stack when we encounter heading-like elements
        heading_level = element.metadata.get("heading_level")
        category_level = HEADING_LEVEL_MAP.get(element_category)
        level = heading_level if isinstance(heading_level, int) else category_level

        if level:
            # 1. Structural Noise Filter
            if not _is_structural_noise(element_text):

                # 2. Repair dangling brackets and cleanup formatting for metadata
                clean_heading = _repair_dangling_brackets(element_text)

                # 3. Update the hierarchy stack
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()

                heading_stack.append((level, clean_heading))

        elif element_category in {"Title", "Header", "Subheader"} and not heading_stack:
            if not _is_structural_noise(element_text):
                clean_title = _repair_dangling_brackets(element_text)
                heading_stack.append((1, clean_title))

        # Skip truly empty elements
        if not element_text:
            continue

        # Check for a new parent document break
        is_major_break = False
        # Only Title or Header triggers a hard break
        if element_category in MAJOR_STRUCTURAL_BREAKS:
            is_major_break = True

        # Safety: If a parent is getting excessively long, force a break
        if current_parent_tokens > 2500:
            is_major_break = True

        # Also consider significant page number jumps as a break
        if "page_number" in element.metadata and current_parent_metadata.get(
            "page_number"
        ):
            try:
                current_page = int(element.metadata["page_number"])
                parent_start_page = int(current_parent_metadata["page_number"])
                if current_page > parent_start_page + 1:  # Jump of more than one page
                    is_major_break = True
            except (ValueError, TypeError):
                pass  # Ignore if page numbers are not easily convertible to int

        if is_major_break and current_parent_content:
            # We extract the file name from the first element's metadata
            fname = raw_elements[0].metadata.get('file_name', 'unknown_file')
            logger.debug(f"Triggering section break for '{fname}' due to: "
                         f"{'Major Category ('+element_category+')' if element_category in MAJOR_STRUCTURAL_BREAKS else 'Token/Page Limit'}")
            _finalize_parent_section(
                parent_documents=parent_documents,
                parent_doc_store=parent_doc_store,
                current_parent_id=current_parent_id,
                current_parent_content=current_parent_content,
                current_parent_metadata=current_parent_metadata,
                logger=logger,
            )

            current_parent_content = []
            current_parent_tokens = 0
            current_parent_id = str(uuid.uuid4())
            current_parent_metadata = dict(element.metadata)
            element.metadata["parent_chunk_id"] = current_parent_id
            current_parent_metadata["heading_hierarchy"] = [h[1] for h in heading_stack]
            current_parent_metadata["breadcrumbs"] = _build_breadcrumbs(
                current_parent_metadata,
                heading_stack,
                element.metadata.get("file_name"),
            )

            if "section_id" in element.metadata:
                current_parent_metadata["section_id"] = element.metadata["section_id"]
            if "section_heading" in element.metadata:
                current_parent_metadata["section_heading"] = element.metadata[
                    "section_heading"
                ]
            if "page_number" in element.metadata:
                try:
                    current_parent_metadata["page_number"] = int(
                        element.metadata["page_number"]
                    )
                except (ValueError, TypeError):
                    current_parent_metadata["page_number"] = str(
                        element.metadata["page_number"]
                    )

        if (
            not current_parent_id
        ):  # This condition will only be true for the very first element of the document
            current_parent_id = str(uuid.uuid4())
            current_parent_metadata = dict(element.metadata)
            element.metadata["parent_chunk_id"] = current_parent_id
            current_parent_metadata["heading_hierarchy"] = [h[1] for h in heading_stack]
            current_parent_metadata["breadcrumbs"] = _build_breadcrumbs(
                current_parent_metadata,
                heading_stack,
                element.metadata.get("file_name"),
            )

            # Propagate section_id and section_heading if available from Unstructured
            if "section_id" in element.metadata:
                current_parent_metadata["section_id"] = element.metadata["section_id"]
            if "section_heading" in element.metadata:
                current_parent_metadata["section_heading"] = element.metadata[
                    "section_heading"
                ]
            elif element.metadata.get("category") in {"Title", "Header", "Subheader"}:
                current_parent_metadata[
                    "section_heading"
                ] = element.page_content.strip()
            # Ensure page_number is handled correctly for the start of a parent
            if "page_number" in element.metadata:
                try:
                    current_parent_metadata["page_number"] = int(
                        element.metadata["page_number"]
                    )
                except (ValueError, TypeError):
                    current_parent_metadata["page_number"] = str(
                        element.metadata["page_number"]
                    )

        current_parent_content.append(element_text)
        if token_encoding:
            current_parent_tokens += len(token_encoding.encode(element_text))
        else:
            current_parent_tokens += max(1, len(element_text) // 3)

        if current_parent_tokens >= TOKEN_CAP_PER_PARENT:
            _finalize_parent_section(
                parent_documents=parent_documents,
                parent_doc_store=parent_doc_store,
                current_parent_id=current_parent_id,
                current_parent_content=current_parent_content,
                current_parent_metadata=current_parent_metadata,
                logger=logger,
            )
            current_parent_content = []
            current_parent_tokens = 0
            current_parent_id = str(uuid.uuid4())
            current_parent_metadata = dict(element.metadata)
            element.metadata["parent_chunk_id"] = current_parent_id
            current_parent_metadata["heading_hierarchy"] = [h[1] for h in heading_stack]
            current_parent_metadata["breadcrumbs"] = _build_breadcrumbs(
                current_parent_metadata,
                heading_stack,
                element.metadata.get("file_name"),
            )

            if "section_id" in element.metadata:
                current_parent_metadata["section_id"] = element.metadata["section_id"]
            if "section_heading" in element.metadata:
                current_parent_metadata["section_heading"] = element.metadata[
                    "section_heading"
                ]
            if "page_number" in element.metadata:
                try:
                    current_parent_metadata["page_number"] = int(
                        element.metadata["page_number"]
                    )
                except (ValueError, TypeError):
                    current_parent_metadata["page_number"] = str(
                        element.metadata["page_number"]
                    )

        element.metadata["parent_chunk_id"] = current_parent_id
        element.metadata["breadcrumbs"] = current_parent_metadata.get("breadcrumbs", [])

        # Update page number range for the parent document
        if "page_number" in element.metadata:
            try:
                current_page_num = int(element.metadata["page_number"])
                if (
                    "page_number_start" not in current_parent_metadata
                    or current_page_num < current_parent_metadata["page_number_start"]
                ):
                    current_parent_metadata["page_number_start"] = current_page_num
                if (
                    "page_number_end" not in current_parent_metadata
                    or current_page_num > current_parent_metadata["page_number_end"]
                ):
                    current_parent_metadata["page_number_end"] = current_page_num
            except (ValueError, TypeError):
                pass  # Non-integer page numbers will be handled as strings if needed, but not for range logic

    # Finalize the last parent document
    if current_parent_id:
        _finalize_parent_section(
            parent_documents=parent_documents,
            parent_doc_store=parent_doc_store,
            current_parent_id=current_parent_id,
            current_parent_content=current_parent_content,
            current_parent_metadata=current_parent_metadata,
            logger=logger,
        )

    logger.info(
        f"Generated {len(parent_documents)} parent documents for '{raw_elements[0].metadata.get('file_name', 'N/A')}'."
    )
    return parent_documents, parent_doc_store


# -------- Content Filtering Helper --------
def _filter_parent_content(
    parent_documents: list[Document],
    enable_filtering: bool,
    whitelisted_keywords_set: set[str],
    min_generic_content_length: int,
    enable_ner_filtering: bool,
    nlp: spacy.Language,
    keep_low_confidence: bool,
    logger: logging.Logger,
) -> list[Document]:

    """
    Applies content filtering rules to a list of parent documents.
    Removes generic, short content unless it's whitelisted,
    or contains named entities (if NER filtering is enabled).

    Args:
        parent_documents: A list of Document objects representing parent sections.
        enable_filtering: Boolean, whether to apply filtering at all.
        whitelisted_keywords_set: A set of keywords to always keep.
        min_generic_content_length: Minimum character length for generic content.
        enable_ner_filtering: Boolean, whether to use NER for short generic content.
        nlp: The loaded SpaCy model for NER.
        logger: The application logger.

    Returns:
        A new list of Document objects containing only the "meaningful" content,
        with original parent metadata preserved.
    """
    filtered_parent_documents = []
    if "verse_warning_shown" not in st.session_state:
        st.session_state.verse_warning_shown = False

    if not enable_filtering:
        logger.info(
            "Content filtering is disabled. All parent document content will be kept."
        )
        return parent_documents

    non_empty_lengths = [
        len((doc.page_content or "").strip())
        for doc in parent_documents
        if (doc.page_content or "").strip()
    ]
    short_ratio = (
        sum(1 for length in non_empty_lengths if length < 80) / len(non_empty_lengths)
        if non_empty_lengths
        else 0
    )
    looks_verse_like = short_ratio > 0.6

    if looks_verse_like and not keep_low_confidence:
        st.warning(
            "Detected many very short sections (poetry/verse-like). "
            "Enable 'Keep short low-confidence snippets' to avoid losing content."
        )
        st.session_state.verse_warning_shown = True
        logger.warning(
            "Verse-like corpus detected but keep_low_confidence is disabled. "
            "Short segments may be dropped."
        )

    for parent_doc in parent_documents:
        original_content = (parent_doc.page_content or "").strip()

        if not original_content:
            logger.debug(
                f"Skipping empty parent document content for '{parent_doc.metadata.get('file_name', 'N/A')}' "
                f"(Parent ID: {parent_doc.metadata.get('parent_chunk_id', 'N/A')})."
            )
            continue

        keep_this_parent = False
        low_confidence = False

        # Rule 1: Whitelisted keywords
        if any(
            keyword in original_content.lower() for keyword in whitelisted_keywords_set
        ):
            keep_this_parent = True
            logger.debug(
                f"Kept parent doc (Parent ID: {parent_doc.metadata.get('parent_chunk_id', 'N/A')}) "
                "due to whitelisted keyword."
            )

        # Rule 2: Sufficiently long content
        elif len(original_content) >= min_generic_content_length:
            keep_this_parent = True
            logger.debug(
                f"Kept parent doc (Parent ID: {parent_doc.metadata.get('parent_chunk_id', 'N/A')}) "
                f"due to sufficient length ({len(original_content)} chars)."
            )

        else:
            # Rule 3: NER filtering for short content
            if enable_ner_filtering and nlp is not None:
                doc_spacy = nlp(original_content)
                if doc_spacy.ents:
                    keep_this_parent = True
                    logger.debug(
                        f"Kept short parent doc (Parent ID: {parent_doc.metadata.get('parent_chunk_id', 'N/A')}) "
                        f"due to NER. Entities: {[ent.text for ent in doc_spacy.ents]}"
                    )

            if not keep_this_parent:
                # Keep but mark as low-confidence instead of discarding
                keep_this_parent = True
                low_confidence = True
                logger.debug(
                    f"Marked short parent doc (Parent ID: {parent_doc.metadata.get('parent_chunk_id', 'N/A')}) "
                    f"as low-confidence (length {len(original_content)})."
                )
        if low_confidence and not keep_low_confidence:
            if looks_verse_like:
                logger.info(
                    "Verse-like corpus detected but user disabled low-confidence snippets; dropping short parent."
                )
            logger.debug(
                f"Dropping short low-confidence parent doc "
                f"(Parent ID: {parent_doc.metadata.get('parent_chunk_id', 'N/A')}), "
                f"length {len(original_content)}."
            )
            continue

        if keep_this_parent:
            filtered_doc = Document(
                page_content=original_content, metadata=parent_doc.metadata.copy()
            )
            if low_confidence:
                filtered_doc.metadata["low_confidence"] = True
            filtered_parent_documents.append(filtered_doc)
        else:
            logger.debug(
                f"Filtered out parent document (Parent ID: {parent_doc.metadata.get('parent_chunk_id', 'N/A')}) "
                f"from '{parent_doc.metadata.get('file_name', 'N/A')}' due to filtering rules."
            )

    logger.info(
        f"Filtered {len(parent_documents)} parent documents down to {len(filtered_parent_documents)} meaningful parent documents."
    )
    dropped_count = len(parent_documents) - len(filtered_parent_documents)
    if dropped_count > 0:
        logger.info(
            f"Dropped {dropped_count} parent sections due to filtering "
            f"({dropped_count / len(parent_documents):.1%} of total)."
        )
    return filtered_parent_documents


def _locate_span_in_parent(
    parent_text: str, snippet: str, search_start: int = 0
) -> tuple[int | None, int | None]:
    """
    Finds the [start, end) character range of `snippet` in `parent_text`.
    Returns (start, end). If no match is found, returns (None, None).
    """
    if not snippet:
        return None, None

    idx = parent_text.find(snippet, search_start)
    if idx == -1 and search_start > 0:
        # Try searching a bit earlier to account for minor overlap shifts
        rewind_start = max(0, search_start - 500)
        idx = parent_text.find(snippet, rewind_start)
    if idx == -1:
        return None, None

    return idx, idx + len(snippet)


# -------- Semantic Chunking & Child Document Generation Helper --------
def _generate_child_chunks(
    filtered_parent_documents: list[Document],
    embedding_model_name: str,
    embedding_api_key: str,
    embedding_dimension: int,
    pinecone_metadata_max_bytes: int,
    parsed_global_custom_metadata: dict,
    document_specific_metadata_map: dict,
    logger: logging.Logger,
    semantic_chunker_threshold_type: str = "percentile",
    semantic_chunker_threshold_amount: float = 98.0,
    min_child_chunk_size: int = 100,
    configured_chunk_size: int = 1000,  # Added for fallback
    configured_chunk_overlap: int = 200,  # Added for fallback
) -> tuple[list[Document], dict[str, list[dict]]]:
    """
    Generates smaller, semantically coherent "child" chunks from parent documents.
    These child chunks are designed for precise retrieval and are linked back to their parents.

    Args:
        filtered_parent_documents: A list of Document objects (parent documents) after filtering.
        embedding_model_name: The name of the OpenAI embedding model.
        embedding_api_key: The OpenAI API key.
        embedding_dimension: The dimensionality of the embedding vectors (user-configured).
        pinecone_metadata_max_bytes: Pinecone's metadata byte limit.
        parsed_global_custom_metadata: Global custom metadata to apply to child chunks.
        document_specific_metadata_map: Map of document-specific metadata.
        logger: The application logger.
        semantic_chunker_threshold_type: Type of threshold for semantic chunking.
        semantic_chunker_threshold_amount: Amount for the semantic chunking threshold.
        min_child_chunk_size: Minimum character length for a child chunk.
        configured_chunk_size: User-configured chunk size for fallback splitter.
        configured_chunk_overlap: User-configured chunk overlap for fallback splitter.

    Returns:
        A list of Document objects, each representing a "child" chunk,
        with metadata linking it to its parent.
    """

    all_child_chunks = []
    parent_to_child_map: dict[str, list[dict]] = defaultdict(list)

    if not filtered_parent_documents:
        logger.warning(
            "No filtered parent documents provided for child chunk generation. Returning empty."
        )
        return [], {}

    max_child_chunk_tokens = 2000
    semantic_chunk_overlap_tokens = 150

    # Reuse one embeddings client plus cached semantic chunkers
    embeddings_for_chunker = OpenAIEmbeddings(
        openai_api_key=embedding_api_key,
        model=embedding_model_name,
        dimensions=embedding_dimension,
    )
    semantic_chunker_cache: dict[tuple[str, float], SemanticChunker] = {}

    for parent_doc in filtered_parent_documents:
        parent_content = (parent_doc.page_content or "").strip()
        parent_chunk_id = parent_doc.metadata.get("parent_chunk_id")

        if not parent_content:
            logger.warning(
                f"Parent document (ID: {parent_chunk_id}) has no content after filtering. Skipping child chunk generation."
            )
            continue

        parent_text = parent_doc.page_content or ""
        search_cursor = 0  # tracks where to resume searching inside parent_text

        logger.debug(
            f"Generating child chunks for parent document (ID: {parent_chunk_id}, File: {parent_doc.metadata.get('file_name', 'N/A')})..."
        )
        parent_to_child_map[parent_chunk_id] = []

        parent_tokens = count_tokens(parent_content, embedding_model_name, logger)
        dynamic_threshold_amount = compute_dynamic_threshold(
            base_amount=semantic_chunker_threshold_amount, parent_tokens=parent_tokens
        )
        rounded_threshold_amount = round(dynamic_threshold_amount, 2)

        threshold_key = (
            embedding_model_name,
            embedding_dimension,
            semantic_chunker_threshold_type,
            rounded_threshold_amount,
            embedding_api_key,
        )

        if threshold_key not in semantic_chunker_cache:
            semantic_chunker_cache[threshold_key] = SemanticChunker(
                embeddings_for_chunker,
                breakpoint_threshold_type=semantic_chunker_threshold_type,
                breakpoint_threshold_amount=rounded_threshold_amount,
            )

        semantic_text_splitter = semantic_chunker_cache[threshold_key]

        try:
            child_docs_from_parent = semantic_text_splitter.create_documents(
                [parent_content]
            )

        except Exception as e:
            logger.error(
                f"Semantic chunking failed for parent ID {parent_chunk_id}: {e}"
            )
            child_docs_from_parent = []

        if not child_docs_from_parent:
            logger.warning(
                f"Semantic chunker produced no chunks for parent ID {parent_chunk_id}. "
                "Falling back to RecursiveCharacterTextSplitter."
            )
            fallback_splitter = RecursiveCharacterTextSplitter(
                chunk_size=configured_chunk_size,
                chunk_overlap=configured_chunk_overlap,
                length_function=len,
                add_start_index=True,
            )
            child_docs_from_parent = fallback_splitter.create_documents(
                [parent_content]
            )

        pending_docs = list(
            child_docs_from_parent
        )  # shallow copy to avoid mutating original
        temp_child_chunks = []
        chunk_order_counter = 0
        prev_chunk_text = None

        while pending_docs:
            current_doc = pending_docs.pop(0)
            contains_overlap = False
            text_to_clean = (current_doc.page_content or "").strip()
            cleaned = text_to_clean.replace("\t", " ")
            cleaned = re.sub(r"[ ]{2,}", " ", cleaned)
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
            raw_child_text = cleaned.strip()

            if not raw_child_text or len(raw_child_text) < min_child_chunk_size:
                logger.debug(
                    f"Skipping chunk due to length ({len(raw_child_text)} chars)."
                )
                continue

            raw_child_token_length = count_tokens(
                raw_child_text, embedding_model_name, logger
            )
            if raw_child_token_length > max_child_chunk_tokens:
                logger.debug(
                    f"Chunk exceeds {max_child_chunk_tokens} tokens; splitting into smaller pieces."
                )

                split_texts = split_text_by_tokens(
                    raw_child_text,
                    max_tokens=max_child_chunk_tokens,
                    overlap_tokens=0,
                    logger=logger,
                )

                # Determine the absolute character start of the original chunk, if available
                original_start = current_doc.metadata.get("start_index")
                try:
                    base_char_start = (
                        int(original_start) if original_start is not None else None
                    )
                except (TypeError, ValueError):
                    base_char_start = None

                # If we don't have a reliable start index, fall back to current search cursor
                if base_char_start is None:
                    base_char_start = search_cursor

                cumulative_chars = 0
                split_segments = []
                for split_text in split_texts:
                    segment_start = (
                        base_char_start + cumulative_chars
                        if base_char_start is not None
                        else None
                    )
                    split_segments.append((split_text, segment_start))
                    cumulative_chars += len(split_text)

                # Reinsert segments to process them in order
                for split_text, absolute_start in reversed(split_segments):
                    split_metadata = current_doc.metadata.copy()
                    if absolute_start is not None:
                        split_metadata["start_index"] = absolute_start
                        split_metadata["search_start_hint"] = absolute_start
                    else:
                        split_metadata.pop("start_index", None)
                        split_metadata.pop("search_start_hint", None)
                    split_metadata["start_index_hint"] = "split"
                    split_metadata["contains_overlap"] = False

                    pending_docs.insert(
                        0, Document(page_content=split_text, metadata=split_metadata)
                    )
                continue

            span_start_hint = current_doc.metadata.get("search_start_hint")

            if span_start_hint is not None:
                search_cursor = int(span_start_hint)
            elif current_doc.metadata.get("contains_overlap"):
                search_cursor = max(0, search_cursor - 500)

            char_start = current_doc.metadata.get("start_index")
            char_end = None
            if char_start is not None:
                try:
                    char_start = int(char_start)
                    char_end = char_start + len(raw_child_text)
                except (ValueError, TypeError):
                    char_start = None
                    char_end = None

            if char_start is None:
                char_start, char_end = _locate_span_in_parent(
                    parent_text=parent_text,
                    snippet=raw_child_text,
                    search_start=search_cursor,
                )

            if char_end is not None:
                search_cursor = char_end

            final_child_text = raw_child_text
            embedding_text = final_child_text
            overlap_chars = 0
            if prev_chunk_text and semantic_chunk_overlap_tokens > 0:
                embedding_text = prepend_token_overlap(
                    previous_text=prev_chunk_text,
                    current_text=final_child_text,
                    overlap_tokens=semantic_chunk_overlap_tokens,
                    model_name=embedding_model_name,
                    logger=logger,
                )
                overlap_chars = len(embedding_text) - len(final_child_text)
                contains_overlap = True
                if char_end is not None:
                    search_cursor = char_end

            child_metadata = parent_doc.metadata.copy()
            child_metadata["child_chunk_id"] = deterministic_chunk_id(
                parent_chunk_id,
                final_child_text,
                child_metadata.get("page_number", ""),
                current_doc.metadata.get("start_index", ""),
            )
            child_metadata["chunk_index_in_parent"] = chunk_order_counter
            child_metadata["chunk_index_in_section"] = chunk_order_counter
            child_metadata["category"] = "semantic_text_chunk"
            child_metadata["contains_overlap"] = contains_overlap
            child_metadata["text"] = final_child_text
            if char_start is not None and char_end is not None:
                child_metadata["char_range"] = [char_start, char_end]
                child_metadata["char_range_unique"] = [char_start, char_end]
                if overlap_chars > 0:
                    child_metadata["overlap_prefix_chars"] = overlap_chars

            parent_breadcrumbs = parent_doc.metadata.get("breadcrumbs")
            if isinstance(parent_breadcrumbs, list) and parent_breadcrumbs:
                breadcrumbs = list(parent_breadcrumbs)
            else:
                breadcrumbs = []
                file_name = child_metadata.get("file_name")
                if file_name:
                    breadcrumbs.append(file_name)
                section_heading = child_metadata.get("section_heading")
                if section_heading:
                    breadcrumbs.append(section_heading)
            child_metadata["breadcrumbs"] = breadcrumbs

            anchor_page_number = child_metadata.get("page_number")
            anchor_start_index = current_doc.metadata.get("start_index")
            anchor_start_index_hint = current_doc.metadata.get("start_index_hint")
            anchor_char_range = child_metadata.get("char_range")

            child_text = final_child_text
            sanitized_pruned_metadata = _shrink_metadata_to_limit(
                _canonicalize_and_filter_metadata(
                    child_metadata,
                    parsed_global_custom_metadata,
                    document_specific_metadata_map.get(
                        child_metadata.get("file_name", ""), {}
                    ),
                    logger,
                ),
                logger,
                pinecone_metadata_max_bytes,
            )

            if sanitized_pruned_metadata.get("text_truncated", False):
                child_text = sanitized_pruned_metadata["text"]
                logger.warning(
                    f"Child chunk (ID: {sanitized_pruned_metadata.get('child_chunk_id', 'N/A')}) "
                    "text truncated during final metadata shrink."
                )

            chunk_doc = Document(
                page_content=child_text, metadata=sanitized_pruned_metadata
            )
            chunk_doc.metadata["_embedding_text"] = embedding_text
            temp_child_chunks.append(chunk_doc)

            parent_to_child_map[parent_chunk_id].append(
                {
                    "chunk_id": sanitized_pruned_metadata["child_chunk_id"],
                    "page_number": anchor_page_number,
                    "chunk_index": chunk_order_counter,
                    "start_index": anchor_start_index,
                    "start_index_hint": anchor_start_index_hint,
                    "char_range": anchor_char_range,
                }
            )

            prev_chunk_text = child_text
            chunk_order_counter += 1

        final_processed_chunks = []
        COHESION_MIN_CHARS = 500

        # Stage 1: Greedy Cohesion Pass
        # Consolidates short 'orphan' chunks into their neighbors
        buffer_doc = None

        for doc in temp_child_chunks:
            if buffer_doc is None:
                buffer_doc = doc
                continue

            # If the current buffer is too small, we merge the NEXT chunk into it
            if len(buffer_doc.page_content) < COHESION_MIN_CHARS:
                # 1. Merge Text Content
                new_content = buffer_doc.page_content + "\n" + doc.page_content
                buffer_doc.page_content = new_content
                buffer_doc.metadata["text"] = new_content

                # 2. Update Character Ranges (Essential for PDF Mapping)
                b_range = buffer_doc.metadata.get("char_range")
                d_range = doc.metadata.get("char_range")
                if b_range and d_range:
                    # New range is [Start of A, End of B]
                    buffer_doc.metadata["char_range"] = [b_range[0], d_range[1]]
                    buffer_doc.metadata["char_range_unique"] = [b_range[0], d_range[1]]

                # 3. Regenerate deterministic ID for the new combined content
                buffer_doc.metadata["child_chunk_id"] = deterministic_chunk_id(
                    buffer_doc.metadata.get("parent_chunk_id", "orphan"),
                    new_content,
                    str(buffer_doc.metadata.get("page_number", "")),
                    str(buffer_doc.metadata.get("char_range", [0])[0]),
                )
                logger.debug(
                    f"Master Cohesion: Merged orphan into cohesive unit. "
                    f"New range: {buffer_doc.metadata.get('char_range')} | "
                    f"New length: {len(new_content)} chars."
                )

            else:
                # Buffer is large enough, move it to final list and start new buffer
                final_processed_chunks.append(buffer_doc)
                buffer_doc = doc

        # Add the final trailing chunk
        if buffer_doc:
            final_processed_chunks.append(buffer_doc)
            
        parent_to_child_map[parent_chunk_id] = []
        for idx, chunk_doc in enumerate(final_processed_chunks):
            meta = chunk_doc.metadata
            parent_to_child_map[parent_chunk_id].append(
                {
                    "chunk_id": meta.get("child_chunk_id"),
                    "page_number": meta.get("page_number"),
                    "chunk_index": idx,
                    "start_index": meta.get("char_range", [None])[0],
                    "start_index_hint": meta.get("start_index"),
                    "char_range": meta.get("char_range"),
                }
            )

        # Stage 2: Linked-List Integrity Pass
        for idx, chunk_doc in enumerate(final_processed_chunks):
            p_id = (
                final_processed_chunks[idx - 1].metadata.get("child_chunk_id")
                if idx > 0
                else None
            )
            n_id = (
                final_processed_chunks[idx + 1].metadata.get("child_chunk_id")
                if idx < len(final_processed_chunks) - 1
                else None
            )

            if p_id:
                chunk_doc.metadata["previous_chunk_id"] = p_id
            if n_id:
                chunk_doc.metadata["next_chunk_id"] = n_id

        all_child_chunks.extend(final_processed_chunks)

    logger.info(
        f"Generated {len(all_child_chunks)} child chunks from {len(filtered_parent_documents)} parent documents."
    )
    return all_child_chunks, parent_to_child_map


# -------- Multi-Vector Indexing for Special Content Helper --------
def _process_special_elements(
    raw_elements: list[Document],
    parent_doc_store: InMemoryByteStore,
    parent_to_child_map: dict[str, list[dict]],
    embedding_model_name: str,
    embedding_api_key: str,
    embedding_dimension: int,
    pinecone_metadata_max_bytes: int,
    parsed_global_custom_metadata: dict,
    document_specific_metadata_map: dict,
    logger: logging.Logger,
) -> list[Document]:
    """
    Identifies and processes special elements (tables, images/figures) from raw_elements.
    Generates LLM-based summaries for these, creates child chunks from summaries,
    and stores raw content in the parent_doc_store.

    Args:
        raw_elements: The initial list of Document objects from UnstructuredLoader.
        parent_doc_store: The InMemoryByteStore where parent documents are stored.
                          We'll also use it to store raw special content.
        embedding_model_name: The name of the OpenAI embedding model.
        embedding_api_key: The OpenAI API key.
        embedding_dimension: The dimensionality of the embedding vectors.
        pinecone_metadata_max_bytes: Pinecone's metadata byte limit.
        parsed_global_custom_metadata: Global custom metadata.
        document_specific_metadata_map: Document-specific metadata map.
        logger: The application logger.

    Returns:
        A list of Document objects, each representing a "summary" child chunk for special content.
    """
    special_content_child_chunks = []
    special_chunk_counters = defaultdict(int)

    def _safe_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _pick_referenced_chunk(
        parent_id: str,
        page_number,
        element_order: int | None,
        element_start: str | None,
    ) -> str | None:
        candidates = parent_to_child_map.get(parent_id, [])
        if not candidates:
            return None

        # 1. Prefer page-number matches
        if page_number is not None:
            try:
                target_page = int(page_number)
                for candidate in candidates:
                    if candidate.get("page_number") == target_page:
                        return candidate["chunk_id"]
            except (ValueError, TypeError):
                pass

        # 2. Fallback: nearest char_range match
        if element_start is not None:
            target_start = _safe_int(element_start)

            def _range_distance(candidate):
                char_range = candidate.get("char_range")
                if (
                    not char_range
                    or not isinstance(char_range, (list, tuple))
                    or len(char_range) != 2
                ):
                    logger.debug(
                        f"No valid char_range for chunk {candidate.get('chunk_id')}; skipping."
                    )
                    return float("inf")
                start_val, end_val = char_range
                if start_val <= target_start <= end_val:
                    return 0
                return min(abs(start_val - target_start), abs(end_val - target_start))

            closest = min(candidates, key=_range_distance)
            if closest and _range_distance(closest) != float("inf"):
                return closest["chunk_id"]

        # 3. Fallback: nearest start_index or start_index_hint
        if element_start is not None:
            target_start = _safe_int(element_start)
            closest = min(
                candidates,
                key=lambda c: abs(
                    _safe_int(c.get("start_index") or c.get("start_index_hint"))
                    - target_start
                ),
            )
            return closest["chunk_id"]

        # 4. Last resort: first candidate
        return candidates[0]["chunk_id"]

    if not raw_elements:
        logger.warning(
            "No raw elements provided for special content processing. Returning empty."
        )
        return []

    # Initialize LLM for summarization (using a multimodal model for images)
    llm_for_summaries = ChatOpenAI(
        openai_api_key=embedding_api_key,
        model="gpt-4o-mini",  # Using a capable multimodal model for summarization
        temperature=0.0,
    )

    def _image_to_base64_str(image_path: str) -> str | None:
        """
        Converts an image file from a given path to a base64 encoded string.
        Returns None if the file cannot be opened or processed.
        """
        try:
            with Image.open(image_path) as image:
                buffered = io.BytesIO()
                # Use PNG for better quality/transparency, or original format if known
                image.save(buffered, format="PNG")
                return base64.b64encode(buffered.getvalue()).decode("utf-8")
        except FileNotFoundError:
            logger.error(f"Image file not found at path: {image_path}")
            return None
        except Exception as e:
            logger.error(f"Error converting image {image_path} to base64: {e}")
            return None

    for element in raw_elements:
        element_category = element.metadata.get("category")
        element_text = (element.page_content or "").strip()

        parent_chunk_id = element.metadata.get(
            "parent_chunk_id"
        ) or element.metadata.get("document_id")

        summary_content = None
        raw_special_content_to_store = None

        if element_category == "Table" and element_text:
            table_html = element.metadata.get("text_as_html")
            if table_html:
                raw_special_content_to_store = table_html.encode("utf-8")
                logger.debug(f"Summarizing table from parent ID {parent_chunk_id}...")
                try:
                    response = llm_for_summaries.invoke(
                        [
                            HumanMessage(
                                content=f"Summarize the following table content concisely, focusing on key data and insights:\n\n{element_text}"
                            )
                        ]
                    )
                    summary_content = response.content
                except Exception as e:
                    logger.error(
                        f"LLM summarization failed for table (Parent ID: {parent_chunk_id}): {e}"
                    )
                    summary_content = f"Table content summary failed. Original content: {element_text[:500]}..."
            else:
                logger.warning(
                    f"Table element (Parent ID: {parent_chunk_id}) found but no HTML content. Summarizing raw text."
                )
                raw_special_content_to_store = element_text.encode("utf-8")
                try:
                    response = llm_for_summaries.invoke(
                        [
                            HumanMessage(
                                content=f"Summarize the following table text concisely, focusing on key data and insights:\n\n{element_text}"
                            )
                        ]
                    )
                    summary_content = response.content
                except Exception as e:
                    logger.error(
                        f"LLM summarization failed for table (Parent ID: {parent_chunk_id}): {e}"
                    )
                    summary_content = f"Table content summary failed. Original content: {element_text[:500]}..."

        elif element_category == "FigureCaption" and element_text:
            raw_special_content_to_store = element_text.encode("utf-8")
            logger.debug(
                f"Summarizing figure caption from parent ID {parent_chunk_id}..."
            )
            try:
                response = llm_for_summaries.invoke(
                    [
                        HumanMessage(
                            content=f"Summarize the following figure caption concisely, highlighting the main point of the figure:\n\n{element_text}"
                        )
                    ]
                )
                summary_content = response.content
            except Exception as e:
                logger.error(
                    f"LLM summarization failed for figure caption (Parent ID: {parent_chunk_id}): {e}"
                )
                summary_content = f"Figure caption summary failed. Original content: {element_text[:500]}..."

        elif element_category == "Image":
            image_path = element.metadata.get("image_path")
            image_base64_from_metadata = element.metadata.get("image_base64")

            base64_image = None
            if image_path and os.path.exists(image_path):
                base64_image = _image_to_base64_str(image_path)
                logger.debug(
                    f"Image element (Parent ID: {parent_chunk_id}) found with path: {image_path}"
                )
            elif image_base64_from_metadata:
                base64_image = image_base64_from_metadata
                logger.debug(
                    f"Image element (Parent ID: {parent_chunk_id}) found with base64 directly in metadata."
                )
            else:
                logger.warning(
                    f"Image element (Parent ID: {parent_chunk_id}) found but no 'image_path' or 'image_base64' in metadata. Skipping image summarization."
                )

            if base64_image:
                raw_special_content_to_store = base64_image.encode("utf-8")
                logger.debug(f"Summarizing image from parent ID {parent_chunk_id}...")
                try:
                    # Use multimodal capabilities of gpt-4o-mini
                    response = llm_for_summaries.invoke(
                        [
                            HumanMessage(
                                content=[
                                    {
                                        "type": "text",
                                        "text": "Describe this image in detail, focusing on elements relevant to a document's content. Be concise and factual.",
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{base64_image}"
                                        },
                                    },
                                ]
                            )
                        ]
                    )
                    summary_content = response.content
                except Exception as e:
                    logger.error(
                        f"LLM summarization failed for image (Parent ID: {parent_chunk_id}): {e}"
                    )
                    summary_content = f"Image description failed. Original image path: {image_path or 'N/A'}."
            else:
                logger.warning(
                    f"Could not get base64 for image element (Parent ID: {parent_chunk_id}). Skipping summarization."
                )

        if summary_content and summary_content.strip():
            if raw_special_content_to_store:
                content_bytes = raw_special_content_to_store
                child_metadata_source = "raw"
            else:
                content_bytes = summary_content.encode("utf-8")
                child_metadata_source = "summary"
                logger.debug(
                    f"No raw bytes available for special content (Parent ID: {parent_chunk_id}). "
                    "Falling back to summary text for hashing."
                )
        else:
            logger.debug(
                f"Skipping special content element (Parent ID: {parent_chunk_id}) "
                "because no meaningful summary was produced."
            )
            continue

        special_hash = hashlib.sha256(content_bytes).hexdigest()[:8]
        special_content_id = f"{parent_chunk_id}_special_{special_hash}"

        if raw_special_content_to_store:
            parent_doc_store.mset([(special_content_id, raw_special_content_to_store)])
            logger.debug(
                f"Stored raw special content with ID: {special_content_id} (raw bytes)."
            )
        else:
            parent_doc_store.mset(
                [(special_content_id, summary_content.encode("utf-8"))]
            )
            logger.debug(
                f"No raw bytes available; stored summary text for special content ID: {special_content_id}."
            )

        child_metadata = element.metadata.copy()
        child_metadata["parent_chunk_id"] = parent_chunk_id
        child_metadata["special_content_id"] = special_content_id
        child_metadata["special_content_source"] = child_metadata_source
        category_label = element_category.lower() if element_category else "unknown"
        child_metadata["category"] = f"summary_{category_label}"
        child_metadata["text"] = summary_content
        referenced_chunk_id = _pick_referenced_chunk(
            parent_id=parent_chunk_id,
            page_number=element.metadata.get("page_number"),
            element_order=element.metadata.get("element_order"),
            element_start=element.metadata.get("start_index"),
        )

        if referenced_chunk_id:
            child_metadata["referenced_text_chunk_id"] = referenced_chunk_id

        parent_breadcrumbs = element.metadata.get("breadcrumbs")
        if isinstance(parent_breadcrumbs, list) and parent_breadcrumbs:
            breadcrumbs = list(parent_breadcrumbs)
        else:
            breadcrumbs = []
            file_name = child_metadata.get("file_name")
            if file_name:
                breadcrumbs.append(file_name)
            section_heading = child_metadata.get("section_heading")
            if section_heading:
                breadcrumbs.append(section_heading)
        child_metadata["breadcrumbs"] = breadcrumbs

        if "section_id" in element.metadata:
            child_metadata["section_id"] = element.metadata["section_id"]
        if "section_heading" in element.metadata:
            child_metadata["section_heading"] = element.metadata["section_heading"]

        counter_key = (parent_chunk_id, category_label)
        special_chunk_counters[counter_key] += 1
        child_metadata["chunk_index_in_section"] = special_chunk_counters[counter_key]

        child_metadata["child_chunk_id"] = deterministic_chunk_id(
            parent_chunk_id,
            summary_content,
            child_metadata.get("page_number", ""),
            child_metadata.get("start_index", ""),
        )

        sanitized_pruned_metadata = _shrink_metadata_to_limit(
            _canonicalize_and_filter_metadata(
                child_metadata,
                parsed_global_custom_metadata,
                document_specific_metadata_map.get(
                    child_metadata.get("file_name", ""), {}
                ),
                logger,
            ),
            logger,
            pinecone_metadata_max_bytes,
        )

        if sanitized_pruned_metadata.get("text_truncated", False):
            summary_content = sanitized_pruned_metadata["text"]
            logger.warning(
                f"Summary chunk (ID: {sanitized_pruned_metadata.get('child_chunk_id', 'N/A')}) "
                "text truncated during final metadata shrink."
            )

        special_content_child_chunks.append(
            Document(page_content=summary_content, metadata=sanitized_pruned_metadata)
        )
        logger.debug(
            f"Generated summary child chunk for {element_category} (Parent ID: {parent_chunk_id})."
        )

    logger.info(
        f"Generated {len(special_content_child_chunks)} summary child chunks for special content."
    )
    return special_content_child_chunks


# -------- API Connection Test Function --------
def test_api_connections(
    pinecone_api_key: str, embedding_api_key: str, logger: logging.Logger
):
    """
    Tests the provided Pinecone and OpenAI API keys for valid connections.
    Displays success or error messages in the Streamlit UI.
    """
    st.info("Attempting to test API connections...")

    # Test Pinecone API Key
    if pinecone_api_key:
        try:
            pc_test = Pinecone(api_key=pinecone_api_key)
            pc_test.list_indexes()  # A simple call to verify connection
            st.success("✅ Pinecone API Key is valid and connected.")
            logger.info("Pinecone API connection test successful.")
        except Exception as e:
            st.error(f"❌ Pinecone API Key test failed: {e}. Please check your key.")
            logger.error(f"Pinecone API connection test failed: {e}")
    else:
        st.warning("⚠️ Pinecone API Key not provided for testing.")

    # Test OpenAI Embeddings API Key
    if embedding_api_key:
        try:
            embed_model_test = OpenAIEmbeddings(
                openai_api_key=embedding_api_key, model="text-embedding-3-small"
            )
            embed_model_test.embed_query(
                "test query"
            )  # Generate a small embedding to verify
            st.success("✅ OpenAI API Key is valid and can generate embeddings.")
            logger.info("OpenAI API connection test successful.")
        except Exception as e:
            st.error(
                f"❌ OpenAI API Key test failed: {e}. Please check your key or model permissions."
            )
            logger.error(f"OpenAI API connection test failed: {e}")
    else:
        st.warning("⚠️ OpenAI API Key not provided for testing.")


# -------- Document Management UI --------
def manage_documents_ui(
    pinecone_api_key: str,
    pinecone_index_name: str,
    namespace: str,
    embedding_dimension: int,
):
    """
    Provides a Streamlit UI for managing existing documents in the Pinecone index.
    Allows viewing index statistics, loading document names, viewing metadata,
    and deleting individual or all documents within a namespace.
    """
    logger = st.session_state.app_logger
    logger.debug("Entering manage_documents_ui function.")
    logger.info(
        f"Current namespace setting in UI: '{namespace}' (empty string means default)."
    )

    # Initialize session state variables for UI control
    if "delete_pending" not in st.session_state:
        st.session_state.delete_pending = False
    if "file_to_delete_staged" not in st.session_state:
        st.session_state.file_to_delete_staged = ""
    if "docid_to_delete_staged" not in st.session_state:
        st.session_state.docid_to_delete_staged = ""
    if "bulk_delete_pending" not in st.session_state:
        st.session_state.bulk_delete_pending = False
    if "all_document_names" not in st.session_state:
        st.session_state.all_document_names = []
    if "metadata_display_doc_name" not in st.session_state:
        st.session_state.metadata_display_doc_name = None

    st.markdown("---")
    st.header("3. Manage Existing Documents")
    st.info(
        "Load, search, and manage documents currently stored in the selected Pinecone namespace. "
        "You can inspect metadata or delete specific documents directly from this panel."
    )
    st.caption(f"Current namespace: {namespace or 'Default'}")



    if not pinecone_api_key or not pinecone_index_name:
        st.info(
            "Please provide Pinecone API Key and Index Name in the configuration above to manage documents."
        )
        logger.info(
            "Skipping document management UI due to missing Pinecone API Key or Index Name."
        )
        return

    try:
        pc = Pinecone(api_key=pinecone_api_key)
        logger.info("Pinecone client initialized successfully for document management.")
    except Exception as e:
        logger.exception(
            "Failed to initialize Pinecone client for document management."
        )
        st.error(f"Failed to initialize Pinecone client: {e}")
        return

    index = None
    try:
        if pinecone_has_index_cached(pc, pinecone_index_name, logger, pinecone_api_key):
            index = pc.Index(pinecone_index_name)
            logger.info(f"Connected to Pinecone index '{pinecone_index_name}'.")
        else:
            logger.warning(f"Pinecone index '{pinecone_index_name}' does not exist.")

    except Exception as e:
        logger.exception(f"Could not resolve index info for '{pinecone_index_name}'.")
        st.warning(f"Could not resolve index info: {e}")

    if index:
        try:
            stats = index.describe_index_stats()
            st.subheader(f"Index Status: `{pinecone_index_name}`")
            st.metric(
                label="Total Vectors in Index",
                value=f"{stats.get('total_vector_count', 'N/A'):,}",
            )

            namespaces_info = stats.get("namespaces", {})
            namespace_to_check = namespace or "__default__"
            current_ui_ns_count = namespaces_info.get(namespace_to_check, {}).get(
                "vector_count", namespaces_info.get("", {}).get("vector_count", 0)
            )

            if namespace:
                st.metric(
                    label=f"Vectors in current namespace `'{namespace}'`",
                    value=f"{current_ui_ns_count:,}",
                )
            else:
                st.metric(
                    label=f"Vectors in default namespace",
                    value=f"{current_ui_ns_count:,}",
                )

            if namespaces_info:
                with st.expander("View All Namespaces & Counts"):
                    if namespaces_info:
                        for ns_name, ns_data in namespaces_info.items():
                            vector_count = ns_data.get("vector_count", 0)
                            display_ns_name = (
                                f"'{ns_name}'"
                                if ns_name
                                else "Default (`''` or `__default__`)"
                            )
                            st.write(
                                f"- Namespace {display_ns_name}: `{vector_count:,}` vectors"
                            )
                            logger.info(
                                f"Namespace '{ns_name}' has {vector_count} vectors."
                            )
                    else:
                        st.info("No namespaces found in this index.")
                        logger.info("No namespaces found in index stats.")

            logger.info(
                f"Displayed index stats for '{pinecone_index_name}'. Total vectors: {stats.get('total_vector_count')}, Current UI target namespace '{namespace_to_check}' vectors: {current_ui_ns_count}"
            )
        except Exception as e:
            logger.exception("Failed to retrieve or display index stats.")
            st.write(f"Index: {pinecone_index_name} (stats unavailable)")

    st.markdown("---")
    st.subheader("Your Documents in Current Namespace")
    col_limit, col_button = st.columns([0.5, 0.5])
    with col_limit:
        load_limit = st.number_input(
            "Max document names to load",
            min_value=1,
            max_value=10000,
            value=1000,
            step=100,
            help="Higher values may increase load time. Pinecone's query limit is 10,000 documents.",
        )
    with col_button:
        load_docs_clicked = st.button(
            "📥 Load Document Names",
            use_container_width=True,
            key="load_all_docs_button",
        )
        if IS_ON_HF:
            st.caption(
                "Fetched live from your Pinecone namespace. Ensure you’re using your own API key/index."
            )
    st.markdown("</div></div>", unsafe_allow_html=True)

    if load_docs_clicked:
        logger.info(f"Loading document names. Sandbox Mode (IS_ON_HF): {IS_ON_HF}")

        entries: list[dict] = []

        if IS_ON_HF:
            if not index:
                st.warning("Pinecone index not available; cannot list documents.")
                st.session_state.all_document_names = []
            else:
                with st.spinner("Fetching document list from Pinecone..."):
                    entries = _list_documents_via_pinecone(
                        index=index,
                        namespace=namespace or None,
                        logger=logger,
                    )
                if not entries:
                    st.info(
                        "No documents found in this namespace. Upload some files to get started."
                    )
                    st.session_state.all_document_names = []
        else:
            entries = _load_manifest()
            if not entries:
                if not index:
                    st.warning(
                        "Local manifest missing and Pinecone index is offline; cannot rebuild."
                    )
                    st.session_state.all_document_names = []
                else:
                    with st.spinner(
                        "Local manifest missing. Scanning Pinecone to rebuild (this may take time)..."
                    ):
                        _bootstrap_manifest_from_pinecone(
                            index=index,
                            namespace=namespace or None,
                            logger=logger,
                            embedding_dimension=embedding_dimension,
                        )
                    entries = _load_manifest()
                    if not entries:
                        st.warning(
                            "Bootstrap complete, but no documents were found in Pinecone."
                        )
                        st.session_state.all_document_names = []
                    else:
                        st.success(
                            f"Local manifest rebuilt with {len(entries)} documents found in Pinecone."
                        )

        if entries:
            all_names = sorted(
                {entry["file_name"] for entry in entries if entry.get("file_name")}
            )
            st.session_state.all_document_names = all_names[: load_limit]

            st.markdown(
                f"""
                <div style="background:#ecfdf5;border:1px solid #bbf7d0;border-radius:12px;padding:0.8rem 1rem;margin-bottom:0.7rem;color:#065f46;">
                    ✅ Successfully loaded {len(st.session_state.all_document_names)} document names (capped at {load_limit}).
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.session_state.document_entries = entries
            logger.debug(f"UI populated with {len(entries)} entries.")
        else:
            st.session_state.document_entries = []
            if not IS_ON_HF:
                st.info("Manifest is empty for this namespace.")
            logger.debug("No document entries returned; state cleared.")

        st.session_state.metadata_display_doc_name = None

    all_document_names = st.session_state.get("all_document_names", [])

    st.subheader("Search loaded documents")

    search_query = st.text_input(
        "",
        key="doc_search_input",
        help="Type to filter the list of loaded documents.",
        placeholder="Search by file name...",
        label_visibility="collapsed",
    )

    filtered_document_names = [
        name for name in all_document_names if search_query.lower() in name.lower()
    ]

    if filtered_document_names:
        st.markdown(
            f"""
            <p style="margin-bottom:0.4rem;color:#475569;">
                Showing {len(filtered_document_names)} of {len(all_document_names)} document names
            </p>
            """,
            unsafe_allow_html=True,
        )
        for doc_name in filtered_document_names:
            with st.expander(f"📄 {doc_name}", expanded=False):
                col_view, col_delete = st.columns([1, 1])
                with col_view:
                    if st.button(f"View Metadata", key=f"view_meta_{doc_name}"):
                        st.session_state.metadata_display_doc_name = doc_name
                        logger.info(f"Requested metadata view for document: {doc_name}")
                with col_delete:
                    if st.button(f"Delete Document", key=f"delete_doc_{doc_name}"):
                        st.session_state.file_to_delete_staged = doc_name
                        st.session_state.docid_to_delete_staged = ""
                        st.session_state.delete_pending = True
                        logger.info(f"Deletion staged for document: {doc_name}")
                        st.rerun()

        if st.session_state.metadata_display_doc_name:
            st.markdown("---")
            st.markdown(
                f"""
                <div style="border:1px solid #c7d2fe;background:#eef2ff;border-radius:16px;padding:1rem 1.3rem;margin-bottom:1rem;">
                    <h3 style="margin:0;color:#1d4ed8;">Metadata for: <code>{st.session_state.metadata_display_doc_name}</code></h3>
                    <p style="margin:0.4rem 0 0;color:#1e293b;font-size:0.9rem;">
                        Displaying metadata from a single representative chunk. Full document metadata may vary across chunks.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if index:
                try:
                    normalized_display_name_for_filter = (
                        st.session_state.metadata_display_doc_name.lower()
                    )
                    if IS_ON_HF:
                        entries = st.session_state.get("document_entries", [])
                    else:
                        entries = _load_manifest()
                    matching_entry = next(
                        (
                            e
                            for e in entries
                            if e.get("file_name") == normalized_display_name_for_filter
                            or e.get("document_id")
                            == st.session_state.metadata_display_doc_name
                        ),
                        None,
                    )

                    if not matching_entry:
                        st.warning("Document not found in manifest.")
                    else:
                        doc_id = matching_entry["document_id"]
                        query_response = None
                        try:
                            actual_dimension = None
                            try:
                                describe_response = pc.describe_index(pinecone_index_name)
                                if isinstance(describe_response, dict):
                                    actual_dimension = describe_response.get("dimension")
                                else:
                                    actual_dimension = getattr(describe_response, "dimension", None)
                            except Exception as describe_err:
                                logger.warning(f"Could not confirm index dimension: {describe_err}")

                            if actual_dimension and actual_dimension != embedding_dimension:
                                st.warning(
                                    f"Index '{pinecone_index_name}' uses dimension {actual_dimension}, "
                                    f"but the configuration form is set to {embedding_dimension}. "
                                    "Update the configuration to match before viewing metadata."
                                )
                            else:
                                vector_dimension = actual_dimension or embedding_dimension
                                probe_vector = [0.0] * vector_dimension
                                if probe_vector:
                                    probe_vector[0] = 1e-6

                                query_response = index.query(
                                    namespace=(namespace or None),
                                    filter={"document_id": doc_id},
                                    top_k=1,
                                    include_metadata=True,
                                    include_values=False,
                                    vector=probe_vector,
                                )

                        except Exception as e:
                            st.error(f"Metadata query failed: {e}")
                            logger.error(
                                f"Metadata query failed for document_id '{doc_id}': {e}"
                            )
                        else:
                            if query_response is None:
                                st.info(
                                    "Metadata lookup skipped because the index dimension does not match the configured embedding dimension."
                                )
                            else:
                                matches = getattr(
                                    query_response, "matches", None
                                ) or query_response.get("matches", [])
                                if matches and matches[0].metadata:
                                    st.json(matches[0].metadata)
                                else:
                                    st.warning(
                                        "No metadata found for this document (0 matching chunks)."
                                    )
                                    logger.warning(
                                        f"No chunk metadata returned for document '{st.session_state.metadata_display_doc_name}'."
                                    )

                except Exception as e:
                    logger.exception(
                        f"Error fetching metadata for '{st.session_state.metadata_display_doc_name}'."
                    )
                    st.error(f"Error fetching metadata: {e}")
            else:
                st.warning("Pinecone index not available to fetch metadata.")
    else:
        st.info("No documents loaded yet. Click 'Load Document Names' to begin.")

    st.markdown("---")
    st.subheader("Delete Specific Document by Name or ID")
    st.markdown(
        """
        <div style="border:1px solid #fee2e2;background:#fef2f2;border-radius:16px;padding:1rem 1.2rem;margin-bottom:1rem;">
            <p style="margin:0;color:#991b1b;font-size:0.9rem;">
                Enter the exact <strong>file_name</strong> (normalized to lowercase) or a full <strong>document_id</strong> (SHA256 hash) to remove all associated vectors.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    doc_name_or_id_to_delete = st.text_input(
        "",
        key="direct_delete_input",
        help="Enter the exact 'file_name' or 'document_id' of a document to delete its associated vectors.",
        placeholder="e.g., contract_v3.pdf or 8f14e45fceea167a5a36dedd4bea2543...",
        label_visibility="collapsed",
    )
    if st.button(
        "Initiate Deletion for Specific Document",
        key="initiate_direct_delete_button",
        use_container_width=True,
    ):
        if doc_name_or_id_to_delete:
            if len(doc_name_or_id_to_delete) == 64 and all(
                c in "0123456789abcdef" for c in doc_name_or_id_to_delete.lower()
            ):
                st.session_state.docid_to_delete_staged = doc_name_or_id_to_delete
                st.session_state.file_to_delete_staged = ""
                logger.info(
                    f"Deletion staged for specific document ID: {doc_name_or_id_to_delete}"
                )
            else:
                st.session_state.file_to_delete_staged = (
                    doc_name_or_id_to_delete.lower()
                )
                st.session_state.docid_to_delete_staged = ""
                logger.info(
                    f"Deletion staged for specific file name: {doc_name_or_id_to_delete.lower()}"
                )
            st.session_state.delete_pending = True
            st.rerun()
        else:
            st.warning("Please enter a Document Name or Document ID to delete.")

    st.markdown("---")
    st.subheader("Bulk Actions")
    st.markdown(
        f"""
        <div style="border:1px solid #fee2e2;background:#fff7ed;border-radius:16px;padding:1rem 1.3rem;margin-bottom:1rem;">
            <div style="display:flex;flex-wrap:wrap;gap:1rem;align-items:center;justify-content:space-between;">
                <div>
                    <p style="margin:0;color:#9a3412;font-size:0.92rem;">
                        Permanently delete <strong>all documents</strong> from index <code>{pinecone_index_name or 'N/A'}</code>
                        in namespace <code>{namespace or 'default'}</code>. This cannot be undone.
                    </p>
                </div>
                <div style="flex:0 0 auto;">
                    """,
        unsafe_allow_html=True,
    )
    bulk_delete_clicked = st.button(
        "Delete ALL documents in this namespace",
        key="bulk_delete_namespace_button",
        use_container_width=True,
    )
    st.markdown("</div></div>", unsafe_allow_html=True)

    if bulk_delete_clicked:
        st.session_state.bulk_delete_pending = True
        logger.info(
            f"Bulk deletion for namespace '{namespace or '__default__'}' prepared."
        )
        st.rerun()

    if st.session_state.bulk_delete_pending:
        st.markdown("---")
        st.subheader("Confirm Bulk Deletion")
        st.markdown(
            f"""
            <div style="border:1px solid #fecaca;background:#fef2f2;border-radius:16px;padding:1rem 1.3rem;margin-bottom:1rem;">
                <p style="margin:0;color:#991b1b;font-size:0.95rem;">
                    🚨 You are about to permanently delete <strong>all documents</strong> from Pinecone index 
                    <code>{pinecone_index_name}</code> in namespace <code>{namespace or '__default__'}</code>.
                </p>
                <p style="margin:0.4rem 0 0;color:#7f1d1d;font-size:0.88rem;">This action cannot be undone.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        bulk_confirm_checkbox = st.checkbox(
            "I understand this will delete ALL vectors in the current namespace permanently.",
            value=False,
            key="bulk_delete_confirm_checkbox",
        )

        col_bulk_exec, col_bulk_cancel = st.columns(2)
        with col_bulk_exec:
            if st.button(
                "Execute BULK Deletion",
                key="execute_bulk_delete_button",
                disabled=not bulk_confirm_checkbox,
                use_container_width=True,
            ):
                logger.info(
                    f"Execute BULK Deletion button clicked for namespace '{namespace or '__default__'}'."
                )
                if not index:
                    st.error(
                        "Index not available. Ensure Pinecone index exists and is reachable."
                    )
                    logger.error("Bulk deletion aborted: Index not available.")
                    st.session_state.bulk_delete_pending = False
                    st.rerun()
                    return
                try:
                    index.delete(delete_all=True, namespace=(namespace or None))
                    st.success(
                        f"Successfully initiated bulk deletion for namespace `'{namespace or '__default__'}'`."
                    )
                    logger.info(
                        f"Bulk deletion request sent for namespace '{namespace or '__default__'}'."
                    )
                    _clear_manifest()
                    time.sleep(3)
                    st.session_state.bulk_delete_pending = False
                    st.session_state.all_document_names = []
                    st.session_state.metadata_display_doc_name = None
                    st.rerun()
                except Exception as e:
                    logger.exception(
                        f"Bulk deletion failed for namespace '{namespace or '__default__'}'."
                    )
                    st.error(f"Bulk deletion failed: {e}")
                    st.session_state.bulk_delete_pending = False
                    st.rerun()
        with col_bulk_cancel:
            if st.button(
                "Cancel Bulk Deletion",
                key="cancel_bulk_delete_button",
                use_container_width=True,
            ):
                st.session_state.bulk_delete_pending = False
                st.info("Bulk deletion cancelled.")
                logger.info("Bulk deletion cancelled by user.")
                st.rerun()

    if st.session_state.delete_pending:
        st.markdown("---")
        st.subheader("Confirm Document Deletion")
        st.markdown(
            f"""
            <div style="border:1px solid #fed7aa;background:#fff7ed;border-radius:16px;padding:1rem 1.3rem;margin-bottom:1rem;">
                <p style="margin:0;color:#9a3412;font-size:0.95rem;">
                    ⚠️ You are about to permanently delete records in Pinecone index <code>{pinecone_index_name}</code> (namespace <code>{namespace or '__default__'}</code>).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.session_state.file_to_delete_staged:
            st.markdown(
                f"<p style='margin:0 0 0.2rem;color:#7c2d12;'>• <strong>File Name:</strong> <code>{st.session_state.file_to_delete_staged}</code></p>",
                unsafe_allow_html=True,
            )
        if st.session_state.docid_to_delete_staged:
            st.markdown(
                f"<p style='margin:0 0 0.2rem;color:#7c2d12;'>• <strong>Document ID:</strong> <code>{st.session_state.docid_to_delete_staged}</code></p>",
                unsafe_allow_html=True,
            )

        confirm_checkbox = st.checkbox(
            "I understand this action is permanent and cannot be undone.",
            value=False,
            key="delete_confirm_checkbox",
        )

        col_exec, col_cancel = st.columns(2)
        with col_exec:
            if st.button(
                "Execute Deletion",
                key="execute_delete_button",
                disabled=not confirm_checkbox,
                use_container_width=True,
            ):
                logger.info("Execute Deletion button clicked.")
                file_to_delete = st.session_state.file_to_delete_staged
                docid_to_delete = st.session_state.docid_to_delete_staged

                logger.debug(
                    f"Executing delete with: file_to_delete='{file_to_delete}', docid_to_delete='{docid_to_delete}', target_namespace='{namespace or '__default__'}'."
                )

                if not index:
                    st.error(
                        "Index not available. Ensure Pinecone index exists and is reachable."
                    )
                    logger.error("Deletion aborted: Index not available.")
                    st.session_state.delete_pending = False
                    st.rerun()
                    return

                try:
                    deleted_summaries = []

                    if file_to_delete:
                        # file_to_delete is already normalized if it came from direct input or loaded names
                        logger.info(
                            f"Attempting to delete records with file_name == '{file_to_delete}' in namespace '{namespace or '__default__'}'."
                        )
                        index.delete(
                            filter={"file_name": file_to_delete},
                            namespace=(namespace or None),
                        )
                        deleted_summaries.append(
                            f"Deleted records with file_name == '{file_to_delete}'"
                        )
                        logger.info(
                            f"Delete request sent for file_name '{file_to_delete}'."
                        )

                    if docid_to_delete:
                        # Assuming docid_to_delete is the document_id (hash of the original file)
                        logger.info(
                            f"Attempting to delete records with document_id == '{docid_to_delete}' in namespace '{namespace or '__default__'}'."
                        )
                        index.delete(
                            filter={"document_id": docid_to_delete},
                            namespace=(namespace or None),
                        )
                        deleted_summaries.append(
                            f"Deleted records with document_id == '{docid_to_delete}'"
                        )
                        logger.info(
                            f"Delete request sent for document_id '{docid_to_delete}'."
                        )

                    if docid_to_delete:
                        _remove_manifest_entry(docid_to_delete)

                    if file_to_delete:
                        manifest_entries = _load_manifest()
                        matching_ids = [
                            entry["document_id"]
                            for entry in manifest_entries
                            if entry.get("file_name") == file_to_delete
                        ]
                        for doc_id in matching_ids:
                            _remove_manifest_entry(doc_id)

                    st.success("Deletion successfully initiated:")
                    for s in deleted_summaries:
                        st.write(f"- {s}")
                    logger.info(f"Deletion summaries: {deleted_summaries}")

                    time.sleep(2)
                    logger.info(
                        "Paused for 2 seconds for Pinecone eventual consistency."
                    )

                    st.session_state.all_document_names = []
                    st.session_state.metadata_display_doc_name = None

                    try:
                        stats_after = index.describe_index_stats()
                        stats_after_dict = (
                            stats_after.to_dict()
                            if hasattr(stats_after, "to_dict")
                            else stats_after
                        )
                        st.info("Index stats (after delete):")
                        st.json(stats_after_dict)
                        logger.info(
                            f"Retrieved index stats after deletion: {stats_after_dict}"
                        )
                    except Exception as e:
                        logger.exception(
                            "Failed to retrieve index stats after deletion."
                        )
                        st.info(f"Index stats unavailable after delete: {e}")

                except Exception as e:
                    logger.exception("Delete operation failed.")
                    st.error(f"Delete operation failed: {e}")
                finally:
                    st.session_state.delete_pending = False
                    st.session_state.file_to_delete_staged = ""
                    st.session_state.docid_to_delete_staged = ""
                    st.rerun()

        with col_cancel:
            if st.button(
                "Cancel Deletion",
                key="cancel_delete_button",
                use_container_width=True,
            ):
                st.session_state.delete_pending = False
                st.session_state.file_to_delete_staged = ""
                st.session_state.docid_to_delete_staged = ""
                st.info("Deletion cancelled.")
                logger.info("Deletion cancelled by user.")
                st.rerun()
    logger.debug("Exiting manage_documents_ui function.")


# -------- Main Application Function --------
def main():
    """
    Main function for the Streamlit application.
    Configures the page, handles UI state, processes user inputs,
    and orchestrates the document ingestion and management workflows.
    """
    # Global Streamlit page configuration
    st.set_page_config(
        page_title="Pinecone Ingestor",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="auto",
    )

    st.title("📚 Pinecone Ingestor")
    st.write(
        "Transform your files into a searchable AI knowledge base. "
        "Upload documents, tune processing behavior, and upsert directly into Pinecone—without writing a single line of code."
    )
    st.markdown(
        "- **Smart chunking:** Automatically segments your content for high-precision retrieval.\n"
        "- **Resilient uploads:** Resume failed ingestions without re-spending token costs.\n"
        "- **Privacy-aware:** API keys stay on your device when running in the cloud."
    )
    st.markdown("---")

    load_dotenv()  # Load environment variables from .env file
    # Determine level from UI session state if it exists, otherwise environment
    stored_level = st.session_state.get("logging_level_selected")
    current_logging_level_name = stored_level if stored_level else os.environ.get(
        "LOGGING_LEVEL", DEFAULT_SETTINGS["LOGGING_LEVEL"]
    )   
    level_mapping = logging.getLevelNamesMapping()
    target_level_int = level_mapping.get(current_logging_level_name, logging.INFO)
    
    logger = setup_logging_once(level=target_level_int)
    logger.info("Environment variables loaded from .env (if present).")

    # --- BROWSER HYDRATION (HF ONLY) ---
    if IS_ON_HF:
        # 1. Check for payload FIRST (regardless of flag)
        if "config_payload" in st.query_params:
            try:
                payload = json.loads(st.query_params["config_payload"])
                if isinstance(payload, dict):
                    logger.info("Hydrating session state from browser storage...")
                    for key, value in payload.items():
                        os.environ[key] = str(value)
                    st.toast("Settings restored from browser cache", icon="🔄")
                
                st.session_state.browser_hydrated = True
                st.query_params.clear()
            except Exception as e:
                logger.error(f"Hydration failed: {e}")
                st.session_state.browser_hydrated = True 
        
        sync_browser_storage(action="load")

    # Load SpaCy model for NER filtering, caching it for performance
    nlp = load_spacy_model()
    if nlp is None:
        st.session_state.enable_ner_filtering = (
            False  # Disable NER if model failed to load
        )
        logger.warning("NER filtering disabled due to SpaCy model loading failure.")
    else:
        if "enable_ner_filtering" not in st.session_state:
            st.session_state.enable_ner_filtering = (
                os.environ.get("ENABLE_NER_FILTERING", "True").lower() == "true"
            )

    # Initialize session state variables for UI control and configuration
    if "show_reset_dialog" not in st.session_state:
        st.session_state.show_reset_dialog = False
    if "config_form_key" not in st.session_state:
        st.session_state.config_form_key = 0
    if "document_specific_metadata_map" not in st.session_state:
        st.session_state.document_specific_metadata_map = {}
    if "document_metadata_file_signature" not in st.session_state:
        st.session_state.document_metadata_file_signature = None

    if "dynamic_metadata_fields" not in st.session_state:
        env_metadata = os.environ.get(
            "CUSTOM_METADATA", DEFAULT_SETTINGS["CUSTOM_METADATA"]
        )
        try:
            st.session_state.dynamic_metadata_fields = json.loads(env_metadata)
            if not isinstance(st.session_state.dynamic_metadata_fields, list):
                st.session_state.dynamic_metadata_fields = []
        except json.JSONDecodeError:
            st.session_state.dynamic_metadata_fields = []

        if not st.session_state.dynamic_metadata_fields:
            st.session_state.dynamic_metadata_fields.append({"key": "", "value": ""})


    st.header("1. Configuration")
    st.write(
        "Set up your API keys, Pinecone index details, and document processing parameters."
    )

    port_col1, port_col2 = st.columns(2)
    with port_col1:
        st.markdown("**📤 Export current settings**")
        st.caption("Download the configuration currently stored in your environment or browser cache.")
        # EXPORT LOGIC:
        export_data = {
            "PINECONE_API_KEY": os.environ.get("PINECONE_API_KEY", ""),
            "EMBEDDING_API_KEY": os.environ.get("EMBEDDING_API_KEY", ""),
            "PINECONE_INDEX_NAME": os.environ.get("PINECONE_INDEX_NAME", ""),
            "PINECONE_CLOUD_REGION": os.environ.get("PINECONE_CLOUD_REGION", "aws-us-east-1"),
            "EMBEDDING_MODEL_NAME": os.environ.get("EMBEDDING_MODEL_NAME", "text-embedding-3-small"),
            "EMBEDDING_DIMENSION": os.environ.get("EMBEDDING_DIMENSION", "1536"),
            "METRIC_TYPE": os.environ.get("METRIC_TYPE", "cosine"),
            "NAMESPACE": os.environ.get("NAMESPACE", ""),
            "CHUNK_SIZE": os.environ.get("CHUNK_SIZE", "3600"),
            "CHUNK_OVERLAP": os.environ.get("CHUNK_OVERLAP", "540"),
            "CUSTOM_METADATA": os.environ.get("CUSTOM_METADATA", DEFAULT_SETTINGS["CUSTOM_METADATA"]),
            "OVERWRITE_EXISTING_DOCS": os.environ.get("OVERWRITE_EXISTING_DOCS", "False"),
            "LOGGING_LEVEL": os.environ.get("LOGGING_LEVEL", "INFO"),
            "ENABLE_FILTERING": os.environ.get("ENABLE_FILTERING", "True"),
            "WHITELISTED_KEYWORDS": os.environ.get("WHITELISTED_KEYWORDS", ""),
            "MIN_GENERIC_CONTENT_LENGTH": os.environ.get("MIN_GENERIC_CONTENT_LENGTH", "150"),
            "ENABLE_NER_FILTERING": os.environ.get("ENABLE_NER_FILTERING", "True"),
            "UNSTRUCTURED_STRATEGY": os.environ.get("UNSTRUCTURED_STRATEGY", "fast"),
            "SEMANTIC_CHUNKER_THRESHOLD_TYPE": os.environ.get("SEMANTIC_CHUNKER_THRESHOLD_TYPE", "percentile"),
            "SEMANTIC_CHUNKER_THRESHOLD_AMOUNT": os.environ.get("SEMANTIC_CHUNKER_THRESHOLD_AMOUNT", "98.0"),
            "MIN_CHILD_CHUNK_LENGTH": os.environ.get("MIN_CHILD_CHUNK_LENGTH", "100"),
            "KEEP_LOW_CONFIDENCE_SNIPPETS": os.environ.get("KEEP_LOW_CONFIDENCE_SNIPPETS", "False"),
        }
        st.download_button(
            label="Download profile (.json)",
            data=json.dumps(export_data, indent=4),
            file_name="pinecone_config_backup.json",
            mime="application/json",
            use_container_width=True,
        )

    with port_col2:
        st.markdown("**📥 Import settings from file**")
        st.caption("Upload a JSON profile to instantly hydrate this form. You can review it before saving.")
        uploaded_config = st.file_uploader(
            "Upload Settings JSON",
            type=["json"],
            label_visibility="collapsed",
        )
        if uploaded_config is not None:
            try:
                import_payload = json.load(uploaded_config)
                for k, v in import_payload.items():
                    os.environ[k] = str(v)
                if "CUSTOM_METADATA" in import_payload:
                    st.session_state.dynamic_metadata_fields = json.loads(import_payload["CUSTOM_METADATA"])
                st.success("✅ Settings imported to form! Click 'Save Configuration' below to persist.")
            except Exception as e:
                st.error(f"Failed to parse config: {e}")

    # Retrieve current API key values for pre-filling inputs and external button access
    pinecone_api_key_val = os.environ.get(
        "PINECONE_API_KEY", DEFAULT_SETTINGS["PINECONE_API_KEY"]
    )
    embedding_api_key_val = os.environ.get(
        "EMBEDDING_API_KEY", DEFAULT_SETTINGS["EMBEDDING_API_KEY"]
    )
    status_cols = st.columns(3)
    status_config = [
        ("Pinecone API Key", bool(pinecone_api_key_val)),
        ("OpenAI API Key", bool(embedding_api_key_val)),
        ("Index Name", bool(os.environ.get("PINECONE_INDEX_NAME", "").strip())),
    ]
    for col, (label, is_ready) in zip(status_cols, status_config):
        with col:
            st.caption(label)
            if is_ready:
                st.success("Ready")
            else:
                st.error("Missing")

    st.markdown(" ")

    with st.form(key=f"config_form_{st.session_state.config_form_key}"):
        # API Keys section within an expander for sensitive information
        api_card = st.container()
        with api_card:
            st.markdown(
                """
                <div style="background:linear-gradient(125deg,#0f172a,#1e3a8a);border-radius:18px;padding:1.2rem 1.4rem;margin-bottom:1.2rem;color:white;">
                    <div style="display:flex;align-items:center;justify-content:space-between;gap:0.6rem;flex-wrap:wrap;">
                        <div>
                            <h3 style="margin:0;">🔑 Secure Connections</h3>
                            <p style="margin:0;font-size:0.92rem;opacity:0.9;">Keys are encrypted in flight. On Hugging Face, they remain in your browser storage only.</p>
                        </div>
                        <span style="background:rgba(255,255,255,0.16);padding:0.35rem 0.9rem;border-radius:999px;font-size:0.85rem;">Never logged or shared</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            pinecone_api_key = st.text_input(
                "Pinecone API Key",
                type="password",
                value=pinecone_api_key_val,
                help="Your Pinecone API key used to authenticate API requests. Keep this secret and do not share it.",
            )
            embedding_api_key = st.text_input(
                "OpenAI API Key (for Embeddings)",
                type="password",
                value=embedding_api_key_val,
                help="Your OpenAI API key. Required to generate vector embeddings using OpenAI's models.",
            )

        st.subheader("Index & Model Settings")
        col_idx_left, col_idx_right = st.columns(2)

        with col_idx_left:
            pinecone_index_name = st.text_input(
                "Pinecone Index Name",
                value=os.environ.get(
                    "PINECONE_INDEX_NAME", DEFAULT_SETTINGS["PINECONE_INDEX_NAME"]
                ),
                help="The name of the Pinecone index where your vectors will be stored. Choose a unique name.",
            )

            region_options = list(SUPPORTED_PINECONE_REGIONS.keys())
            current_region = os.environ.get(
                "PINECONE_CLOUD_REGION", DEFAULT_SETTINGS["PINECONE_CLOUD_REGION"]
            )
            default_region_index = (
                region_options.index(current_region)
                if current_region in region_options
                else 0
            )
            pinecone_cloud_region = st.selectbox(
                "Pinecone Cloud Region",
                options=region_options,
                index=default_region_index,
                help=(
                    "Select the cloud provider and region where your Pinecone index will be hosted. "
                    "Free tier supports only 'aws-us-east-1'. Choose accordingly."
                ),
            )

            namespace = st.text_input(
                "Namespace (Optional)",
                value=os.environ.get("NAMESPACE", DEFAULT_SETTINGS["NAMESPACE"]),
                help=(
                    "Optional: A label to group documents together. This allows you to "
                    "search only specific subsets of your data later (e.g., 'LegalDocs' or 'Client_A'). "
                    "Leave blank to use the default namespace."
                ),
            )

        with col_idx_right:
            embedding_model_name = st.text_input(
                "OpenAI Embedding Model Name",
                value=os.environ.get(
                    "EMBEDDING_MODEL_NAME", DEFAULT_SETTINGS["EMBEDDING_MODEL_NAME"]
                ),
                help="The specific OpenAI embedding model to use (e.g., 'text-embedding-3-small').",
            )
            embedding_dimension = st.number_input(
                "Embedding Dimension",
                min_value=1,
                value=int(
                    os.environ.get(
                        "EMBEDDING_DIMENSION", DEFAULT_SETTINGS["EMBEDDING_DIMENSION"]
                    )
                ),
                help="Dimensionality of the embedding vectors. Typically 1536 for 'text-embedding-3-small' and 3072 for 'text-embedding-3-large'.",
            )
            metric_type = st.selectbox(
                "Pinecone Metric Type",
                options=["cosine", "euclidean", "dotproduct"],
                index=["cosine", "euclidean", "dotproduct"].index(
                    os.environ.get("METRIC_TYPE", DEFAULT_SETTINGS["METRIC_TYPE"])
                ),
                help=(
                    "Similarity metric used by Pinecone for vector search. "
                    "'Cosine' is the standard and recommended setting for most AI applications."
                ),
            )

        st.subheader("Document Processing Settings")
        proc_col_left, proc_col_right = st.columns(2)

        with proc_col_left:
            unstructured_strategy_options = ["hi_res", "fast", "auto"]
            current_unstructured_strategy = os.environ.get(
                "UNSTRUCTURED_STRATEGY", DEFAULT_SETTINGS["UNSTRUCTURED_STRATEGY"]
            )
            default_strategy_index = (
                unstructured_strategy_options.index(current_unstructured_strategy)
                if current_unstructured_strategy in unstructured_strategy_options
                else 0
            )
            unstructured_strategy = st.selectbox(
                "Document Scanning Precision",
                options=unstructured_strategy_options,
                index=default_strategy_index,
                help=(
                    "Choose how thoroughly to scan your files. "
                    "'hi_res' is recommended for complex PDFs with tables or images. "
                    "'fast' is best for simple text files. "
                    "'auto' lets the system choose."
                ),
            )

            chunk_size = st.number_input(
                "Search Segment Size (Characters)",
                min_value=256,
                value=int(os.environ.get("CHUNK_SIZE", DEFAULT_SETTINGS["CHUNK_SIZE"])),
                help=(
                    "Documents are broken into smaller segments so the AI can find specific answers. "
                    "Larger segments provide more context, while smaller segments are more precise."
                ),
            )
            chunk_overlap = st.number_input(
                "Topic Continuity (Overlap)",
                min_value=0,
                value=int(
                    os.environ.get("CHUNK_OVERLAP", DEFAULT_SETTINGS["CHUNK_OVERLAP"])
                ),
                help=(
                    "The number of characters repeated between segments. This ensures that a "
                    "sentence or topic isn't accidentally cut in half at the edge of a segment."
                ),
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with proc_col_right:


            whitelisted_keywords_input = st.text_input(
                "Whitelisted Keywords (comma-separated)",
                value=os.environ.get(
                    "WHITELISTED_KEYWORDS",
                    DEFAULT_SETTINGS["WHITELISTED_KEYWORDS"],
                ),
                help=(
                    "If a segment of text is very short but contains one of these words, "
                    "it will NOT be deleted. Useful for unique codes or product names."
                ),
                key="config_whitelisted_keywords_input",
            )
            min_generic_content_length_ui = st.number_input(
                "Minimum Segment Length (characters)",
                min_value=0,
                value=int(
                    os.environ.get(
                        "MIN_GENERIC_CONTENT_LENGTH",
                        DEFAULT_SETTINGS["MIN_GENERIC_CONTENT_LENGTH"],
                    )
                ),
                help=(
                    "The smallest number of characters required for a section to be kept. "
                    "Text shorter than this will be discarded unless it contains a name, date, or whitelisted keyword."
                ),
                key="config_min_generic_content_length_input",
            )
            enable_filtering = st.checkbox(
                "Enable Content Filtering",
                value=(os.environ.get("ENABLE_FILTERING", "True").lower() == "true"),
                help=(
                    "Removes background noise, headers/footers, and irrelevant fragments. "
                    "Disable to keep all extracted text, regardless of length or category."
                ),
                key="config_enable_filtering_checkbox",
            )

            keep_low_confidence = st.checkbox(
                "Preserve Short Fragments",
                value=(
                    os.environ.get("KEEP_LOW_CONFIDENCE_SNIPPETS", "False").lower()
                    == "true"
                ),
                help=(
                    "Keep very short lines of text that the system might otherwise ignore. "
                    "Enable this if you are uploading poetry, scripts, or itemized lists."
                ),
                key="config_keep_low_confidence_checkbox",
            )
            enable_ner_filtering = st.checkbox(
                "Smart Entity Protection",
                value=st.session_state.get(
                    "enable_ner_filtering",
                    DEFAULT_SETTINGS["ENABLE_NER_FILTERING"].lower() == "true",
                ),
                help=(
                    "Prevents the system from deleting short lines that contain important "
                    "identifiers like People, Organizations, or Dates. "
                    "Requires the SpaCy AI model to be loaded."
                ),
                disabled=(nlp is None),
                key="config_enable_ner_filtering_checkbox",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("Custom Metadata (Optional)")
        st.caption(
            "Attach a CSV or JSON file containing per-document tags such as author, department, or access level. "
            "Must include a 'file_name' column/key."
        )
        document_metadata_file = st.file_uploader(
            label="Attach metadata file (.csv or .json)",
            type=["csv", "json"],
            accept_multiple_files=False,
            label_visibility="collapsed",
        )
        st.caption("Example structure")
        st.table(
            {
                "file_name": ["contract.pdf", "handbook.docx"],
                "author": ["Jane Doe", "Ops Team"],
                "department": ["Legal", "HR"],
            }
        )
        st.markdown(
            """
            <div style="border:1px solid #fcd34d;background:#fffbeb;border-radius:14px;padding:0.9rem 1.1rem;margin:1rem 0;">
                <div style="display:flex;align-items:center;gap:0.6rem;">
                    <span style="font-size:1.2rem;">⚠️</span>
                    <div>
                        <strong style="color:#713f12;">Overwrite Existing Documents?</strong>
                        <p style="margin:0;color:#92400e;font-size:0.88rem;">Enable only if you want newer uploads to replace older versions when the file name is identical.</p>
                    </div>
                </div>
            """,
            unsafe_allow_html=True,
        )
        overwrite_existing_docs = st.checkbox(
            "Allow overwriting documents with the same file name",
            value=(
                os.environ.get("OVERWRITE_EXISTING_DOCS", "False").lower() == "true"
            ),
            help=(
                "If enabled, re-uploading a file with the same name will delete the old version "
                "from your database and replace it with the new one. If disabled, the system "
                "will skip files that are already present to prevent duplicates and save costs."
            ),
            key="config_overwrite_checkbox",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("⚙️ Advanced Chunking Settings", expanded=False):
            st.subheader("Automated Topic Discovery (Semantic Chunking)")
            st.caption(
                "Fine-tune how aggressively the system splits parent sections into smaller retrieval units."
            )
            semantic_chunker_threshold_type_options = [
                "percentile",
                "standard_deviation",
                "interquartile",
                "gradient",
            ]
            current_semantic_threshold_type = os.environ.get(
                "SEMANTIC_CHUNKER_THRESHOLD_TYPE", "percentile"
            )
            default_semantic_threshold_type_index = (
                semantic_chunker_threshold_type_options.index(
                    current_semantic_threshold_type
                )
                if current_semantic_threshold_type
                in semantic_chunker_threshold_type_options
                else 0
            )
            semantic_chunker_threshold_type = st.selectbox(
                "Topic Separation Method",
                options=semantic_chunker_threshold_type_options,
                index=default_semantic_threshold_type_index,
                help=(
                    "This setting determines how the AI identifies a 'break' between two topics:\n\n"
                    "• PERCENTILE (Recommended): Splits when the difference between sentences is in the top X% of all detected differences.\n"
                    "• STANDARD DEVIATION: Splits when a difference is significantly above average.\n"
                    "• INTERQUARTILE: Focuses on statistical outliers in the flow.\n"
                    "• GRADIENT: Looks for abrupt meaning changes, good for highly structured docs."
                ),
                key="config_semantic_chunker_threshold_type",
            )

            threshold_amount_min = 0.0
            threshold_amount_max = 100.0
            threshold_amount_value = 98.0
            threshold_amount_step = 0.1

            if semantic_chunker_threshold_type == "standard_deviation":
                threshold_amount_min = 0.1
                threshold_amount_max = 10.0
                threshold_amount_value = 3.0
                threshold_amount_step = 0.1
            elif semantic_chunker_threshold_type == "interquartile":
                threshold_amount_min = 0.1
                threshold_amount_max = 5.0
                threshold_amount_value = 1.5
                threshold_amount_step = 0.1
            elif semantic_chunker_threshold_type == "gradient":
                threshold_amount_min = 0.0
                threshold_amount_max = 100.0
                threshold_amount_value = 98.0
                threshold_amount_step = 0.1

            semantic_chunker_threshold_amount = st.slider(
                "Topic Transition Sensitivity",
                min_value=threshold_amount_min,
                max_value=threshold_amount_max,
                value=float(
                    os.environ.get(
                        "SEMANTIC_CHUNKER_THRESHOLD_AMOUNT",
                        str(threshold_amount_value),
                    )
                ),
                step=threshold_amount_step,
                format="%.1f",
                help=(
                    "How sensitive the AI is to changes in topic. Higher values = fewer, larger sections. "
                    "Lower values = more granular sections."
                ),
                key="config_semantic_chunker_threshold_amount",
            )

            min_child_chunk_length_ui = st.number_input(
                "Minimum Useful Segment Length (characters)",
                min_value=1,
                value=int(os.environ.get("MIN_CHILD_CHUNK_LENGTH", "100")),
                step=10,
                help=(
                    "Prevents the AI from creating segments that are too small to be useful. "
                    "Shorter than this will be merged with neighbors."
                ),
                key="config_min_child_chunk_length",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # Advanced Logging Settings
        with st.expander("🔎 Advanced Logging Settings", expanded=False):
            st.subheader("Control Log Verbosity")
            st.caption("Adjust log noise for troubleshooting or production monitoring.")
            logging_level_options = ["DEBUG", "INFO", "WARNING", "ERROR"]
            current_logging_level_name_from_env = os.environ.get(
                "LOGGING_LEVEL", DEFAULT_SETTINGS["LOGGING_LEVEL"]
            )
            default_logging_index = (
                logging_level_options.index(current_logging_level_name_from_env)
                if current_logging_level_name_from_env in logging_level_options
                else 1
            )
            logging_level_selected = st.selectbox(
                "Logging Level",
                options=logging_level_options,
                index=default_logging_index,
                help="Set the verbosity of the application logs. DEBUG is most verbose, ERROR is least verbose.",
                key="logging_level_selected",
            )

            current_level_name = logging.getLevelName(logger.level)
            if current_level_name != logging_level_selected:
                new_level_int = logging.getLevelNamesMapping().get(
                    logging_level_selected, logging.INFO
                )
                logger.setLevel(new_level_int)
                st.toast(f"Logger set to {logging_level_selected}", icon="🔧")
                logger.info(
                    f"Logging level dynamically set to {logging_level_selected}."
                )
            st.markdown("</div>", unsafe_allow_html=True)

        # Form submission buttons
        st.markdown(
            """
            <div style="display:flex;flex-wrap:wrap;gap:1rem;margin-top:1.4rem;">
                <div style="flex:1;min-width:220px;">
                """,
            unsafe_allow_html=True,
        )
        save_conf = st.form_submit_button(
            "✔️ Save Configuration",
            use_container_width=True,
        )
        st.markdown(
            """
                </div>
                <div style="flex:1;min-width:220px;">
                """,
            unsafe_allow_html=True,
        )
        reset_conf = st.form_submit_button(
            "♻️ Reset to Defaults",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Guarantee runtime value is False when SpaCy isn't available
    enable_ner_filtering = bool(enable_ner_filtering and nlp is not None)
    keep_low_confidence = bool(keep_low_confidence)
    st.session_state.enable_ner_filtering = enable_ner_filtering

    # Test API Connections button (outside the form, uses current state of API key inputs)
    if st.button(
        "Test API Connections", key="test_api_connections_button_outside_form"
    ):
        test_api_connections(pinecone_api_key, embedding_api_key, logger)

    # Logic for saving configuration to .env file
    # Logic for saving configuration
    if save_conf:
        # 1. Prepare the settings payload
        custom_metadata_json_string = json.dumps(st.session_state.dynamic_metadata_fields)
        to_save = {
            "PINECONE_API_KEY": pinecone_api_key,
            "EMBEDDING_API_KEY": embedding_api_key,
            "PINECONE_INDEX_NAME": pinecone_index_name,
            "PINECONE_CLOUD_REGION": pinecone_cloud_region,
            "EMBEDDING_MODEL_NAME": embedding_model_name,
            "EMBEDDING_DIMENSION": str(embedding_dimension),
            "METRIC_TYPE": metric_type,
            "NAMESPACE": namespace,
            "CHUNK_SIZE": str(chunk_size),
            "CHUNK_OVERLAP": str(chunk_overlap),
            "CUSTOM_METADATA": custom_metadata_json_string,
            "OVERWRITE_EXISTING_DOCS": str(overwrite_existing_docs),
            "LOGGING_LEVEL": logging_level_selected,
            "ENABLE_FILTERING": str(enable_filtering),
            "WHITELISTED_KEYWORDS": whitelisted_keywords_input,
            "MIN_GENERIC_CONTENT_LENGTH": str(min_generic_content_length_ui),
            "ENABLE_NER_FILTERING": str(enable_ner_filtering),
            "UNSTRUCTURED_STRATEGY": unstructured_strategy,
            "SEMANTIC_CHUNKER_THRESHOLD_TYPE": semantic_chunker_threshold_type,
            "SEMANTIC_CHUNKER_THRESHOLD_AMOUNT": str(semantic_chunker_threshold_amount),
            "MIN_CHILD_CHUNK_LENGTH": str(min_child_chunk_length_ui),
            "KEEP_LOW_CONFIDENCE_SNIPPETS": str(keep_low_confidence),
        }
        for key, value in to_save.items():
            os.environ[key] = value
        # 2. Handle Persistence based on environment
        if not IS_ON_HF:
            # Local: Save to .env
            with open(".env", "w") as f:
                for k, v in to_save.items():
                    f.write(f"{k}={v}\n")
            st.success("💾 Configuration saved locally to .env")
        else:
            # Hugging Face: Save to Browser Storage
            sync_browser_storage(data=to_save, action="save")
            st.success("✅ Configuration saved to your browser cache.")
            time.sleep(1) # Give JS a moment to execute before rerun
            
        st.rerun()

    # Logic for resetting configuration to defaults
    if reset_conf:
        st.session_state.show_reset_dialog = True
        st.rerun()


    if st.session_state.show_reset_dialog:
        warning_msg = "Reset ALL configuration to defaults? (affects only local .env)" if not IS_ON_HF else "Reset ALL configuration to defaults? (affects browser cache)"
        st.warning(warning_msg)
        keep_keys = st.checkbox("Keep API keys?", value=True)
        if st.button("Confirm reset"):
            defaults = DEFAULT_SETTINGS.copy()
            if keep_keys:
                defaults["PINECONE_API_KEY"] = os.environ.get("PINECONE_API_KEY", "")
                defaults["EMBEDDING_API_KEY"] = os.environ.get("EMBEDDING_API_KEY", "")

            st.session_state.dynamic_metadata_fields = [{"key": "", "value": ""}]
            defaults["CUSTOM_METADATA"] = json.dumps(
                st.session_state.dynamic_metadata_fields
            )

            if not IS_ON_HF:
                # Local: Write defaults to .env
                with open(".env", "w") as f:
                    for k, v in defaults.items():
                        f.write(f"{k}={v}\n")
            else:
                # Hugging Face: Save defaults to browser storage
                sync_browser_storage(data=defaults, action="save")

            # Update current environment memory
            for k, v in defaults.items():
                os.environ[k] = v
            
            st.session_state.show_reset_dialog = False
            st.session_state.config_form_key += 1          
            success_msg = "Configuration reset" if not IS_ON_HF else "Browser cache reset"
            st.success(success_msg)
            logger.info(f"{success_msg} to defaults.")
            time.sleep(1)
            st.rerun()

        if st.button("Cancel"):
            st.session_state.show_reset_dialog = False
            st.info("Reset cancelled")
            logger.info("Configuration reset cancelled.")
            st.rerun()

    st.markdown("---")
    st.subheader("Global Custom Metadata (Tags)")
    st.info(
        "Add metadata tags that will be attached to every document processed in this session. "
        "Helpful for project names, departments, sensitivity level, etc."
    )

    st.markdown(
        f"""
        <div style="border:1px solid #fee2e2;background:#fef2f2;border-radius:12px;padding:0.8rem 1rem;margin-bottom:0.8rem;">
            <p style="margin:0;color:#991b1b;font-size:0.9rem;">
                ⚠️ Some keys are reserved by the system and cannot be used: {', '.join(RESERVED_METADATA_KEYS)}.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Example")
    st.code('{"project_name": "MyRAGProject", "department": "Engineering"}', language="json")

    # Dynamic UI for adding/removing custom metadata fields
    for i, field in enumerate(st.session_state.dynamic_metadata_fields):
        cols = st.columns([0.45, 0.45, 0.1])
        with cols[0]:
            field["key"] = st.text_input(
                f"Key {i+1}",
                value=field["key"],
                key=f"meta_key_{i}",
                label_visibility="collapsed",
            )
        with cols[1]:
            field["value"] = st.text_input(
                f"Value {i+1}",
                value=field["value"],
                key=f"meta_value_{i}",
                label_visibility="collapsed",
            )
        with cols[2]:
            if len(st.session_state.dynamic_metadata_fields) > 1 or (
                len(st.session_state.dynamic_metadata_fields) == 1
                and i == 0
                and (field["key"] != "" or field["value"] != "")
            ):
                if st.button(
                    "🗑️", key=f"remove_meta_{i}", help="Remove this custom field"
                ):
                    st.session_state.dynamic_metadata_fields.pop(i)
                    st.rerun()
            else:
                st.empty()

    col_add, col_remove_last = st.columns([0.5, 0.5])
    with col_add:
        if st.button("➕ Add Custom Field", key="add_meta_field"):
            st.session_state.dynamic_metadata_fields.append({"key": "", "value": ""})
            st.rerun()
    with col_remove_last:
        if st.session_state.dynamic_metadata_fields and st.button(
            "➖ Remove Last Field", key="remove_last_meta_field"
        ):
            st.session_state.dynamic_metadata_fields.pop()
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")

    st.header("2. Upload Documents")
    st.markdown(
        """
        <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:16px;padding:1rem 1.3rem;margin-bottom:1rem;">
            <div style="display:flex;flex-wrap:wrap;justify-content:space-between;gap:0.8rem;">
                <div style="flex:1;min-width:220px;">
                    <p style="margin:0;color:#374151;font-size:0.95rem;">
                        Upload files to embed them and upsert into your Pinecone knowledge base.
                        You can resume failed ingestion runs without re-uploading or re-paying for embeddings.
                    </p>
                </div>
                <div style="flex:0 0 auto;background:white;border:1px solid #d1d5db;border-radius:12px;padding:0.7rem 1rem;">
                    <p style="margin:0;font-size:0.85rem;color:#111827;"><strong>Supported:</strong></p>
                    <p style="margin:0;color:#4b5563;font-size:0.85rem;">PDF, DOCX, PPTX, XLSX, TXT, Markdown, CSV, HTML, Images, and more</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="border:2px dashed #c7d2fe;border-radius:16px;padding:1rem 1.3rem;margin-bottom:1rem;background:#f8fafc;">
            <div style="display:flex;gap:1rem;align-items:center;flex-wrap:wrap;">
                <div style="font-size:2rem;">📄</div>
                <div>
                    <p style="margin:0;font-weight:600;color:#0f172a;">Drag & drop files or choose from your computer</p>
                    <p style="margin:0;color:#475569;font-size:0.9rem;">Multiple files are supported. Documents over 40MB are best handled on the local/Docker build.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_files = st.file_uploader(
        label="Upload Documents",
        type=[
            "pdf",
            "txt",
            "md",
            "docx",
            "xlsx",
            "pptx",
            "csv",
            "html",
            "xml",
            "eml",
            "epub",
            "rtf",
            "odt",
            "org",
            "rst",
            "tsv",
            "jpg",
            "jpeg",
            "png",
            "gif",
            "bmp",
            "tiff",
            "webp",
        ],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Supported file types include common document, spreadsheet, presentation, and image formats.",
    )

    # Display uploaded file names for better user experience
    if uploaded_files:
        st.markdown(
            """
            <div style="border:1px solid #e5e7eb;border-radius:16px;padding:1rem 1.2rem;margin-bottom:1rem;background:#ffffff;">
                <h4 style="margin:0 0 0.6rem;color:#0f172a;">Uploaded Files</h4>
            """,
            unsafe_allow_html=True,
        )
        for file in uploaded_files:
            size_kb = f"{len(file.getvalue()) / 1024:.1f} KB"
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;justify-content:space-between;border:1px solid #f1f5f9;border-radius:12px;padding:0.7rem 1rem;margin-bottom:0.5rem;">
                    <div style="display:flex;align-items:center;gap:0.75rem;">
                        <span style="font-size:1.4rem;">🗂️</span>
                        <div>
                            <p style="margin:0;color:#0f172a;font-weight:600;">{file.name}</p>
                            <p style="margin:0;color:#475569;font-size:0.85rem;">{size_kb}</p>
                        </div>
                    </div>
                    <span style="color:#2563eb;font-size:0.85rem;">Queued</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <div style="border:1px dashed #cbd5f5;background:#f8fafc;border-radius:16px;padding:1rem 1.2rem;text-align:center;">
                <p style="margin:0;color:#1e3a8a;font-size:0.95rem;">
                    No files selected yet. Drag and drop documents above or click to browse your computer.
                </p>
                <p style="margin:0.3rem 0 0;color:#475569;font-size:0.85rem;">
                    Need a sample? Try uploading a small PDF or DOCX to get started.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Input validation for the main processing button
    can_process = bool(
        pinecone_api_key
        and embedding_api_key
        and pinecone_index_name
        and uploaded_files
    )
    if not can_process:
        st.warning(
            "Please ensure Pinecone API Key, OpenAI API Key, Pinecone Index Name are set and at least one file is uploaded to enable processing."
        )

    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#2563eb,#1e3a8a);border-radius:18px;padding:1rem 1.3rem;margin-bottom:1rem;color:white;box-shadow:0 12px 28px rgba(37,99,235,0.25);">
            <div style="display:flex;align-items:center;gap:0.8rem;flex-wrap:wrap;">
                <div style="font-size:2rem;">🚀</div>
                <div>
                    <p style="margin:0;font-weight:600;font-size:1rem;">Process, Embed & Upsert</p>
                    <p style="margin:0;opacity:0.85;font-size:0.9rem;">Runs extraction, chunking, filtering, OpenAI embeddings, and Pinecone upserts in sequence.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button(
        "Start Processing",
        disabled=not can_process,
        use_container_width=True,
    ):
        logger.info("Process, Embed & Upsert button clicked.")
        logger.debug(
            f"Pinecone Index: {pinecone_index_name}, Namespace: {namespace or 'default'}, Region: {pinecone_cloud_region}"
        )
        logger.debug(
            f"Embedding Model: {embedding_model_name}, Dimension: {embedding_dimension}, Metric: {metric_type}"
        )
        logger.debug(
            f"Chunk Size: {chunk_size}, Chunk Overlap: {chunk_overlap}, Unstructured Strategy: {unstructured_strategy}"
        )
        logger.debug(f"Overwrite existing documents setting: {overwrite_existing_docs}")
        logger.debug(
            f"Filtering Enabled: {enable_filtering}, Whitelisted Keywords: '{whitelisted_keywords_input}', Min Generic Content Length: {min_generic_content_length_ui}, NER Filtering Enabled: {enable_ner_filtering}"
        )

        # Parse whitelisted keywords for efficient lookup
        whitelisted_keywords_set = {
            k.strip().lower()
            for k in whitelisted_keywords_input.split(",")
            if k.strip()
        }
        logger.info(f"Parsed whitelisted keywords: {whitelisted_keywords_set}")

        # --- CLOUD RESOURCE SAFETY GATES ---
        if uploaded_files and IS_ON_HF:
            if len(uploaded_files) > 15:
                st.error(
                    "Cloud limit reached: 15 files per session. Use the Local Docker version for bulk ingestion."
                )
                logger.warning(
                    "User attempted to upload more than 15 files on Hugging Face."
                )
                return

            for f in uploaded_files:
                if f.size > 40 * 1024 * 1024:  # 40MB limit
                    st.error(
                        f"File '{f.name}' is too large for the cloud (limit 40MB). Use the Local Docker version for large books."
                    )
                    logger.warning(
                        f"User attempted to upload a file over 40MB: {f.name}"
                    )
                    return
        # ----------------------------------

        # Process global custom metadata from UI inputs
        parsed_global_custom_metadata = {}
        for i, field in enumerate(st.session_state.dynamic_metadata_fields):
            key = field["key"].strip()
            value = field["value"]
            if key:
                if key in RESERVED_METADATA_KEYS:
                    logger.warning(
                        f"Global custom metadata: Key '{key}' is reserved and will be ignored or overwritten."
                    )
                    st.warning(
                        f"Warning: Custom metadata key '{key}' is reserved and will be ignored or overwritten."
                    )
                try:
                    parsed_value = json.loads(value)
                    parsed_global_custom_metadata[key] = parsed_value
                except (json.JSONDecodeError, TypeError):
                    parsed_global_custom_metadata[key] = value
        logger.info(
            f"Global custom metadata generated from dynamic fields: {parsed_global_custom_metadata}"
        )

        if not embedding_api_key:
            st.error("Embedding API key is required. Please configure it above.")
            logger.error("Embedding API key is missing. Aborting process.")
            return
        if not uploaded_files:
            st.warning("Please upload at least one file to process.")
            logger.warning("No files uploaded. Aborting process.")
            return

        # Process document-specific metadata file if uploaded
        document_specific_metadata_map = {}
        if document_metadata_file:
            file_bytes = document_metadata_file.getvalue()
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            file_signature = f"{document_metadata_file.name}:{file_hash}"
            needs_parse = (
                st.session_state.document_metadata_file_signature != file_signature
            )

            if needs_parse:
                try:
                    # 1. Handle Encoding (Try UTF-8, fallback to Latin-1 for Excel exports)
                    try:
                        file_content = file_bytes.decode("utf-8")
                    except UnicodeDecodeError:
                        file_content = file_bytes.decode("latin-1")

                    fname_lower = document_metadata_file.name.lower()
                    parsed_map = {}

                    # 2. Process CSV Files
                    if fname_lower.endswith(".csv"):
                        df = pd.read_csv(io.StringIO(file_content), sep=None, engine="python")
                        df.columns = [
                            str(c).lower().replace(" ", "_").strip() for c in df.columns
                        ]
                        if "file_name" not in df.columns:
                            st.error("Missing Column: Your CSV must have a column named 'file_name'.")
                            return
                        df = df.fillna("")

                        for _, row in df.iterrows():
                            target_file = str(row["file_name"]).lower().strip()
                            doc_md = {}
                            for k, v in row.drop("file_name").items():
                                if k in RESERVED_METADATA_KEYS or v == "":
                                    continue
                                val_str = str(v).strip()
                                if val_str.lower() == "true":
                                    doc_md[k] = True
                                elif val_str.lower() == "false":
                                    doc_md[k] = False
                                else:
                                    try:
                                        doc_md[k] = json.loads(val_str)
                                    except Exception:
                                        doc_md[k] = val_str
                            parsed_map[target_file] = doc_md

                    # 3. Process JSON Files
                    elif fname_lower.endswith(".json"):
                        json_data = json.loads(file_content)
                        if not isinstance(json_data, list):
                            st.error("JSON Error: The file must contain a list of objects: `[{...}, {...}]`")
                            return

                        for entry in json_data:
                            clean_entry = {
                                str(k).lower().replace(" ", "_"): v for k, v in entry.items()
                            }
                            if "file_name" not in clean_entry:
                                continue
                            target_file = str(clean_entry.pop("file_name")).lower().strip()
                            filtered_md = {
                                k: v for k, v in clean_entry.items() if k not in RESERVED_METADATA_KEYS
                            }
                            parsed_map[target_file] = filtered_md
                    else:
                        st.error("Unsupported metadata file type. Use .csv or .json.")
                        return

                    st.session_state.document_specific_metadata_map = parsed_map
                    st.session_state.document_metadata_file_signature = file_signature
                    st.success(
                        f"✅ Metadata Loaded: Found settings for {len(parsed_map)} files."
                    )
                    logger.info(
                        f"Metadata map built with {len(parsed_map)} entries (signature {file_signature})."
                    )
                except Exception as e:
                    logger.exception("Metadata File Error")
                    st.error(f"Failed to read metadata file: {e}")
                    return
            document_specific_metadata_map = st.session_state.document_specific_metadata_map
        else:
            if st.session_state.document_metadata_file_signature is not None:
                logger.info("Metadata file removed; clearing cached document metadata map.")
            st.session_state.document_specific_metadata_map = {}
            st.session_state.document_metadata_file_signature = None
            document_specific_metadata_map = {}


        # Initialize Pinecone client
        try:
            pc = Pinecone(api_key=pinecone_api_key) if pinecone_api_key else None
            if pc:
                logger.info("Pinecone client initialized for upsert process.")
            else:
                logger.warning(
                    "Pinecone API key not provided, Pinecone client not initialized."
                )
        except Exception as e:
            logger.exception(
                "Pinecone client initialization failed for upsert process."
            )
            st.error(f"Pinecone initialization error: {e}. Please check your API key.")
            pc = None

        # --- PHASE 1: Generate and Display Initial Processing Plan ---
        files_to_process_plan = []
        plan_summary_messages = []

        st.subheader("Processing Plan Summary")
        st.markdown(
            """
            <div style="border:1px solid #e5e7eb;border-radius:16px;padding:1rem 1.3rem;margin-bottom:1rem;background:#fcfdff;">
                <p style="margin:0 0 0.4rem;color:#475569;">Each file is evaluated before processing. You'll see whether it will be processed, skipped, or overwritten.</p>
            """,
            unsafe_allow_html=True,
        )
        plan_summary_placeholder = st.empty()

        manifest_entries = _load_manifest()
        manifest_doc_ids = {entry["document_id"] for entry in manifest_entries}

        for uploaded_file in uploaded_files:
            original_file_name = uploaded_file.name
            normalized_file_name = original_file_name.lower()  # Normalize file name
            file_bytes = uploaded_file.getvalue()
            document_id = deterministic_document_id(normalized_file_name, file_bytes)

            status_message = ""
            should_process = True

            if pc and pinecone_has_index_cached(pc, pinecone_index_name, logger, pinecone_api_key):
                idx = pc.Index(pinecone_index_name)
                if overwrite_existing_docs:
                    status_message = f"🔄 '{original_file_name}': Overwrite enabled. Existing records will be removed."
                    logger.info(f"Plan: Overwrite enabled for '{original_file_name}'.")
                else:
                    if document_id in manifest_doc_ids:
                        should_process = False
                        status_message = (
                            f"⏩ '{original_file_name}': SKIPPED "
                            "(already present per manifest, overwrite disabled)."
                        )
                        logger.info(
                            f"Plan: Skipped '{original_file_name}': found in manifest and overwrite disabled."
                        )
                    else:
                        try:
                            query_response = idx.query(
                                namespace=(namespace or None),
                                top_k=1,
                                filter={"document_id": document_id},
                                include_values=False,
                                include_metadata=False,
                                vector=[1e-9] * embedding_dimension,
                            )
                            matches = getattr(
                                query_response, "matches", None
                            ) or query_response.get("matches", [])
                            if matches:
                                should_process = False
                                status_message = (
                                    f"⏩ '{original_file_name}': SKIPPED "
                                    "(already present in Pinecone, overwrite disabled)."
                                )
                                logger.info(
                                    f"Plan: Skipped '{original_file_name}': found in Pinecone and overwrite disabled."
                                )
                            else:
                                status_message = (
                                    f"📄 '{original_file_name}': Will be processed "
                                    "(not found in manifest/Pinecone)."
                                )
                                logger.info(
                                    f"Plan: '{original_file_name}' will be processed."
                                )
                        except Exception as e:
                            logger.exception(
                                f"Plan: Presence check for '{original_file_name}' failed."
                            )
                            status_message = f"⚠️ '{original_file_name}': Presence check failed ({e}). Will attempt to process."
                            should_process = True
            else:
                status_message = f"📄 '{original_file_name}': Will be processed (Pinecone index not yet available or provided)."
                logger.info(
                    f"Plan: '{original_file_name}' will be processed. Pinecone client not ready."
                )

            plan_summary_messages.append(status_message)

            if should_process:
                files_to_process_plan.append(
                    (uploaded_file, document_id, normalized_file_name)
                )

        formatted_msgs = "\n".join(
            f"<li style='margin-bottom:0.2rem;'>{msg}</li>"
            for msg in plan_summary_messages
        )
        with plan_summary_placeholder.container():
            st.markdown(
                f"<ul style='padding-left:1.2rem;color:#0f172a;'>{formatted_msgs}</ul>",
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        logger.info(f"Initial file processing plan generated: {plan_summary_messages}")
        if not files_to_process_plan:
            st.error(
                "No documents selected for processing after initial plan. Please check your files and overwrite settings."
            )
            logger.error("No documents to process after initial plan. Aborting.")
            return

        # --- PHASE 2: Process Documents with Real-time Updates ---
        total_files_to_process = len(files_to_process_plan)
        st.subheader("Document Processing Progress")

        # Create Pinecone index if it doesn't exist
        if pc and not pinecone_has_index_cached(pc, pinecone_index_name, logger, pinecone_api_key):
            conf = SUPPORTED_PINECONE_REGIONS.get(
                pinecone_cloud_region, SUPPORTED_PINECONE_REGIONS["aws-us-east-1"]
            )
            logger.info(
                f"Pinecone index '{pinecone_index_name}' does not exist. Creating new index."
            )
            st.info(f"Creating Pinecone index '{pinecone_index_name}'...")
            with st.spinner("Waiting for index to become ready..."):
                try:
                    pc.create_index(
                        name=pinecone_index_name,
                        dimension=int(embedding_dimension),
                        metric=metric_type,
                        spec=ServerlessSpec(cloud=conf["cloud"], region=conf["region"]),
                    )
                except Exception as create_err:
                    logger.exception("Failed to create Pinecone index.")
                    st.error(f"Failed to create index '{pinecone_index_name}': {create_err}")
                    return

                max_wait_seconds = 180  # 3 minutes
                poll_interval = 2
                waited = 0
                while waited < max_wait_seconds:
                    try:
                        status_response = pc.describe_index(pinecone_index_name)
                        is_ready = status_response.status.get("ready") if hasattr(status_response, "status") else status_response.get("status", {}).get("ready")
                        if is_ready:
                            break
                    except Exception as status_err:
                        logger.warning(f"Error polling index status: {status_err}")
                    time.sleep(poll_interval)
                    waited += poll_interval
                else:
                    error_msg = (
                        f"Pinecone index '{pinecone_index_name}' did not become ready "
                        f"within {max_wait_seconds} seconds. Please check your Pinecone console."
                    )
                    st.error(error_msg)
                    logger.error(error_msg)
                    return
            cache = st.session_state.setdefault("pinecone_has_index_cache", {})
            scope = pinecone_api_key or "default"
            cache[f"has_index::{scope}::{pinecone_index_name}"] = True
            st.success(f"Pinecone index '{pinecone_index_name}' is ready.")
            logger.info(f"Pinecone index '{pinecone_index_name}' created and is ready.")

        index = (
            pc.Index(pinecone_index_name)
            if pc and pinecone_has_index_cached(pc, pinecone_index_name, logger, pinecone_api_key)
            else None
        )
        if not index:
            st.error(
                "Pinecone index could not be created or connected. Aborting document processing."
            )
            logger.error("Pinecone index not available for processing loop. Aborting.")
            return

        for file_idx, (uploaded_file, document_id, normalized_file_name) in enumerate(
            files_to_process_plan
        ):
            original_file_name = uploaded_file.name  # Keep original for display

            st.markdown(
                f"""
                <div style="border:1px solid #e5e7eb;border-radius:16px;padding:0.8rem 1rem;margin-bottom:1rem;background:#ffffff;box-shadow:0 8px 20px rgba(15,23,42,0.05);">
                    <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:0.6rem;">
                        <div>
                            <p style="margin:0;color:#0f172a;font-weight:600;">{original_file_name}</p>
                            <p style="margin:0;color:#475569;font-size:0.87rem;">Document {file_idx + 1} of {total_files_to_process}</p>
                        </div>
                        <span style="background:#e0f2ff;color:#0369a1;padding:0.25rem 0.8rem;border-radius:999px;font-size:0.85rem;">
                            Processing
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.status(
                f"Processing document: **{original_file_name}** ({file_idx + 1}/{total_files_to_process})",
                expanded=True,
            ) as status_container:
                temp_dir = None
                try:
                    status_container.update(
                        label=f"Processing document: **{original_file_name}** - Saving to temporary storage...",
                        state="running",
                    )

                    file_path, temp_dir, file_bytes = save_uploaded_file_to_temp(
                        uploaded_file
                    )
                    if not file_path:
                        raise Exception("Failed to save uploaded file.")

                    # Delete existing records if overwrite is enabled
                    if overwrite_existing_docs:
                        status_container.update(
                            label=f"Processing document: **{original_file_name}** - Deleting existing records...",
                            state="running",
                        )
                        target_namespace = namespace or None
                        try:
                            logger.info(
                                f"Attempting to delete records with document_id == '{document_id}' "
                                f"in namespace '{target_namespace or '__default__'}'."
                            )
                            try:
                                index.delete(
                                    filter={"document_id": document_id},
                                    namespace=target_namespace,
                                )
                            except NotFoundException:
                                logger.info(
                                    f"Namespace '{target_namespace or '__default__'}' missing when deleting document_id '{document_id}'. Skipping."
                                )

                            try:
                                index.delete(
                                    filter={"file_name": normalized_file_name},
                                    namespace=target_namespace,
                                )
                            except NotFoundException:
                                logger.info(
                                    f"Namespace '{target_namespace or '__default__'}' missing when deleting file_name '{normalized_file_name}'. Skipping."
                                )

                            status_container.write(
                                f"✅ Existing records for '{original_file_name}' "
                                f"(Document ID: {document_id}, Normalized File Name: {normalized_file_name}) removed."
                            )
                            logger.info(
                                f"Existing records for '{original_file_name}' deleted from Pinecone during processing."
                            )
                        except Exception as e:
                            status_container.write(
                                f"⚠️ Failed to delete existing records for '{original_file_name}': {e}"
                            )
                            logger.exception(
                                f"Failed to delete existing records for '{original_file_name}' during processing."
                            )

                    status_container.update(
                        label=f"Processing document: **{original_file_name}** - Loading content...",
                        state="running",
                    )
                    status_container.update(
                        label=f"Parsing and Sanitizing **{original_file_name}**...",
                        state="running",
                    )

                    loader = UnstructuredLoader(
                        file_path, strategy=unstructured_strategy
                    )
                    initial_elements = list(loader.lazy_load())

                    raw_elements = []
                    for element in initial_elements:
                        clean_text = surgical_text_cleaner(element.page_content or "")
                        if clean_text.strip():
                            element.page_content = clean_text
                            raw_elements.append(element)

                    status_container.write(
                        f"✅ Loaded {len(initial_elements)} parts. Sanitized to {len(raw_elements)} elements."
                    )
                    logger.info(f"Loaded and sanitized {original_file_name}")

                    status_container.write(
                        f"Loaded {len(raw_elements)} raw parts using '{unstructured_strategy}' strategy."
                    )
                    logger.debug(
                        f"UnstructuredLoader loaded {len(raw_elements)} raw parts for '{original_file_name}' with strategy '{unstructured_strategy}'."
                    )

                    for idx, element in enumerate(raw_elements):
                        element.metadata["document_id"] = document_id
                        element.metadata["file_name"] = normalized_file_name
                        element.metadata["original_file_path"] = file_path
                        element.metadata["element_order"] = idx

                    status_container.update(
                        label=f"Processing document: **{original_file_name}** - Creating parent documents...",
                        state="running",
                    )
                    parent_documents, parent_doc_store = _create_parent_documents(
                        raw_elements=raw_elements,
                        logger=logger,
                        embedding_model_name=embedding_model_name,
                    )

                    status_container.write(
                        f"✅ Created {len(parent_documents)} parent documents."
                    )
                    logger.info(
                        f"'{original_file_name}': Parent document creation complete."
                    )

                    if not parent_documents:
                        status_container.write(
                            f"⚠️ No parent documents could be created for '{original_file_name}'. This document will not be embedded."
                        )
                        logger.warning(
                            f"No parent documents for '{original_file_name}'. Aborting processing for this file."
                        )
                        status_container.update(
                            label=f"Document: **{original_file_name}** - No content for parent documents!",
                            state="warning",
                            expanded=False,
                        )
                        continue

                    status_container.update(
                        label=f"Processing document: **{original_file_name}** - Filtering parent document content...",
                        state="running",
                    )
                    filtered_parent_documents = _filter_parent_content(
                        parent_documents=parent_documents,
                        enable_filtering=enable_filtering,
                        whitelisted_keywords_set=whitelisted_keywords_set,
                        min_generic_content_length=min_generic_content_length_ui,
                        enable_ner_filtering=enable_ner_filtering,
                        nlp=nlp,
                        keep_low_confidence=keep_low_confidence,
                        logger=logger,
                    )
                    status_container.write(
                        f"✅ Filtered to {len(filtered_parent_documents)} meaningful parent documents."
                    )
                    logger.info(
                        f"'{original_file_name}': Parent document filtering complete."
                    )

                    if not filtered_parent_documents:
                        status_container.write(
                            f"⚠️ No meaningful content found for '{original_file_name}' after filtering parent documents. This document will not be embedded."
                        )
                        logger.warning(
                            f"No meaningful content for '{original_file_name}' after filtering parent documents. Aborting processing for this file."
                        )
                        status_container.update(
                            label=f"Document: **{original_file_name}** - No meaningful content after filtering!",
                            state="warning",
                            expanded=False,
                        )
                        continue

                    status_container.update(
                        label=f"Finding logical topic breaks in **{original_file_name}**...",
                        state="running",
                    )
                    child_chunks, current_parent_to_child_map = _generate_child_chunks(
                        filtered_parent_documents=filtered_parent_documents,
                        embedding_model_name=embedding_model_name,
                        embedding_api_key=embedding_api_key,
                        embedding_dimension=int(embedding_dimension),
                        pinecone_metadata_max_bytes=PINECONE_METADATA_MAX_BYTES,
                        parsed_global_custom_metadata=parsed_global_custom_metadata,
                        document_specific_metadata_map=document_specific_metadata_map,
                        logger=logger,
                        semantic_chunker_threshold_type=semantic_chunker_threshold_type,
                        semantic_chunker_threshold_amount=semantic_chunker_threshold_amount,
                        min_child_chunk_size=min_child_chunk_length_ui,
                        configured_chunk_size=chunk_size,  # Pass user-configured chunk size
                        configured_chunk_overlap=chunk_overlap,  # Pass user-configured chunk overlap
                    )
                    status_container.write(
                        f"✅ Generated {len(child_chunks)} semantic child chunks."
                    )
                    logger.info(
                        f"'{original_file_name}': Child chunk generation complete."
                    )
                    status_container.update(
                        label=f"Describing tables and images in **{original_file_name}**...",
                        state="running",
                    )
                    special_content_child_chunks = _process_special_elements(
                        raw_elements=raw_elements,
                        parent_doc_store=parent_doc_store,
                        parent_to_child_map=current_parent_to_child_map,
                        embedding_model_name=embedding_model_name,
                        embedding_api_key=embedding_api_key,
                        embedding_dimension=int(embedding_dimension),
                        pinecone_metadata_max_bytes=PINECONE_METADATA_MAX_BYTES,
                        parsed_global_custom_metadata=parsed_global_custom_metadata,
                        document_specific_metadata_map=document_specific_metadata_map,
                        logger=logger,
                    )
                    status_container.write(
                        f"✅ Generated {len(special_content_child_chunks)} summary child chunks for special content."
                    )
                    logger.info(
                        f"'{original_file_name}': Special content summarization complete."
                    )

                    if not child_chunks and not special_content_child_chunks:
                        status_container.write(
                            f"⚠️ No embeddable chunks (semantic or special) could be generated for '{original_file_name}'. This document will not be embedded."
                        )
                        logger.warning(
                            f"No embeddable chunks for '{original_file_name}'. Aborting processing for this file."
                        )
                        status_container.update(
                            label=f"Document: **{original_file_name}** - No embeddable chunks!",
                            state="warning",
                            expanded=False,
                        )
                        continue

                    all_embeddable_child_chunks = (
                        child_chunks + special_content_child_chunks
                    )
                    if all_embeddable_child_chunks:
                        chunk_token_lengths = [
                            count_tokens(
                                c.page_content or "", embedding_model_name, logger
                            )
                            for c in all_embeddable_child_chunks
                        ]

                        truncated_count = sum(
                            1
                            for c in all_embeddable_child_chunks
                            if c.metadata.get("text_truncated")
                        )

                        min_tokens = min(chunk_token_lengths)
                        max_tokens = max(chunk_token_lengths)
                        avg_tokens = sum(chunk_token_lengths) / len(chunk_token_lengths)

                        status_container.write(
                            f"Chunk token stats — min: {min_tokens}, max: {max_tokens}, avg: {avg_tokens:.1f}"
                        )

                        if truncated_count:
                            status_container.write(
                                f"⚠️ {truncated_count} chunks were truncated to fit metadata limits."
                            )
                            logger.warning(
                                f"{truncated_count} chunks for '{original_file_name}' were truncated by metadata shrinking."
                            )

                        total_file_tokens = sum(chunk_token_lengths)
                        logger.info(
                            f"PROCESSED: '{original_file_name}' | "
                            f"Total Chunks: {len(all_embeddable_child_chunks)} | "
                            f"Total Tokens: {total_file_tokens:,} | "
                            f"Avg Tokens/Chunk: {avg_tokens:.1f}"
                        )
                        if total_file_tokens > 100000:
                            logger.warning(f"Large document detected: '{original_file_name}' ({total_file_tokens:,} tokens). Ingestion may be slow.")

                    # Embedding and Upserting for the current file's chunks
                    status_container.update(
                        label=f"Processing document: **{original_file_name}** - Generating embeddings...",
                        state="running",
                    )
                    embed_model = OpenAIEmbeddings(
                        openai_api_key=embedding_api_key,
                        model=embedding_model_name,
                        dimensions=int(embedding_dimension),
                    )

                    # 1. Load the local progress for this specific document
                    doc_cache = load_doc_checkpoint(document_id)
                    all_chunks = all_embeddable_child_chunks

                    # 2. Filter for chunks that still need embedding
                    missing_chunks = [
                        c
                        for c in all_chunks
                        if c.metadata.get("child_chunk_id") not in doc_cache
                    ]

                    file_vectors = []

                    if missing_chunks:
                        logger.info(
                            f"Resuming {original_file_name}: {len(missing_chunks)} new chunks to embed."
                        )
                        status_container.write(
                            f"🔄 Resuming: {len(all_chunks) - len(missing_chunks)} cached, {len(missing_chunks)} new."
                        )

                        texts_to_send = [
                            c.metadata.get("_embedding_text", c.page_content)
                            for c in missing_chunks
                        ]

                        current_batch_texts = []
                        current_batch_tokens = 0
                        new_vectors = []

                        for i, text in enumerate(texts_to_send):
                            text_tokens = count_tokens(
                                text, embedding_model_name, logger
                            )

                            if text_tokens > OPENAI_MAX_TOKENS_PER_EMBEDDING_REQUEST:
                                error_msg = f"Chunk {i+1} of '{original_file_name}' exceeds embedding limit."
                                logger.error(error_msg)
                                raise ValueError(error_msg)

                            if (
                                current_batch_tokens + text_tokens
                                > OPENAI_MAX_TOKENS_PER_EMBEDDING_REQUEST
                            ) or (
                                len(current_batch_texts)
                                >= OPENAI_MAX_TEXTS_PER_EMBEDDING_REQUEST
                            ):

                                if current_batch_texts:
                                    batch_vectors = embed_model.embed_documents(
                                        current_batch_texts
                                    )
                                    new_vectors.extend(batch_vectors)
                                current_batch_texts = [text]
                                current_batch_tokens = text_tokens
                            else:
                                current_batch_texts.append(text)
                                current_batch_tokens += text_tokens

                            status_container.write(
                                f"Generating new embeddings: {len(new_vectors) + len(current_batch_texts)}/{len(texts_to_send)}"
                            )

                        if current_batch_texts:
                            batch_vectors = embed_model.embed_documents(
                                current_batch_texts
                            )
                            new_vectors.extend(batch_vectors)

                        # 3. Update the checkpoint on disk immediately
                        for c, vec in zip(missing_chunks, new_vectors):
                            doc_cache[c.metadata["child_chunk_id"]] = vec
                        save_doc_checkpoint(document_id, doc_cache)
                        logger.info(f"Checkpoint saved for {original_file_name}")
                    else:
                        status_container.write(
                            "✅ All embeddings recovered from local cache."
                        )

                    # 4. Reconstruct the final vector list for Pinecone
                    file_vectors = [
                        doc_cache[c.metadata["child_chunk_id"]] for c in all_chunks
                    ]

                    if len(file_vectors) != len(all_embeddable_child_chunks):
                        raise ValueError(
                            f"Mismatch: Expected {len(all_embeddable_child_chunks)} embeddings but got {len(file_vectors)} for '{original_file_name}'."
                        )

                    file_records = []
                    for c, vec in zip(all_embeddable_child_chunks, file_vectors):
                        chunk_id = c.metadata.get("child_chunk_id")
                        if not chunk_id:
                            chunk_id = deterministic_chunk_id(
                                document_id,
                                c.page_content or "",
                                c.metadata.get("page_number", ""),
                                c.metadata.get("start_index", ""),
                            )
                            logger.warning(
                                f"Chunk ID missing from Layer 3 output. Generated fallback ID: {chunk_id}"
                            )
                            c.metadata["child_chunk_id"] = chunk_id

                        if "text" not in c.metadata:
                            c.metadata["text"] = c.page_content
                            logger.warning(
                                f"Metadata 'text' field missing for chunk '{chunk_id}'. Added from page_content."
                            )

                        c.metadata.pop("_embedding_text", None)

                        final_metadata_for_upsert = c.metadata
                        file_records.append((chunk_id, vec, final_metadata_for_upsert))

                    logger.info(
                        f"Prepared {len(file_records)} records for upsert for '{original_file_name}'."
                    )

                    status_container.update(
                        label=f"Processing document: **{original_file_name}** - Upserting to Pinecone...",
                        state="running",
                    )
                    total_file_records = len(file_records)
                    for i in range(0, total_file_records, UPSERT_BATCH_SIZE):
                        batch = file_records[i : i + UPSERT_BATCH_SIZE]
                        index.upsert(vectors=batch, namespace=(namespace or None))
                        status_container.write(
                            f"Upserted batch {i//UPSERT_BATCH_SIZE + 1}/{(total_file_records + UPSERT_BATCH_SIZE - 1) // UPSERT_BATCH_SIZE} ({len(batch)} records) for '{original_file_name}'."
                        )
                        logger.debug(
                            f"Upserted batch for '{original_file_name}': {i//UPSERT_BATCH_SIZE + 1}/{(total_file_records + UPSERT_BATCH_SIZE - 1) // UPSERT_BATCH_SIZE}."
                        )

                    status_container.update(
                        label=f"Document: **{original_file_name}** processed successfully!",
                        state="complete",
                        expanded=False,
                    )
                    logger.info(
                        f"Successfully processed and upserted all records for '{original_file_name}'."
                    )
                    clear_doc_checkpoint(
                        document_id
                    )  # Process 100% complete, delete temp cache
                    _append_manifest_entry(
                        document_id=document_id, file_name=normalized_file_name
                    )

                except Exception as e:
                    logger.exception(
                        f"Error processing document '{original_file_name}'."
                    )
                    status_container.error(
                        f"Error processing '{original_file_name}': {e}"
                    )
                    status_container.update(
                        label=f"Document: **{original_file_name}** failed!",
                        state="error",
                        expanded=True,
                    )
                finally:
                    if temp_dir and os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                        logger.debug(
                            f"Cleaned up temp directory for '{original_file_name}': {temp_dir}"
                        )

        st.markdown(
            """
            <div style="background:linear-gradient(135deg,#22c55e,#16a34a);border-radius:18px;padding:1.1rem 1.4rem;margin:1.2rem 0;color:white;box-shadow:0 14px 32px rgba(34,197,94,0.25);">
                <div style="display:flex;align-items:center;gap:0.9rem;flex-wrap:wrap;">
                    <div style="font-size:2rem;">✅</div>
                    <div>
                        <p style="margin:0;font-size:1.05rem;font-weight:600;">Ingestion Complete</p>
                        <p style="margin:0;opacity:0.9;font-size:0.92rem;">All selected documents have been processed or skipped per your plan. You can now explore them via the "Manage Documents" section below.</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        logger.info("All selected documents processing complete.")


        try:
            stats = index.describe_index_stats()
            stats_dict = stats.to_dict()
            total_vectors = stats_dict.get("total_vector_count", "N/A")
            if isinstance(total_vectors, (int, float)):
                total_vectors_display = f"{total_vectors:,}"
            else:
                total_vectors_display = str(total_vectors)

            namespaces = stats_dict.get("namespaces", {})

            st.markdown(
                """
                <div style="border:1px solid #dbeafe;background:#eff6ff;border-radius:18px;padding:1rem 1.3rem;margin-bottom:1rem;">
                    <h4 style="margin:0 0 0.4rem;color:#1d4ed8;">Final Index Summary</h4>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <p style="margin:0;color:#0f172a;font-size:0.95rem;">
                    • Total vectors: <strong>{total_vectors_display}</strong>
                </p>
                """,
                unsafe_allow_html=True,
            )
            if namespaces:
                st.markdown(
                    "<p style='margin:0.6rem 0 0;color:#1e3a8a;font-size:0.9rem;'>Namespace breakdown:</p>",
                    unsafe_allow_html=True,
                )
                for ns_name, ns_data in namespaces.items():
                    ns_vectors = ns_data.get("vector_count", "N/A")
                    ns_vectors_display = (
                        f"{ns_vectors:,}" if isinstance(ns_vectors, (int, float)) else str(ns_vectors)
                    )
                    display_ns_name = ns_name if ns_name else "Default (`''`)"
                    st.markdown(
                        f"<p style='margin:0;color:#0f172a;font-size:0.88rem;'>• {display_ns_name}: <strong>{ns_vectors_display}</strong> vectors</p>",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    "<p style='margin:0.6rem 0 0;color:#475569;font-size:0.9rem;'>No namespace-specific data returned.</p>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

            logger.info(f"Final index stats: {stats_dict}")
        except Exception as e:
            logger.exception("Failed to retrieve final index stats.")
            st.info("Final index stats unavailable.")

    # Render document management UI
    st.markdown("---")
    manage_documents_ui(
        pinecone_api_key, pinecone_index_name, namespace, int(embedding_dimension)
    )

    # Application Logs section
    st.markdown("---")
    st.header("🛠️ System Diagnostics")
    st.markdown(
        """
        <div style="border:1px solid #e5e7eb;border-radius:16px;padding:1rem 1.3rem;margin-bottom:1rem;background:#f8fafc;">
            <p style="margin:0;color:#475569;font-size:0.92rem;">
                Review application logs, wipe browser-side configuration caches (Hugging Face), or clear the in-app log history.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if IS_ON_HF:
        st.markdown(
            """
            <div style="border:1px solid #e0f2fe;background:#ecfeff;border-radius:12px;padding:0.9rem 1rem;margin-bottom:0.8rem;">
                <p style="margin:0;color:#0369a1;font-size:0.9rem;">
                    💡 Your API keys and settings are stored locally in your browser's cache. Nothing is stored on the server.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            "🗑️ Wipe Browser Cache",
            help="Permanently delete your saved settings from this browser.",
            use_container_width=True,
        ):
            sync_browser_storage(action="clear")

    with st.expander("View Application Logs", expanded=False):
        if "streamlit_handler" in st.session_state:
            records = st.session_state.streamlit_handler.get_records()
            if records:
                current_id = st.session_state.session_id
                display_logs = [r for r in records if current_id in r or "AppLogger_" not in r]
                st.markdown(
                    """
                    <div style="border:1px solid #e2e8f0;border-radius:12px;padding:0.6rem 0.8rem;background:#1e1e1e;color:#f5f5f5;font-family:monospace;font-size:0.83rem;max-height:320px;overflow:auto;">
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<br>".join(reversed(display_logs)),
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("No activity logged yet.")
        
        if st.button("Clear History", use_container_width=True):
            st.session_state.streamlit_handler.log_records.clear()
            st.rerun()

if __name__ == "__main__":
    main()
