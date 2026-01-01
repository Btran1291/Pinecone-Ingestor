import streamlit as st
import os
import tempfile
import shutil
import json
import hashlib
import re
import uuid
import time
import logging
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

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
except Exception as e:
    print(f"An unexpected error occurred during NLTK data download: {e}")

# -------- Logging Configuration --------
class StreamlitLogHandler(logging.Handler):
    """
    A custom logging handler that captures log records and stores them in a deque
    for display within a Streamlit application.
    """

    def __init__(self, max_records=100):
        super().__init__()
        self.log_records = deque(maxlen=max_records)
        self.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

    def emit(self, record):
        """Appends formatted log record to the deque."""
        self.log_records.append(self.format(record))

    def get_records(self):
        """Retrieves all stored log records."""
        return list(self.log_records)


def setup_logging_once(level=logging.INFO):
    """
    Initializes the application logger and attaches the StreamlitLogHandler.
    Ensures that the logger is set up only once per session.
    """
    if "streamlit_handler" not in st.session_state:
        st.session_state.streamlit_handler = StreamlitLogHandler()

    if "app_logger" not in st.session_state:
        st.session_state.app_logger = logging.getLogger(__name__)
        st.session_state.app_logger.setLevel(level)

        if not any(
            isinstance(h, StreamlitLogHandler)
            for h in st.session_state.app_logger.handlers
        ):
            st.session_state.app_logger.addHandler(st.session_state.streamlit_handler)
        if not any(
            isinstance(h, logging.StreamHandler)
            for h in st.session_state.app_logger.handlers
        ):
            st.session_state.app_logger.addHandler(logging.StreamHandler())

        st.session_state.app_logger.info(
            f"Logger initialized with level {logging.getLevelName(level)}."
        )

    return st.session_state.app_logger


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
    "UNSTRUCTURED_STRATEGY": "hi_res",
    "MIN_CHUNK_LENGTH": "150",
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
def save_uploaded_file_to_temp(uploaded_file):
    """
    Saves an uploaded Streamlit file to a temporary directory on the server.
    Returns the file path, the temporary directory path, and the file bytes.
    """
    logger = st.session_state.app_logger
    logger.debug(f"Attempting to save uploaded file: {uploaded_file.name}")
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, uploaded_file.name)
        file_bytes = uploaded_file.getvalue()
        with open(file_path, "wb") as f:
            f.write(file_bytes)
        logger.info(
            f"Successfully saved {uploaded_file.name} to temporary path: {file_path}"
        )
        return file_path, temp_dir, file_bytes
    except (IOError, OSError) as e:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
        logger.exception(f"Error saving uploaded file {uploaded_file.name}")
        st.error(f"Error saving uploaded file: {e}")
        return None, None, None
    except Exception as e:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
        logger.exception(
            f"An unexpected error occurred while saving uploaded file {uploaded_file.name}"
        )
        st.error(f"An unexpected error occurred: {e}")
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


MANIFEST_PATH = "pinecone_manifest.json"


def _load_manifest() -> list[dict]:
    if not os.path.exists(MANIFEST_PATH):
        return []
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_manifest(entries: list[dict]):
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def _append_manifest_entry(document_id: str, file_name: str):
    entries = _load_manifest()
    # Remove existing entry for same document_id
    entries = [e for e in entries if e.get("document_id") != document_id]
    entries.append({"document_id": document_id, "file_name": file_name})
    _save_manifest(entries)


def _remove_manifest_entry(document_id: str):
    entries = _load_manifest()
    entries = [e for e in entries if e.get("document_id") != document_id]
    _save_manifest(entries)


def _clear_manifest():
    _save_manifest([])


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


def _is_structural_noise(text: str) -> bool:
    """
    A surgical filter to prevent watermarks and boilerplates from becoming
    chapter titles or breadcrumbs.
    """
    if not text:
        return True

    # 1. Preparation: Remove visual decoration (e.g. '*** Header ***' -> 'header')
    # and lower the case for matching.
    cleaned_text = text.lower().strip(" \t\n\r*-_=<>[]")

    # 2. Pure Decoration Check
    # If the line has NO alphanumeric characters after stripping, it's noise.
    # Safe for technical docs because things like "[A] + [B]" have A and B.
    if cleaned_text and not any(c.isalnum() for c in cleaned_text):
        return True

    # 3. High-Confidence Keyword Match
    # We check the master list of commercial artifacts.
    NOISE_TARGETS = {
        "feboqok",
        "febokok",
        "abisource",
        "unregistered",
        "trial version",
        "evaluation copy",
        "demo watermark",
        "demo version",
        "converted by",
        "produced by",
        "watermarked",
        "nitro pdf",
        "abbyy finereader",
        "all rights reserved",
        "click here to buy",
        "ocr technology",
        "www.",
        ".com",
    }

    # Check if ANY target exists in the cleaned text
    if any(target in cleaned_text for target in NOISE_TARGETS):
        return True

    # 4. Short-Line Numeric Noise (e.g. just a page number "88 of 200")
    # This prevents page numbers from becoming 'Chapter Titles'
    if len(cleaned_text) < 15 and re.match(
        r"^(page\s?\d+|[0-9]+\s?of\s?[0-9]+)$", cleaned_text
    ):
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
    Adjusts the semantic chunker threshold so short parents split less aggressively
    and long parents split more aggressively.
    """
    if parent_tokens <= 0:
        return base_amount

    # Heuristic:
    # - Very short (<800 tokens): lower threshold (easier to split)
    # - Medium (800–4000): stay near base
    # - Very long (>4000): raise threshold (stricter splits)
    if parent_tokens < 800:
        return max(75.0, base_amount - 10.0)
    if parent_tokens > 4000:
        return min(99.0, base_amount + 4.0)
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

    # Try removing lowest priority items first (reverse of prioritized)
    pruned = dict(metadata)  # copy
    for key in reversed(prioritized):
        if len(json.dumps(pruned).encode("utf-8")) <= pinecone_metadata_max_bytes:
            break
        if key == "text":
            # don't remove text yet
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
        preview = unique_keys[:5]
        suffix = "..." if len(unique_keys) > 5 else ""
        logger.warning(f"Metadata shrink removed keys: {preview}{suffix}")

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
    MAJOR_STRUCTURAL_BREAKS = {
        "Title",
        "Header",
        "Subheader",
        "NarrativeText",
        "ListItem",
    }

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
        if element_category in MAJOR_STRUCTURAL_BREAKS:
            # If it's a heading-like category, it's a strong break
            if element_category in {"Title", "Header", "Subheader"}:
                is_major_break = True
            # If it's a NarrativeText or ListItem, but it's very short and distinct,
            # it might also indicate a new logical section.
            elif len(element_text) < SHORT_ELEMENT_AS_BREAK_THRESHOLD and (
                not current_parent_content
                or (  # Always start a new parent if it's the first element
                    current_parent_content
                    and len(" ".join(current_parent_content))
                    > SHORT_ELEMENT_AS_BREAK_THRESHOLD * 2
                )
            ):  # Only break if current parent has significant content
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

    max_child_chunk_tokens = 1000
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

        for idx, chunk_doc in enumerate(temp_child_chunks):
            prev_id = (
                temp_child_chunks[idx - 1].metadata.get("child_chunk_id")
                if idx > 0
                else None
            )
            next_id = (
                temp_child_chunks[idx + 1].metadata.get("child_chunk_id")
                if idx < len(temp_child_chunks) - 1
                else None
            )

            if prev_id:
                chunk_doc.metadata["previous_chunk_id"] = prev_id
            if next_id:
                chunk_doc.metadata["next_chunk_id"] = next_id

        all_child_chunks.extend(temp_child_chunks)

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
    st.write(
        "Load, search, and manage individual documents in the selected Pinecone namespace."
    )

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
        if pc.has_index(pinecone_index_name):
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

    load_limit = st.number_input(
        "Max Document Names to Load (for display/management)",
        min_value=1,
        max_value=10000,
        value=1000,
        step=100,
        help="Specify the maximum number of document names to attempt to load from Pinecone. Higher values may be slower and might not retrieve all names if the index is very large (Pinecone's query limit is 10,000).",
    )

    if st.button("Load Document Names", key="load_all_docs_button"):
        logger.info("Loading document names from manifest.")
        entries = _load_manifest()

        if not entries:
            if not index:
                st.warning("Manifest missing and index unavailable; cannot bootstrap.")
                st.session_state.all_document_names = []
            else:
                with st.spinner(
                    "Manifest missing. Scanning Pinecone to rebuild (may take time)..."
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
                        "Failed to rebuild manifest. Upload documents first or check permissions."
                    )
                    st.session_state.all_document_names = []
                else:
                    st.success(f"Manifest rebuilt with {len(entries)} entries.")

        if entries:
            st.session_state.all_document_names = sorted(
                {entry["file_name"] for entry in entries if entry.get("file_name")}
            )
            st.success(
                f"Loaded {len(st.session_state.all_document_names)} document names."
            )
        st.session_state.metadata_display_doc_name = None

    all_document_names = st.session_state.get("all_document_names", [])

    search_query = st.text_input(
        "Filter loaded documents by name:",
        key="doc_search_input",
        help="Type to filter the list of loaded documents.",
        value="",
    )

    # Filter against the already normalized document names
    filtered_document_names = [
        name for name in all_document_names if search_query.lower() in name.lower()
    ]

    if filtered_document_names:
        st.write(
            f"Displaying {len(filtered_document_names)} of {len(all_document_names)} loaded documents:"
        )
        for doc_name in filtered_document_names:
            with st.expander(f"📄 **{doc_name}**"):
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
            st.subheader(
                f"Metadata for: `{st.session_state.metadata_display_doc_name}`"
            )
            st.info(
                "Displaying metadata from a single representative chunk. Full document metadata may vary across chunks."
            )
            if index:
                try:
                    # Use the normalized display name for the filter
                    normalized_display_name_for_filter = (
                        st.session_state.metadata_display_doc_name.lower()
                    )
                    query_filter = {"file_name": normalized_display_name_for_filter}
                    # Heuristic to check if it's likely a document_id (SHA256 hash)
                    if (
                        "_" not in st.session_state.metadata_display_doc_name
                        and len(st.session_state.metadata_display_doc_name) == 64
                        and all(
                            c in "0123456789abcdef"
                            for c in st.session_state.metadata_display_doc_name.lower()
                        )
                    ):
                        query_filter = {
                            "document_id": st.session_state.metadata_display_doc_name
                        }

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
                        try:
                            probe_vector = [0.0] * embedding_dimension
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
    doc_name_or_id_to_delete = st.text_input(
        "Enter Document Name or Document ID to Delete (case-sensitive)",
        key="direct_delete_input",
        help="Enter the exact 'file_name' or 'document_id' of a document to delete its associated vectors. This is useful if the document is not listed above or if you want to delete by its unique ID.",
    )
    if st.button(
        "Initiate Deletion for Specific Document", key="initiate_direct_delete_button"
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
                # Normalize user input for file_name deletion
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
    st.write("Perform actions on all documents within the current namespace.")

    if st.button(
        "Delete ALL documents in current namespace", key="bulk_delete_namespace_button"
    ):
        st.session_state.bulk_delete_pending = True
        logger.info(
            f"Bulk deletion for namespace '{namespace or '__default__'}' prepared."
        )
        st.rerun()

    if st.session_state.bulk_delete_pending:
        st.markdown("---")
        st.subheader("Confirm Bulk Deletion")
        st.error(
            f"WARNING: You are about to permanently delete ALL documents from Pinecone index `{pinecone_index_name}` in namespace `'{namespace or '__default__'}'`."
        )
        st.write("This action cannot be undone.")

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
            if st.button("Cancel Bulk Deletion", key="cancel_bulk_delete_button"):
                st.session_state.bulk_delete_pending = False
                st.info("Bulk deletion cancelled.")
                logger.info("Bulk deletion cancelled by user.")
                st.rerun()

    if st.session_state.delete_pending:
        st.markdown("---")
        st.subheader("Confirm Document Deletion")
        st.warning(f"You are about to permanently delete records for:")
        if st.session_state.file_to_delete_staged:
            st.write(f"- **File Name:** `{st.session_state.file_to_delete_staged}`")
        if st.session_state.docid_to_delete_staged:
            st.write(f"- **Document ID:** `{st.session_state.docid_to_delete_staged}`")
        st.write(
            f"From Pinecone index `{pinecone_index_name}` in namespace `'{namespace or '__default__'}'`."
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
                        st.info(f"Index stats (after delete):")
                        st.json(stats_after.to_dict())
                        logger.info(
                            f"Retrieved index stats after deletion: {stats_after.to_dict()}"
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
            if st.button("Cancel Deletion", key="cancel_delete_button"):
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
    st.markdown(
        """
        This tool helps you build and manage a Retrieval-Augmented Generation (RAG) knowledge base
        by ingesting your documents into a Pinecone vector database.
        """
    )

    load_dotenv()  # Load environment variables from .env file
    current_logging_level_name = os.environ.get(
        "LOGGING_LEVEL", DEFAULT_SETTINGS["LOGGING_LEVEL"]
    )
    logger = setup_logging_once(level=logging.getLevelName(current_logging_level_name))
    logger.info("Environment variables loaded from .env (if present).")

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

    st.markdown("---")
    st.header("1. Configuration")
    st.write(
        "Set up your API keys, Pinecone index details, and document processing parameters."
    )

    # Retrieve current API key values for pre-filling inputs and external button access
    pinecone_api_key_val = os.environ.get(
        "PINECONE_API_KEY", DEFAULT_SETTINGS["PINECONE_API_KEY"]
    )
    embedding_api_key_val = os.environ.get(
        "EMBEDDING_API_KEY", DEFAULT_SETTINGS["EMBEDDING_API_KEY"]
    )

    with st.form(key=f"config_form_{st.session_state.config_form_key}"):
        # API Keys section within an expander for sensitive information
        with st.expander("🔑 API Keys (Sensitive)", expanded=True):
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

        embedding_model_name = st.text_input(
            "Embedding Model Name",
            value=os.environ.get(
                "EMBEDDING_MODEL_NAME", DEFAULT_SETTINGS["EMBEDDING_MODEL_NAME"]
            ),
            help="The embedding model identifier (e.g., 'text-embedding-3-small'). Choose based on your provider's available models.",
        )
        embedding_dimension = st.number_input(
            "Embedding Dimension",
            min_value=1,
            value=int(
                os.environ.get(
                    "EMBEDDING_DIMENSION", DEFAULT_SETTINGS["EMBEDDING_DIMENSION"]
                )
            ),
            help="Dimensionality of the embedding vectors. Typically 1536 for 'text-embedding-3-small'.",
        )
        metric_type = st.selectbox(
            "Pinecone Metric Type",
            options=["cosine", "euclidean", "dotproduct"],
            index=["cosine", "euclidean", "dotproduct"].index(
                os.environ.get("METRIC_TYPE", DEFAULT_SETTINGS["METRIC_TYPE"])
            ),
            help=(
                "Similarity metric used by Pinecone for vector search. "
                "'cosine' is common for semantic similarity. Choose based on your embedding model's training."
            ),
        )
        namespace = st.text_input(
            "Namespace (Optional)",
            value=os.environ.get("NAMESPACE", DEFAULT_SETTINGS["NAMESPACE"]),
            help=(
                "Optional partition within the Pinecone index to isolate data (e.g., per customer or project). "
                "Leave blank to use the default namespace."
            ),
        )

        st.subheader("Document Processing Settings")

        # Unstructured Loader Strategy selection
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
            "Unstructured Loader Strategy",
            options=unstructured_strategy_options,
            index=default_strategy_index,
            help=(
                "Select the strategy for Unstructured.io document parsing: "
                "'hi_res' (high-resolution, slower, more accurate), "
                "'fast' (quicker, less accurate), or "
                "'auto' (Unstructured decides based on document type)."
            ),
        )

        chunk_size = st.number_input(
            "Chunk Size (characters)",
            min_value=256,
            value=int(os.environ.get("CHUNK_SIZE", DEFAULT_SETTINGS["CHUNK_SIZE"])),
            help=(
                "Maximum size of each text chunk to embed, measured in characters. "
                "This will be dynamically adjusted if needed to fit Pinecone's metadata limit."
            ),
        )
        chunk_overlap = st.number_input(
            "Chunk Overlap (characters)",
            min_value=0,
            value=int(
                os.environ.get("CHUNK_OVERLAP", DEFAULT_SETTINGS["CHUNK_OVERLAP"])
            ),
            help=(
                "Number of characters overlapping between consecutive chunks to maintain context. "
                "Typical values are 100-200."
            ),
        )

        # Content Filtering Settings
        st.subheader("Filtering Settings")
        enable_filtering = st.checkbox(
            "Enable Content Filtering",
            value=(os.environ.get("ENABLE_FILTERING", "True").lower() == "true"),
            help=(
                "Disable to keep all extracted text, regardless of length or category. "
                "Use with caution as it may introduce noise and increase costs."
            ),
            key="config_enable_filtering_checkbox",
        )

        keep_low_confidence = st.checkbox(
            "Keep short low-confidence snippets?",
            value=(
                os.environ.get("KEEP_LOW_CONFIDENCE_SNIPPETS", "False").lower()
                == "true"
            ),
            help=(
                "Leave unchecked to drop very short, non-whitelisted text (recommended). "
                "Enable for verse-like corpora where short lines carry meaning."
            ),
            key="config_keep_low_confidence_checkbox",
        )

        whitelisted_keywords_input = st.text_input(
            "Whitelisted Keywords (comma-separated)",
            value=os.environ.get(
                "WHITELISTED_KEYWORDS", DEFAULT_SETTINGS["WHITELISTED_KEYWORDS"]
            ),
            help=(
                "Enter words or short phrases that should always be kept, even if short or generic. "
                "E.g., 'Error Code', 'API Key', 'Product ID'. Case-insensitive matching."
            ),
            key="config_whitelisted_keywords_input",
        )
        min_generic_content_length_ui = st.number_input(
            "Min Length for Generic Content (characters)",
            min_value=0,
            value=int(
                os.environ.get(
                    "MIN_GENERIC_CONTENT_LENGTH",
                    DEFAULT_SETTINGS["MIN_GENERIC_CONTENT_LENGTH"],
                )
            ),
            help=(
                "Minimum character length for generic text segments to be included. "
                "Text shorter than this, and not categorized as important or whitelisted, will be filtered out. "
                "A lower value may capture more short facts but could introduce noise."
            ),
            key="config_min_generic_content_length_input",
        )
        enable_ner_filtering = st.checkbox(
            "Enable NER Filtering for Short Generic Content",
            value=st.session_state.get(
                "enable_ner_filtering",
                DEFAULT_SETTINGS["ENABLE_NER_FILTERING"].lower() == "true",
            ),
            help=(
                "If enabled, short generic text (below min length) will be kept if it contains recognized Named Entities (e.g., dates, organizations, persons). "
                "Requires SpaCy 'en_core_web_sm' model. NER filtering is automatically disabled if SpaCy model fails to load."
            ),
            disabled=(nlp is None),
            key="config_enable_ner_filtering_checkbox",
        )
        if enable_ner_filtering and nlp is None:
            st.warning(
                "SpaCy 'en_core_web_sm' model is required for NER filtering but failed to load. Please install it using `python -m spacy download en_core_web_sm` and restart the app."
            )

        document_metadata_file = st.file_uploader(
            "Upload Document-Specific Metadata (CSV/JSON, optional)",
            type=["csv", "json"],
            accept_multiple_files=False,
            help=(
                "Upload a CSV or JSON file containing metadata for individual documents. "
                "For CSV, include a 'file_name' column. For JSON, use an array of objects with a 'file_name' key. "
                "This metadata will override global settings for matching documents. "
                f"Note: Reserved keys ({', '.join(RESERVED_METADATA_KEYS)}) will be ignored."
            ),
        )

        overwrite_existing_docs = st.checkbox(
            "Overwrite existing documents with the same file name?",
            value=(
                os.environ.get("OVERWRITE_EXISTING_DOCS", "False").lower() == "true"
            ),
            help=(
                "If checked, any existing vectors in Pinecone associated with uploaded files (matched by file name) "
                "will be deleted before new chunks are uploaded. Use with caution."
            ),
            key="config_overwrite_checkbox",
        )

        with st.expander("⚙️ Advanced Chunking Settings", expanded=False):
            st.markdown("---")
            st.subheader("Semantic Chunking Parameters")
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
                "Semantic Split Threshold Type",
                options=semantic_chunker_threshold_type_options,
                index=default_semantic_threshold_type_index,
                help=(
                    "Determines how semantic breakpoints are identified. "
                    "'percentile': Splits at distances greater than a certain percentile of all distances (default 98). "
                    "'standard_deviation': Splits at distances greater than X standard deviations from the mean (default 3). "
                    "'interquartile': Uses interquartile range for splitting (default 1.5). "
                    "'gradient': Applies anomaly detection on gradient array (default 98)."
                ),
                key="config_semantic_chunker_threshold_type",
            )

            # Adjust min/max/value/step based on the selected type for better UX
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
                "Semantic Split Threshold Amount",
                min_value=threshold_amount_min,
                max_value=threshold_amount_max,
                value=float(
                    os.environ.get(
                        "SEMANTIC_CHUNKER_THRESHOLD_AMOUNT", str(threshold_amount_value)
                    )
                ),
                step=threshold_amount_step,
                format="%.1f",
                help=(
                    f"The specific value for the selected threshold type. "
                    f"For '{semantic_chunker_threshold_type}', a higher value means fewer, larger chunks (more strict splitting)."
                ),
                key="config_semantic_chunker_threshold_amount",
            )

            min_child_chunk_length_ui = st.number_input(
                "Minimum Semantic Child Chunk Length (characters)",
                min_value=1,
                value=int(
                    os.environ.get("MIN_CHILD_CHUNK_LENGTH", "100")
                ),  # Default to 100 chars
                step=10,
                help=(
                    "The absolute minimum character length for any semantic child chunk. "
                    "Chunks shorter than this will be skipped. This helps ensure embeddable chunks are meaningful."
                ),
                key="config_min_child_chunk_length",
            )

        # Advanced Logging Settings
        with st.expander("Advanced Logging Settings", expanded=False):
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
            )

            if logging.getLevelName(logger.level) != logging_level_selected:
                logger.setLevel(logging_level_selected)
                logger.info(
                    f"Logging level dynamically set to {logging_level_selected}."
                )

        # Form submission buttons
        col1, col2 = st.columns(2)
        with col1:
            save_conf = st.form_submit_button("💾 Save Configuration (local .env)")
        with col2:
            reset_conf = st.form_submit_button("🔄 Reset to Defaults (local only)")

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
    if save_conf:
        custom_metadata_json_string = json.dumps(
            st.session_state.dynamic_metadata_fields
        )
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

        with open(".env", "w") as f:
            for k, v in to_save.items():
                f.write(f"{k}={v}\n")
        for k, v in to_save.items():
            os.environ[k] = v
        st.success("Configuration saved locally to .env")
        logger.info("Configuration saved locally to .env file.")
        st.rerun()

    # Logic for resetting configuration to defaults
    if reset_conf and is_running_locally():
        st.session_state.show_reset_dialog = True
        st.rerun()

    if st.session_state.show_reset_dialog:
        st.warning("Reset ALL configuration to defaults? (affects only local .env)")
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

            with open(".env", "w") as f:
                for k, v in defaults.items():
                    f.write(f"{k}={v}\n")
            for k, v in defaults.items():
                os.environ[k] = v
            st.session_state.show_reset_dialog = False
            st.session_state.config_form_key += 1
            st.success("Local configuration reset")
            logger.info("Local configuration reset to defaults.")
            st.rerun()
        if st.button("Cancel"):
            st.session_state.show_reset_dialog = False
            st.info("Reset cancelled")
            logger.info("Configuration reset cancelled.")
            st.rerun()

    st.markdown("---")
    st.subheader("Global Custom Metadata")
    st.write(
        "Define custom key-value pairs here. These will be added to all uploaded document chunks."
    )
    st.info(
        f"Note: Reserved keys ({', '.join(RESERVED_METADATA_KEYS)}) will be ignored or overwritten by internal values."
    )
    st.markdown(
        'Example: `{"project_name": "MyRAGProject", "department": "Engineering"}`'
    )

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
    st.markdown("---")

    st.header("2. Upload Documents")
    st.write(
        "Upload your documents here to embed them and upsert into your Pinecone knowledge base."
    )

    uploaded_files = st.file_uploader(
        "Select documents to upload",
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
        help="Supported file types include common document, spreadsheet, presentation, and image formats.",
    )

    # Display uploaded file names for better user experience
    if uploaded_files:
        st.subheader("Uploaded Files:")
        for i, file in enumerate(uploaded_files):
            st.markdown(f"- {file.name}")
    else:
        st.info("No files selected yet.")

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

    if st.button("🚀 Process, Embed & Upsert to Pinecone", disabled=not can_process):
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
            try:
                file_content = document_metadata_file.getvalue().decode("utf-8")
                if document_metadata_file.type == "text/csv":
                    df = pd.read_csv(io.StringIO(file_content))
                    if "file_name" not in df.columns:
                        raise ValueError(
                            "CSV metadata file must contain a 'file_name' column."
                        )
                    for _, row in df.iterrows():
                        file_name_val = str(
                            row["file_name"]
                        ).lower()  # Normalize file_name from CSV
                        doc_md = {}
                        for k, v in row.drop("file_name").items():
                            if k not in RESERVED_METADATA_KEYS:
                                try:
                                    parsed_v = json.loads(str(v))
                                    doc_md[k] = parsed_v
                                except (json.JSONDecodeError, TypeError):
                                    doc_md[k] = str(v)
                            else:
                                logger.warning(
                                    f"Document-specific CSV metadata for '{file_name_val}': Key '{k}' is reserved and will be ignored."
                                )
                        document_specific_metadata_map[file_name_val] = doc_md
                    logger.info(
                        f"Loaded {len(document_specific_metadata_map)} document-specific metadata entries from CSV."
                    )
                elif document_metadata_file.type == "application/json":
                    json_data = json.loads(file_content)
                    if not isinstance(json_data, list):
                        raise ValueError(
                            "JSON metadata file must be an array of objects."
                        )
                    for entry in json_data:
                        if "file_name" not in entry:
                            raise ValueError(
                                "Each object in JSON metadata array must contain a 'file_name' key."
                            )
                        file_name_val = str(
                            entry["file_name"]
                        ).lower()  # Normalize file_name from JSON
                        cleaned_entry = {}
                        for k, v in entry.items():
                            if k == "file_name":
                                continue
                            if k not in RESERVED_METADATA_KEYS:
                                cleaned_entry[k] = v
                            else:
                                logger.warning(
                                    f"Document-specific JSON metadata for '{file_name_val}': Key '{k}' is reserved and will be ignored."
                                )
                        document_specific_metadata_map[file_name_val] = cleaned_entry
                    logger.info(
                        f"Loaded {len(document_specific_metadata_map)} document-specific metadata entries from JSON."
                    )
                st.success(
                    f"Successfully loaded document-specific metadata from '{document_metadata_file.name}'."
                )
            except Exception as e:
                logger.exception(
                    f"Error processing document-specific metadata file '{document_metadata_file.name}'."
                )
                st.error(
                    f"Error processing document-specific metadata file: {e}. Please check its format."
                )
                return

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

            if pc and pc.has_index(pinecone_index_name):
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

            with plan_summary_placeholder.container():
                for msg in plan_summary_messages:
                    st.markdown(f"- {msg}")
            time.sleep(0.05)

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
        if pc and not pc.has_index(pinecone_index_name):
            conf = SUPPORTED_PINECONE_REGIONS.get(
                pinecone_cloud_region, SUPPORTED_PINECONE_REGIONS["aws-us-east-1"]
            )
            logger.info(
                f"Pinecone index '{pinecone_index_name}' does not exist. Creating new index."
            )
            st.info(f"Creating Pinecone index '{pinecone_index_name}'...")
            with st.spinner("Waiting for index to become ready..."):
                pc.create_index(
                    name=pinecone_index_name,
                    dimension=int(embedding_dimension),
                    metric=metric_type,
                    spec=ServerlessSpec(cloud=conf["cloud"], region=conf["region"]),
                )
                while not pc.describe_index(pinecone_index_name).status["ready"]:
                    time.sleep(1)
            st.success(f"Pinecone index '{pinecone_index_name}' is ready.")
            logger.info(f"Pinecone index '{pinecone_index_name}' created and is ready.")

        index = (
            pc.Index(pinecone_index_name)
            if pc and pc.has_index(pinecone_index_name)
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
                    loader = UnstructuredLoader(
                        file_path, strategy=unstructured_strategy
                    )
                    raw_elements = list(loader.lazy_load())
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
                        label=f"Processing document: **{original_file_name}** - Generating semantic child chunks...",
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
                        label=f"Processing document: **{original_file_name}** - Summarizing special content (tables, figures, images)...",
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

                        logger.info(
                            f"Chunk token stats for '{original_file_name}': "
                            f"min={min_tokens}, max={max_tokens}, avg={avg_tokens:.1f}, "
                            f"count={len(chunk_token_lengths)}"
                        )

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

                    file_texts_to_embed = [
                        c.metadata.pop("_embedding_text", c.page_content)
                        for c in all_embeddable_child_chunks
                    ]
                    file_vectors = []
                    current_batch_texts = []
                    current_batch_tokens = 0

                    for i, text in enumerate(file_texts_to_embed):
                        text_tokens = count_tokens(text, embedding_model_name, logger)

                        if text_tokens > OPENAI_MAX_TOKENS_PER_EMBEDDING_REQUEST:
                            error_msg = (
                                f"Chunk {i+1}/{len(file_texts_to_embed)} of '{original_file_name}' "
                                f"contains {text_tokens} tokens, which exceeds the embedding model limit "
                                f"({OPENAI_MAX_TOKENS_PER_EMBEDDING_REQUEST}). "
                                "Please reduce the chunk size or adjust filtering settings."
                            )
                            logger.error(error_msg)
                            st.error(error_msg)
                            raise ValueError(error_msg)

                        if (
                            current_batch_tokens + text_tokens
                            > OPENAI_MAX_TOKENS_PER_EMBEDDING_REQUEST
                        ) or (
                            len(current_batch_texts)
                            >= OPENAI_MAX_TEXTS_PER_EMBEDDING_REQUEST
                        ):

                            if current_batch_texts:
                                logger.debug(
                                    f"Embedding batch of {len(current_batch_texts)} texts ({current_batch_tokens} tokens) for '{original_file_name}'."
                                )
                                batch_vectors = embed_model.embed_documents(
                                    current_batch_texts
                                )
                                file_vectors.extend(batch_vectors)

                            current_batch_texts = [text]
                            current_batch_tokens = text_tokens
                        else:
                            current_batch_texts.append(text)
                            current_batch_tokens += text_tokens

                        status_container.write(
                            f"Generating embeddings for '{original_file_name}': {i+1}/{len(file_texts_to_embed)} chunks."
                        )

                    if current_batch_texts:
                        logger.debug(
                            f"Embedding final batch of {len(current_batch_texts)} texts ({current_batch_tokens} tokens) for '{original_file_name}'."
                        )
                        batch_vectors = embed_model.embed_documents(current_batch_texts)
                        file_vectors.extend(batch_vectors)

                    status_container.write(
                        f"Generated {len(file_vectors)} embeddings for '{original_file_name}'."
                    )
                    logger.info(
                        f"Generated {len(file_vectors)} embeddings for '{original_file_name}'."
                    )

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

                        # Metadata should already be canonicalized and shrunk in _generate_child_chunks or _process_special_elements
                        # No need for a second _shrink_metadata_to_limit call here.
                        final_metadata_for_upsert = c.metadata

                        # The "truncated AGAIN" warning should no longer appear if the metadata was properly handled upstream.
                        # If it still appears, it indicates a deeper issue in the _shrink_metadata_to_limit function itself
                        # or an extremely large chunk that cannot fit even minimal metadata.

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

        st.success(
            "All selected documents have been processed (or skipped as per plan)."
        )
        logger.info("All selected documents processing complete.")

        try:
            stats = index.describe_index_stats()
            st.info(f"Final index stats: {stats.to_dict()}")
            logger.info(f"Final index stats: {stats.to_dict()}")
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
    st.header("Application Logs")
    with st.expander("View Logs", expanded=False):
        log_display_area = st.empty()
        log_display_area.code(
            "\n".join(reversed(st.session_state.streamlit_handler.get_records()))
        )
        if st.button("Clear Logs", key="clear_logs_button"):
            st.session_state.streamlit_handler.log_records.clear()
            log_display_area.code("")
            logger.info("Application logs cleared by user.")


if __name__ == "__main__":
    main()
