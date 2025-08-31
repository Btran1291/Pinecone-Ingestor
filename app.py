import streamlit as st
import os
import tempfile
import shutil
import json
import hashlib
import uuid
import time
import logging
from collections import deque
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
import io
import tiktoken
import spacy
import numpy as np
import nltk
import re

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
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
        self.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

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
    if 'streamlit_handler' not in st.session_state:
        st.session_state.streamlit_handler = StreamlitLogHandler()

    if 'app_logger' not in st.session_state:
        st.session_state.app_logger = logging.getLogger(__name__)
        st.session_state.app_logger.setLevel(level)

        if not any(isinstance(h, StreamlitLogHandler) for h in st.session_state.app_logger.handlers):
            st.session_state.app_logger.addHandler(st.session_state.streamlit_handler)
        if not any(isinstance(h, logging.StreamHandler) for h in st.session_state.app_logger.handlers):
            st.session_state.app_logger.addHandler(logging.StreamHandler())

        st.session_state.app_logger.info(f"Logger initialized with level {logging.getLevelName(level)}.")

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
    "CHUNK_SIZE": "1000",
    "CHUNK_OVERLAP": "200",
    "CUSTOM_METADATA": '[{"key": "", "value": ""}]',
    "OVERWRITE_EXISTING_DOCS": "False",
    "LOGGING_LEVEL": "INFO",
    "ENABLE_FILTERING": "True",
    "WHITELISTED_KEYWORDS": "",
    "MIN_GENERIC_CONTENT_LENGTH": "120",
    "ENABLE_NER_FILTERING": "True",
    "UNSTRUCTURED_STRATEGY": "hi_res",
    "HEADING_DETECTION_CONFIDENCE_THRESHOLD": "0.65",
    "MIN_CHUNKS_PER_SECTION_FOR_MERGE": "2",
    "SENTENCE_SPLIT_THRESHOLD_CHARS": "300",
    "MIN_CHUNK_LENGTH": "100",
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
MIN_RE_SPLIT_CHUNK_SIZE = 100
MAX_UTF8_BYTES_PER_CHAR = 4

# OpenAI embedding API limits
OPENAI_MAX_TOKENS_PER_EMBEDDING_REQUEST = 300000
OPENAI_MAX_TEXTS_PER_EMBEDDING_REQUEST = 1000

# Categories from Unstructured.io that are considered important and typically kept during filtering
STRUCTURAL_CUES_AND_CRITICAL_CONCISE_CONTENT = {
    "Title", "NarrativeText", "ListItem", "Table", "FigureCaption", "Formula", "Header", "Subheader"
}

# Metadata keys reserved for internal use or special handling within Pinecone
RESERVED_METADATA_KEYS = {
    "document_id", "file_name", "text",
    "page_number", "page", "start_index", "category",
    "original_file_path",
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
        logger.error(f"Failed to load SpaCy model '{model_name}': {e}. NER filtering will be disabled.")
        st.error(f"Failed to load SpaCy model '{model_name}'. NER filtering will be disabled. Please ensure it's installed (`python -m spacy download {model_name}`).")
        return None

# -------- Helper Functions --------
def save_uploaded_file_to_temp(uploaded_file):
    """
    Saves an uploaded Streamlit file to a temporary directory on the server.
    Returns the file path, the temporary directory path, and the file bytes.
    """
    logger = st.session_state.app_logger
    logger.debug(f"Attempting to save uploaded file: {uploaded_file.name}")
    try:
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, uploaded_file.name)
        file_bytes = uploaded_file.getvalue()
        with open(file_path, "wb") as f:
            f.write(file_bytes)
        logger.info(f"Successfully saved {uploaded_file.name} to temporary path: {file_path}")
        return file_path, temp_dir, file_bytes
    except (IOError, OSError) as e:
        logger.exception(f"Error saving uploaded file {uploaded_file.name}")
        st.error(f"Error saving uploaded file: {e}")
        return None, None, None
    except Exception as e:
        logger.exception(f"An unexpected error occurred while saving uploaded file {uploaded_file.name}")
        st.error(f"An unexpected error occurred: {e}")
        return None, None, None

def is_running_locally():
    """
    Checks if the Streamlit application is running in a local environment
    by looking for common cloud environment variables.
    """
    logger = st.session_state.app_logger
    cloud_env_vars = [
        "RENDER_EXTERNAL_HOSTNAME", "STREAMLIT_CLOUD_APP_NAME",
        "AWS_EXECUTION_ENV", "GCP_PROJECT", "AZURE_FUNCTIONS_ENVIRONMENT",
        "KUBERNETES_SERVICE_HOST",
    ]
    is_local = not any(os.environ.get(var) for var in cloud_env_vars)
    logger.debug(f"Running locally check: {is_local}")
    return is_local

def deterministic_document_id(file_name: str, file_bytes: bytes) -> str:
    """
    Generates a deterministic document ID based on the file name and content hash.
    Ensures uniqueness and handles Pinecone's ID length limit.
    """
    logger = st.session_state.app_logger
    doc_id = f"{file_name}_{hashlib.sha256(file_bytes).hexdigest()}"
    if len(doc_id) > 512:
        doc_id = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()
    logger.debug(f"Generated document ID for {file_name}: {doc_id}")
    return doc_id

def deterministic_chunk_id(document_id: str, chunk_text: str, page: str, start_index: str) -> str:
    """
    Generates a deterministic chunk ID based on the document ID, chunk text hash,
    page number, and start index.
    """
    logger = st.session_state.app_logger
    text_hash = hashlib.sha256((chunk_text or "").encode("utf-8")).hexdigest()
    pos = f"{page}_{start_index}"
    chunk_id = hashlib.sha256(f"{document_id}_{text_hash}_{pos}".encode("utf-8")).hexdigest()
    logger.debug(f"Generated chunk ID for doc {document_id}, page {page}, start {start_index}: {chunk_id}")
    return chunk_id

def count_tokens(text: str, model_name: str, logger: logging.Logger) -> int:
    """
    Counts tokens in a text string using tiktoken for OpenAI models.
    Falls back to a character-based estimation if the model encoding is not found.
    """
    try:
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(text))
    except KeyError:
        logger.warning(f"Could not find tiktoken encoding for model '{model_name}'. Falling back to conservative character-based estimation.")
        return len(text) // 3
    except Exception as e:
        logger.error(f"Error counting tokens for model '{model_name}': {e}")
        return len(text) // 3

def get_estimated_base_metadata_overhead(
    parsed_global_custom_metadata: dict,
    document_specific_metadata_map: dict,
    file_names_in_batch: list[str],
    logger: logging.Logger
) -> int:
    """
    Estimates the maximum possible byte size of metadata (excluding the 'text' field)
    to calculate the safe maximum text size for a chunk, adhering to Pinecone's metadata limits.
    Considers reserved keys, global custom metadata, and document-specific metadata.
    """
    dummy_metadata = {}

    # Estimate overhead from reserved internal keys using max possible lengths
    dummy_metadata["document_id"] = "a" * 64
    longest_file_name_placeholder = "placeholder_file_name_for_estimation_very_long_indeed_123456789012345678901234567890.pdf"
    longest_file_name = max(file_names_in_batch, key=len) if file_names_in_batch else longest_file_name_placeholder
    dummy_metadata["file_name"] = longest_file_name
    dummy_metadata["page_number"] = 9999
    dummy_metadata["start_index"] = 99999999
    dummy_metadata["category"] = "LongestCategoryNamePossible"
    dummy_metadata["original_file_path"] = "/path/to/a/very/long/original/file/name/on/user/machine/that/might/be/stored/long_path_long_name.pdf"

    # Include global custom metadata in the overhead estimation
    for k, v in parsed_global_custom_metadata.items():
        if k not in RESERVED_METADATA_KEYS:
            if isinstance(v, (str, int, float, bool, list, dict)):
                dummy_metadata[k] = v
            else:
                dummy_metadata[k] = str(v)

    # Include document-specific metadata in the overhead estimation (find the largest one)
    max_doc_specific_md_size = 0
    best_doc_specific_md_for_estimation = {}
    for file_name, doc_md in document_specific_metadata_map.items():
        temp_md = {}
        for k, v in doc_md.items():
            if k not in RESERVED_METADATA_KEYS:
                if isinstance(v, (str, int, float, bool, list, dict)):
                    temp_md[k] = v
                else:
                    temp_md[k] = str(v)
        current_size = len(json.dumps(temp_md).encode("utf-8"))
        if current_size > max_doc_specific_md_size:
            max_doc_specific_md_size = current_size
            best_doc_specific_md_for_estimation = temp_md

    for k, v in best_doc_specific_md_for_estimation.items():
        if k not in RESERVED_METADATA_KEYS:
            dummy_metadata[k] = v

    overhead_bytes = len(json.dumps(dummy_metadata).encode("utf-8"))
    logger.debug(f"Estimated base metadata overhead: {overhead_bytes} bytes with dummy: {dummy_metadata}")
    return overhead_bytes

# Recommended allowed keys and priority list (highest -> lowest)
RAG_ALLOWED_KEYS = [
    "text", 
    "chunk_id",
    "document_id",
    "file_name",
    "section_id",
    "section_heading",
    "heading_level",
    "chunk_confidence",
    "chunk_index_in_section",
    "total_chunks_in_section",
    "page_number",
    "languages",
    "filetype",
    "last_modified",
]

# Keys to always remove (sensitive or irrelevant)
RAG_DISCARD_KEYS = {
    "file_directory", "original_file_path", "source", "filename",
    "element_id", "parent_id", "coordinates", "start_index", "page", "file_path"
}


def _canonicalize_and_filter_metadata(
    raw_meta: dict,
    global_custom_md: dict,
    doc_specific_md: dict,
    logger: logging.Logger
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
            logger.warning(f"Could not convert page_number '{meta['page_number']}' to int. Storing as string.")
    # If 'page_number' is not found, check for 'page'
    elif "page" in meta and meta["page"] is not None:
        try:
            # Attempt to convert 'page' to integer and store as 'page_number'
            cleaned["page_number"] = int(meta["page"])
        except (ValueError, TypeError):
            # If conversion fails, store as string and log a warning
            cleaned["page_number"] = str(meta["page"])
            logger.warning(f"Could not convert page '{meta['page']}' to int. Storing as string.")
    
    meta.pop("page", None)
    meta.pop("page_number", None)

    # Canonicalize: prefer 'file_name', but accept 'filename' as fallback
    if "file_name" not in meta and "filename" in meta:
        meta["file_name"] = meta.pop("filename")

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
                cleaned[k] = json.loads(json_repr) if (isinstance(v, list) and all(isinstance(i, str) for i in v)) else json_repr
            except Exception:
                cleaned[k] = str(v)
                logger.debug(f"Converted non-serializable metadata key '{k}' to string.")
        else:
            # Fallback to string
            cleaned[k] = str(v)
            logger.debug(f"Converted metadata key '{k}' of type {type(v)} to string.")

    # 2) Merge global and document-specific custom metadata (document-specific overrides)
    for k, v in (global_custom_md or {}).items():
        if not k or k in RESERVED_METADATA_KEYS:  # keep reserved protection
            logger.debug(f"Ignoring global custom metadata key '{k}' (reserved or empty).")
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
            logger.debug(f"Ignoring doc-specific metadata key '{k}' (reserved or empty).")
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


def _shrink_metadata_to_limit(metadata: dict, logger: logging.Logger, pinecone_metadata_max_bytes: int = PINECONE_METADATA_MAX_BYTES) -> dict:
    """
    Ensure metadata JSON byte size <= pinecone_metadata_max_bytes.
    Strategy:
      1) Keep highest priority keys in RAG_ALLOWED_KEYS. Remove keys not in allowed list while respecting allowed custom keys
         (we allow additional keys but prefer RAG_ALLOWED_KEYS first).
      2) If still too big, remove lower-priority allowed keys in reverse priority order.
      3) If still too big, truncate the 'text' field (as last resort) and mark text_truncated=True.
    Returns the pruned metadata (may be modified).
    """
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

    logger.warning(f"Metadata size {size} bytes exceeds limit {pinecone_metadata_max_bytes} bytes. Beginning shrink process.")

    # Build prioritized keys to keep: start with RAG_ALLOWED_KEYS in order, then any remaining small custom keys
    prioritized = [k for k in RAG_ALLOWED_KEYS if k in metadata]
    # Add other keys by ascending length of their JSON repr (shorter first)
    other_keys = [k for k in metadata.keys() if k not in prioritized]
    other_keys_sorted = sorted(other_keys, key=lambda k: len(json.dumps({k: metadata[k]})))
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
            pruned.pop(key, None)

    # If still too big, attempt to remove other keys not in RAG_ALLOWED_KEYS
    if len(json.dumps(pruned).encode("utf-8")) > pinecone_metadata_max_bytes:
        for key in list(pruned.keys()):
            if key not in RAG_ALLOWED_KEYS and key != "text":
                logger.debug(f"Removing additional non-priority metadata key '{key}'.")
                pruned.pop(key, None)
                if len(json.dumps(pruned).encode("utf-8")) <= pinecone_metadata_max_bytes:
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
            allowed_for_text = max(0, pinecone_metadata_max_bytes - overhead - SAFETY_BUFFER_BYTES)
            if allowed_for_text <= 0:
                # remove text entirely (should be last resort)
                logger.critical("No room left for 'text' in metadata; removing it as last resort. You should use external text storage.")
                pruned.pop("text", None)
                pruned["text_truncated"] = True
            else:
                # Determine character count conservatively (UTF-8 worst-case 4 bytes per char)
                max_chars = allowed_for_text // MAX_UTF8_BYTES_PER_CHAR
                truncated_text = text[:max_chars]
                pruned["text"] = truncated_text
                pruned["text_truncated"] = True
                logger.warning(f"Truncated 'text' to {len(truncated_text)} chars to fit metadata limit ({pinecone_metadata_max_bytes} bytes).")
        else:
            logger.critical("Metadata too large and no 'text' field to shrink. Metadata will be pruned aggressively.")
            # prune additional keys until we fit
            for key in list(pruned.keys()):
                if key != "chunk_id":
                    pruned.pop(key, None)
                if len(json.dumps(pruned).encode("utf-8")) <= pinecone_metadata_max_bytes:
                    break

    final_size = len(json.dumps(pruned).encode("utf-8"))
    if final_size > pinecone_metadata_max_bytes:
        logger.error(f"Unable to reduce metadata to {pinecone_metadata_max_bytes} bytes; final size {final_size} bytes. Upsert may fail.")
    else:
        logger.debug(f"Shrunk metadata to {final_size} bytes (limit {pinecone_metadata_max_bytes} bytes).")

    return pruned

def _merge_chunk_metadata(meta1: dict, meta2: dict, logger: logging.Logger) -> dict:
    """
    Intelligently merges metadata from two chunks into a single dictionary.
    Prioritizes meta1 for core identifiers, combines lists, and handles ranges for page numbers.
    Assumes meta1 and meta2 are already canonicalized.
    """
    merged_meta = meta1.copy()

    for key in ["document_id", "file_name", "section_id", "section_heading", "heading_level"]:
        if key in meta2 and meta1.get(key) != meta2.get(key):
            logger.warning(f"Metadata merge conflict for key '{key}': '{meta1.get(key)}' vs '{meta2.get(key)}'. Prioritizing '{meta1.get(key)}'.")
        # merged_meta[key] already has meta1's value, no change needed unless meta1 is missing

    page_values = []
    
    for m in [meta1, meta2]:
        if "page_number" in m and m["page_number"] is not None:
            if isinstance(m["page_number"], (int, str)):
                # If it's a single int or string, add it
                page_values.append(m["page_number"])
            elif isinstance(m["page_number"], list):
                # If it's already a list (e.g., from a previous merge), extend
                page_values.extend(m["page_number"])
            # Handle string ranges like "1-3"
            elif isinstance(m["page_number"], str) and '-' in m["page_number"]:
                try:
                    start, end = map(int, m["page_number"].split('-'))
                    page_values.extend(range(start, end + 1))
                except ValueError:
                    logger.warning(f"Invalid page range string '{m['page_number']}' during merge. Skipping.")
            
    # Process collected page values
    if page_values:
        all_page_numbers = set()
        for p_val in page_values:
            try:
                all_page_numbers.add(int(p_val))
            except (ValueError, TypeError):
                logger.warning(f"Could not convert page value '{p_val}' to int during merge. Skipping.")
        
        unique_pages_sorted = sorted(list(all_page_numbers))

        if unique_pages_sorted:
            if len(unique_pages_sorted) == 1:
                merged_meta["page_number"] = unique_pages_sorted[0]
            else:
                # Represent as a range string (e.g., "1-5")
                # This is a common and useful way to represent page ranges for merged chunks.
                merged_meta["page_number"] = f"{unique_pages_sorted[0]}-{unique_pages_sorted[-1]}"
        else:
            merged_meta.pop("page_number", None) # Remove if no valid pages found after processing
    else:
        merged_meta.pop("page_number", None) # Remove if no page info was found at all
    
    # Start Index (take the minimum)
    if "start_index" in meta1 and "start_index" in meta2:
        merged_meta["start_index"] = min(meta1["start_index"], meta2["start_index"])
    elif "start_index" in meta2:
        merged_meta["start_index"] = meta2["start_index"]

    # Categories (combine unique categories)
    categories = set()
    if "category" in meta1:
        if isinstance(meta1["category"], list):
            categories.update(meta1["category"])
        else:
            categories.add(meta1["category"])
    if "category" in meta2:
        if isinstance(meta2["category"], list):
            categories.update(meta2["category"])
        else:
            categories.add(meta2["category"])
    if categories:
        merged_meta["category"] = sorted(list(categories))
    else:
        merged_meta.pop("category", None)

    # Languages (combine unique languages)
    languages = set()
    if "languages" in meta1:
        if isinstance(meta1["languages"], list):
            languages.update(meta1["languages"])
        else:
            languages.add(meta1["languages"])
    if "languages" in meta2:
        if isinstance(meta2["languages"], list):
            languages.update(meta2["languages"])
        else:
            languages.add(meta2["languages"])
    if languages:
        merged_meta["languages"] = sorted(list(languages))
    else:
        merged_meta.pop("languages", None)

    # Other metadata keys: prioritize meta1, but add from meta2 if not present in meta1
    for k, v in meta2.items():
        if k not in merged_meta: # Only add if not already in meta1
            merged_meta[k] = v
        elif k not in ["document_id", "file_name", "section_id", "section_heading", "heading_level",
                       "page_number", "page", "start_index", "category", "languages", "text", "chunk_id"]:
            # For other non-critical, non-reserved keys, if values differ, prioritize meta1 but log
            if merged_meta[k] != v:
                logger.debug(f"Metadata key '{k}' differs during merge: '{merged_meta[k]}' vs '{v}'. Keeping '{merged_meta[k]}'.")

    # Ensure final merged metadata is canonicalized and filtered (e.g., removes RAG_DISCARD_KEYS again)
    # This also handles custom metadata merging from global/doc-specific sources if not done earlier
    final_cleaned_merged_meta = _canonicalize_and_filter_metadata(
        merged_meta,
        logger
    )
    
    return final_cleaned_merged_meta


# -------- Adaptive Document Structure Analysis --------
def _analyze_document_structure(meaningful_chunks: list[Document], logger: logging.Logger) -> dict:
    """
    Analyzes the meaningful_chunks to derive statistical and pattern-based insights
    about the document's inherent structure. This forms Layer 1 of the intelligent chunking.
    """
    doc_context = {}
    
    if not meaningful_chunks:
        logger.warning("No meaningful chunks provided for structural analysis. Returning empty context.")
        return doc_context

    # --- 1. Statistical Text Analysis ---
    element_lengths = [len(d.page_content) for d in meaningful_chunks if d.page_content]
    if element_lengths:
        doc_context["mean_element_length"] = np.mean(element_lengths)
        doc_context["median_element_length"] = np.median(element_lengths)
        doc_context["std_dev_element_length"] = np.std(element_lengths)
        
        # Optimal derivation for thresholds: using quartiles and IQR for robustness
        q1 = np.percentile(element_lengths, 25)
        q3 = np.percentile(element_lengths, 75)
        iqr = q3 - q1
        
        # Elements significantly shorter than Q1 are candidates for headings
        # Factor of 1.5 is a common heuristic for outlier detection (e.g., in box plots)
        doc_context["short_element_threshold"] = max(MIN_RE_SPLIT_CHUNK_SIZE, q1 - 1.5 * iqr) 
        # Elements significantly longer than Q3 are candidates for large paragraphs/sections
        doc_context["long_element_threshold"] = q3 + 1.5 * iqr 
    else:
        doc_context["mean_element_length"] = 0
        doc_context["median_element_length"] = 0
        doc_context["std_dev_element_length"] = 0
        doc_context["short_element_threshold"] = MIN_RE_SPLIT_CHUNK_SIZE # Fallback
        doc_context["long_element_threshold"] = 0

    all_lines = []
    for d in meaningful_chunks:
        all_lines.extend(d.page_content.splitlines())

    # Line-level feature analysis
    line_features = {
        "all_caps": [], "title_case": [], "ends_with_punctuation": [],
        "starts_with_numbering": [], "leading_indent": [], "short_line": []
    }
    short_line_char_threshold = 80 # A fixed threshold for line-level analysis

    for line in all_lines:
        stripped_line = line.strip()
        if not stripped_line: continue # Skip empty lines for feature analysis

        # Avoid noise from very short lines when checking capitalization
        if len(stripped_line) > 5: 
            line_features["all_caps"].append(stripped_line.isupper())
            line_features["title_case"].append(stripped_line.istitle())
        
        line_features["ends_with_punctuation"].append(stripped_line and stripped_line[-1] in ".!?")
        line_features["starts_with_numbering"].append(bool(re.match(r'^(\d+\.?|\d+\.\d+\.?|[A-Z]\.?|[IVX]+\.)\s', stripped_line)))
        line_features["leading_indent"].append(len(line) - len(line.lstrip()) > 2) # More than 2 leading spaces
        line_features["short_line"].append(len(stripped_line) < short_line_char_threshold)

    total_analyzed_lines = len(all_lines)
    if total_analyzed_lines > 0:
        doc_context["prevalence_all_caps_lines"] = np.mean(line_features["all_caps"]) if line_features["all_caps"] else 0
        doc_context["prevalence_title_case_lines"] = np.mean(line_features["title_case"]) if line_features["title_case"] else 0
        doc_context["prevalence_ends_with_punctuation"] = np.mean(line_features["ends_with_punctuation"])
        doc_context["prevalence_starts_with_numbering"] = np.mean(line_features["starts_with_numbering"])
        doc_context["prevalence_leading_indent"] = np.mean(line_features["leading_indent"])
        doc_context["prevalence_short_lines"] = np.mean(line_features["short_line"])
    else:
        doc_context["prevalence_all_caps_lines"] = 0
        doc_context["prevalence_title_case_lines"] = 0
        doc_context["prevalence_ends_with_punctuation"] = 0
        doc_context["prevalence_starts_with_numbering"] = 0
        doc_context["prevalence_leading_indent"] = 0
        doc_context["prevalence_short_lines"] = 0

    # Blank Line Analysis (simple heuristic for now, can be refined with raw file bytes if needed)
    blank_line_sequences = 0
    consecutive_blank_lines = 0
    for line in all_lines:
        if not line.strip():
            consecutive_blank_lines += 1
        else:
            if consecutive_blank_lines >= 2: # Detect sequences of 2 or more blank lines
                blank_line_sequences += 1
            consecutive_blank_lines = 0
    if consecutive_blank_lines >= 2: # Catch sequence at end of document
        blank_line_sequences += 1
    
    doc_context["avg_blank_lines_between_sections"] = blank_line_sequences # Simple count for now

    # --- 2. Adaptive Pattern Identification ---
    heading_prefixes_candidates = ["Chapter", "Section", "Appendix", "Figure", "Table", "Part", "Article", "Clause", "Introduction", "Conclusion", "Summary"]
    detected_prefixes = {p: 0 for p in heading_prefixes_candidates}
    numbering_patterns_counts = {"decimal_multi_level": 0, "decimal_single_level": 0, "alpha_upper": 0, "roman_upper": 0}

    for d in meaningful_chunks:
        text_start = d.page_content.strip().split('\n')[0] # Only check first line for patterns
        
        # Common Prefixes
        for prefix in heading_prefixes_candidates:
            if text_start.lower().startswith(prefix.lower()):
                detected_prefixes[prefix] += 1
        
        # Dominant Numbering Schemes
        if re.match(r'^\d+\.\d+\.\d+\.?\s', text_start):
            numbering_patterns_counts["decimal_multi_level"] += 1
        elif re.match(r'^\d+\.\s', text_start):
            numbering_patterns_counts["decimal_single_level"] += 1
        elif re.match(r'^[A-Z]\.\s', text_start):
            numbering_patterns_counts["alpha_upper"] += 1
        elif re.match(r'^[IVX]+\.\s', text_start):
            numbering_patterns_counts["roman_upper"] += 1
            
    doc_context["detected_heading_prefixes"] = [p for p, count in detected_prefixes.items() if count > 0]
    doc_context["dominant_numbering_pattern"] = max(numbering_patterns_counts, key=numbering_patterns_counts.get) if any(numbering_patterns_counts.values()) else "none"
    
    # Unstructured Category Distribution
    category_counts = {}
    for d in meaningful_chunks:
        category = d.metadata.get("category", "Unknown")
        category_counts[category] = category_counts.get(category, 0) + 1
    doc_context["category_distribution"] = category_counts

    # --- 3. Document Type Inference (Heuristic-based) ---
    inferred_type = "unknown"
    
    # Heuristics for technical/academic reports
    if (doc_context["prevalence_starts_with_numbering"] > 0.15 or doc_context["category_distribution"].get("Header", 0) > 5) and \
       (doc_context["category_distribution"].get("Table", 0) > 0 or doc_context["category_distribution"].get("FigureCaption", 0) > 0):
        inferred_type = "technical_report"
    # Heuristics for narrative documents (e.g., books)
    elif doc_context["prevalence_ends_with_punctuation"] > 0.7 and doc_context["mean_element_length"] > 300 and \
         doc_context["category_distribution"].get("NarrativeText", 0) > doc_context["category_distribution"].get("Title", 0) * 5:
        inferred_type = "narrative_document"
    # Heuristics for presentations (e.g., slides)
    elif doc_context["category_distribution"].get("Title", 0) + doc_context["category_distribution"].get("Header", 0) > len(meaningful_chunks) / 5 and \
         doc_context["mean_element_length"] < 200:
        inferred_type = "presentation_slides"
    # Heuristics for legal documents (more specific keywords)
    elif any(p in doc_context["detected_heading_prefixes"] for p in ["Article", "Clause"]) or \
         (doc_context["prevalence_all_caps_lines"] > 0.1 and doc_context["prevalence_starts_with_numbering"] > 0.05):
        inferred_type = "legal_document"
    
    doc_context["inferred_document_type"] = inferred_type
    
    logger.debug(f"Document context generated: {doc_context}")
    return doc_context

# -------- Universal Heading Detection & Hierarchy Extraction --------
def _extract_document_outline(
    meaningful_chunks: list[Document],
    doc_context: dict,
    heading_detection_confidence_threshold: float,
    min_chunks_per_section_for_merge: int,
    logger: logging.Logger
) -> list[dict]:
    """
    Extracts a hierarchical outline of the document by identifying headings and their levels.
    This forms Layer 2 of the intelligent chunking.
    """
    document_outline = []
    
    if not meaningful_chunks:
        logger.warning("No meaningful chunks provided for outline extraction. Returning empty outline.")
        return document_outline

    # Constants for scoring (can be tuned)
    WEIGHT_CATEGORY_MATCH = 0.4
    WEIGHT_LENGTH_PROFILE = 0.2
    WEIGHT_FORMATTING = 0.2
    WEIGHT_NUMBERING_PREFIX = 0.15
    WEIGHT_POSITIONAL = 0.05

    # Adaptive threshold adjustment based on document type
    adjusted_threshold = heading_detection_confidence_threshold
    if doc_context.get("inferred_document_type") == "technical_report":
        adjusted_threshold += 0.05 # Be slightly more stringent for formal docs
    elif doc_context.get("inferred_document_type") == "narrative_document":
        adjusted_threshold -= 0.05 # Be slightly more permissive for less formal docs
    adjusted_threshold = max(0.0, min(1.0, adjusted_threshold)) # Clamp between 0 and 1

    current_heading_stack = [] # Stores (level, section_id)

    # Helper to finalize a section
    def _finalize_section(current_section_start_idx, current_idx, current_heading_text, current_heading_level, current_confidence, current_parent_id, current_unstructured_category):
        if current_section_start_idx is not None and current_idx >= current_section_start_idx:
            document_outline.append({
                "section_id": str(uuid.uuid4()),
                "heading_level": current_heading_level,
                "heading_text": current_heading_text,
                "start_chunk_index": current_section_start_idx,
                "end_chunk_index": current_idx,
                "confidence": current_confidence,
                "parent_section_id": current_parent_id,
                "unstructured_category": current_unstructured_category
            })
            logger.debug(f"Finalized section: Level {current_heading_level}, Text: '{current_heading_text[:50]}...', Chunks: {current_section_start_idx}-{current_idx}")

    current_section_start_idx = 0
    current_heading_text = "Document Start"
    current_heading_level = 0 # Root level
    current_confidence = 1.0
    current_parent_id = None
    current_unstructured_category = "Document"

    for i, chunk in enumerate(meaningful_chunks):
        text = chunk.page_content.strip()
        category = chunk.metadata.get("category")
        
        if not text:
            continue

        # Calculate heading confidence score for the current chunk
        score = 0.0

        # 1. Unstructured Category Match
        if category in {"Title", "Header", "Subheader"}:
            score += WEIGHT_CATEGORY_MATCH
        elif category in {"ListItem", "Table", "FigureCaption", "Formula"}: # These are important but not primary headings
            score += WEIGHT_CATEGORY_MATCH * 0.5

        # 2. Length Profile Match (adaptive)
        if doc_context["short_element_threshold"] and len(text) <= doc_context["short_element_threshold"]:
            score += WEIGHT_LENGTH_PROFILE * (1 - (len(text) / doc_context["short_element_threshold"])) # Shorter means higher score
        elif doc_context["mean_element_length"] and len(text) < doc_context["mean_element_length"] / 2: # Also consider significantly shorter than mean
             score += WEIGHT_LENGTH_PROFILE * 0.5

        # 3. Line-Level Formatting
        first_line = text.splitlines()[0].strip()
        if first_line:
            if first_line.isupper() and len(first_line) > 5: # All caps, not too short
                score += WEIGHT_FORMATTING * doc_context["prevalence_all_caps_lines"]
            if first_line.istitle():
                score += WEIGHT_FORMATTING * doc_context["prevalence_title_case_lines"]
            if first_line[-1] not in ".!?" and len(first_line) > 10: # Doesn't end with sentence punctuation
                score += WEIGHT_FORMATTING * 0.5 # Fixed boost for non-sentence endings

        # 4. Numbering/Prefix Match
        if doc_context["prevalence_starts_with_numbering"] > 0.05 and re.match(r'^(\d+\.?|\d+\.\d+\.?|[A-Z]\.?|[IVX]+\.)\s', first_line):
            score += WEIGHT_NUMBERING_PREFIX
        elif any(first_line.lower().startswith(p.lower()) for p in doc_context["detected_heading_prefixes"]):
            score += WEIGHT_NUMBERING_PREFIX * 0.75

        if i > 0 and not meaningful_chunks[i-1].page_content.strip():
             score += WEIGHT_POSITIONAL * 0.5

        # Normalize score to 0-1 range (simple sum might exceed 1)
        heading_score = min(1.0, score / (WEIGHT_CATEGORY_MATCH + WEIGHT_LENGTH_PROFILE + WEIGHT_FORMATTING + WEIGHT_NUMBERING_PREFIX + WEIGHT_POSITIONAL))

        is_heading = heading_score >= adjusted_threshold
        logger.debug(f"Chunk {i}: '{text[:50]}...' Category: {category}, Score: {heading_score:.2f}, Is Heading: {is_heading}")

        if is_heading:
            # Finalize the previous section
            if i > current_section_start_idx: # Only finalize if current section has content
                _finalize_section(current_section_start_idx, i - 1, current_heading_text, current_heading_level, current_confidence, current_parent_id, current_unstructured_category)
            
            # Infer new heading level
            new_level = 1 # Default to level 1
            if re.match(r'^\d+\.\d+\.\d+\.?\s', first_line): new_level = 3
            elif re.match(r'^\d+\.\d+\.?\s', first_line): new_level = 2
            elif re.match(r'^\d+\.\s', first_line): new_level = 1
            elif re.match(r'^[A-Z]\.\s', first_line): new_level = 1 # A. Introduction
            elif category == "Subheader": new_level = (current_heading_level + 1) if current_heading_level < 6 else 6
            elif category == "Header": new_level = (current_heading_level + 1) if current_heading_level < 6 else 6
            elif category == "Title": new_level = 1 # Major title

            # Hierarchy enforcement: don't jump levels too much
            if current_heading_stack:
                last_level, _ = current_heading_stack[-1]
                if new_level > last_level + 1: # Cannot jump from L1 to L3 directly, make it L2
                    new_level = last_level + 1
                elif new_level < last_level: # Going up the hierarchy
                    while current_heading_stack and current_heading_stack[-1][0] >= new_level:
                        current_heading_stack.pop()
            
            # Update current section details
            current_section_start_idx = i
            current_heading_text = first_line
            current_heading_level = new_level
            current_confidence = heading_score
            current_parent_id = current_heading_stack[-1][1] if current_heading_stack else None
            current_unstructured_category = category

            # Update heading stack
            current_heading_stack.append((new_level, document_outline[-1]["section_id"] if document_outline else None))
            # If stack is empty, this is the first top-level heading, its parent is None.
            # If document_outline is empty, it means this is the very first section.

    # Finalize the last section after the loop
    _finalize_section(current_section_start_idx, len(meaningful_chunks) - 1, current_heading_text, current_heading_level, current_confidence, current_parent_id, current_unstructured_category)

    # --- Robust Fallback Strategies (if primary detection failed or was too sparse) ---
    if not document_outline or len(document_outline) < 2: # If less than 2 sections detected, or none
        logger.info("Primary heading detection yielded sparse results. Applying fallback strategies.")
        document_outline = [] # Reset outline

        # Fallback 1: Blank Line Sectioning
        if doc_context.get("avg_blank_lines_between_sections", 0) >= 1.5: # If blank lines are somewhat common
            current_fallback_section_start_idx = 0
            fallback_section_counter = 1
            for i in range(1, len(meaningful_chunks)):
                # Heuristic: if a significant break (e.g., 2+ lines) or a very short chunk followed by a long one
                if not meaningful_chunks[i-1].page_content.strip() and not meaningful_chunks[i].page_content.strip(): # Two consecutive empty lines
                    if i > current_fallback_section_start_idx:
                        document_outline.append({
                            "section_id": str(uuid.uuid4()),
                            "heading_level": 1,
                            "heading_text": f"Content Block {fallback_section_counter}",
                            "start_chunk_index": current_fallback_section_start_idx,
                            "end_chunk_index": i - 1,
                            "confidence": 0.4, # Lower confidence
                            "parent_section_id": None,
                            "unstructured_category": "FallbackSection"
                        })
                        fallback_section_counter += 1
                        current_fallback_section_start_idx = i
            # Add last fallback section
            if len(meaningful_chunks) > current_fallback_section_start_idx:
                 document_outline.append({
                    "section_id": str(uuid.uuid4()),
                    "heading_level": 1,
                    "heading_text": f"Content Block {fallback_section_counter}",
                    "start_chunk_index": current_fallback_section_start_idx,
                    "end_chunk_index": len(meaningful_chunks) - 1,
                    "confidence": 0.4,
                    "parent_section_id": None,
                    "unstructured_category": "FallbackSection"
                })
        
        # Fallback 2: Paragraph Grouping (if no sections yet or still too few)
        if not document_outline or len(document_outline) < 2:
            document_outline = [] # Reset again
            fallback_section_counter = 1
            chunks_per_group = 5 # Group into blocks of 5 meaningful chunks
            for i in range(0, len(meaningful_chunks), chunks_per_group):
                document_outline.append({
                    "section_id": str(uuid.uuid4()),
                    "heading_level": 1,
                    "heading_text": f"Content Block {fallback_section_counter}",
                    "start_chunk_index": i,
                    "end_chunk_index": min(i + chunks_per_group - 1, len(meaningful_chunks) - 1),
                    "confidence": 0.3, # Even lower confidence
                    "parent_section_id": None,
                    "unstructured_category": "FallbackParagraphGroup"
                })
                fallback_section_counter += 1

    # --- Post-processing: Merge very small sections ---
    # This loop might need to run multiple times if merges create new small sections
    merged_count = 0
    while True:
        initial_outline_len = len(document_outline)
        new_outline = []
        i = 0
        while i < len(document_outline):
            current_section = document_outline[i]
            section_content_length = current_section["end_chunk_index"] - current_section["start_chunk_index"] + 1
            
            # Only merge if it's a content section (level > 0) and has too few content elements
            if current_section["heading_level"] > 0 and section_content_length < min_chunks_per_section_for_merge:
                logger.debug(f"Merging small section: '{current_section['heading_text'][:50]}...' (len: {section_content_length})")
                merged_count += 1
                # Try to merge with previous section if available and same level or previous is parent
                if new_outline and new_outline[-1]["heading_level"] >= current_section["heading_level"]:
                    new_outline[-1]["end_chunk_index"] = current_section["end_chunk_index"]
                    # Optionally, update heading text to reflect combined content
                    new_outline[-1]["heading_text"] += f" / {current_section['heading_text']}"
                    new_outline[-1]["confidence"] = min(new_outline[-1]["confidence"], current_section["confidence"]) # Take lower confidence
                # Else, merge with next section
                elif i + 1 < len(document_outline):
                    next_section = document_outline[i+1]
                    next_section["start_chunk_index"] = current_section["start_chunk_index"]
                    next_section["heading_text"] = f"{current_section['heading_text']} / {next_section['heading_text']}"
                    next_section["confidence"] = min(next_section["confidence"], current_section["confidence"])
                    new_outline.append(next_section) # Add the modified next section
                    i += 1 # Skip next section as it's merged
                else: # Cannot merge, keep as is (should be rare with good min_chunks_per_section_for_merge)
                    new_outline.append(current_section)
            else:
                new_outline.append(current_section)
            i += 1
        
        document_outline = new_outline
        if len(document_outline) == initial_outline_len: # No more merges happened
            break
    
    logger.info(f"Post-processing merged {merged_count} small sections.")
    logger.debug(f"Final document outline: {document_outline}")
    return document_outline

# -------- Adaptive Hierarchical Semantic Chunking --------
def _generate_semantic_chunks(
    meaningful_chunks: list[Document],
    document_outline: list[dict],
    user_chunk_size: int,
    user_chunk_overlap: int,
    pinecone_metadata_max_bytes: int,
    parsed_global_custom_metadata: dict,
    document_specific_metadata_map: dict,
    document_id: str,
    file_name: str,
    sentence_split_threshold_chars: int,
    logger: logging.Logger
) -> list[Document]:
    """
    Generates final, semantically coherent, and context-rich chunks based on the document outline.
    This forms Layer 3 of the intelligent chunking.
    """
    final_semantic_chunks = []
    
    # Constants for internal re-chunking (from original code, now centralized)
    MIN_RE_SPLIT_CHUNK_SIZE = 100
    MAX_UTF8_BYTES_PER_CHAR = 4
    SAFETY_BUFFER_BYTES = 1024

    # Helper for internal re-splitting if metadata payload is too large
    def _resplit_chunk_for_pinecone_limit(text_content, original_metadata, current_logger):
        non_text_metadata_bytes = len(json.dumps(original_metadata).encode("utf-8"))
        remaining_space_for_text = pinecone_metadata_max_bytes - non_text_metadata_bytes - SAFETY_BUFFER_BYTES
        remaining_space_for_text = max(0, remaining_space_for_text)
        new_re_split_chunk_size_chars = max(MIN_RE_SPLIT_CHUNK_SIZE, remaining_space_for_text // MAX_UTF8_BYTES_PER_CHAR)

        current_logger.warning(f"Chunk too large for Pinecone metadata. Re-splitting with max_chars={new_re_split_chunk_size_chars}.")
        
        # Use a simple character splitter for this emergency re-split
        emergency_splitter = RecursiveCharacterTextSplitter(
            chunk_size=new_re_split_chunk_size_chars,
            chunk_overlap=0,
            length_function=len,
            add_start_index=True
        )
        temp_doc_for_resplit = Document(page_content=text_content, metadata={})
        sub_chunks = emergency_splitter.split_documents([temp_doc_for_resplit])
        
        resplit_docs = []
        for sc in sub_chunks:
            # Ensure original metadata is carried over and text is updated
            new_meta = original_metadata.copy()
            new_meta["text"] = sc.page_content # Update text field
            resplit_docs.append(Document(page_content=sc.page_content, metadata=new_meta))
        return resplit_docs

    # Iterate through each section identified in the outline
    for section_dict in document_outline:
        section_id = section_dict["section_id"]
        heading_level = section_dict["heading_level"]
        heading_text = section_dict["heading_text"]
        start_idx = section_dict["start_chunk_index"]
        end_idx = section_dict["end_chunk_index"]
        confidence = section_dict["confidence"]
        
        # Determine the content elements for this specific section
        content_elements_for_section = meaningful_chunks[start_idx : end_idx + 1]

        # Construct section prefix (e.g., markdown heading)
        section_prefix = f"#{'#' * min(heading_level, 6)} {heading_text}\n\n"
        
        current_chunk_content_elements = []
        current_chunk_raw_text_length = 0 # Length of text content, excluding prefix
        
        section_chunks_temp = [] # Chunks generated for this section before final metadata/ID assignment

    raw_semantic_chunks = [] # This will store Document objects for the section
    current_text_buffer = [] # Accumulates text parts for the current chunk
    current_metadata_accumulator = {} # Merges metadata for the current chunk

    # Define a threshold for what constitutes "substantial content" before a critical boundary forces a split
    MIN_CONTENT_FOR_BOUNDARY_SPLIT = max(100, user_chunk_size // 5) # e.g., 1/5th of target chunk size

    for i, element in enumerate(content_elements_for_section):
        element_text = element.page_content.strip()
        if not element_text:
            continue # Skip empty elements

        # Initialize accumulator with first element's metadata if it's empty
        if not current_metadata_accumulator:
            current_metadata_accumulator = element.metadata.copy()
        else:
            # Merge current element's metadata into the accumulator
            current_metadata_accumulator = _merge_chunk_metadata(current_metadata_accumulator, element.metadata, logger)

        # Handle very long individual elements by pre-splitting into sentences
        if len(element_text) > sentence_split_threshold_chars:
            logger.debug(f"Pre-splitting long element ({len(element_text)} chars) into sentences for section '{heading_text}'.")
            sentences = nltk.sent_tokenize(element_text)
            for sent_idx, sentence in enumerate(sentences):
                sentence_meta = element.metadata.copy()
                sentence_meta["start_index"] = sentence_meta.get("start_index", 0) + element_text.find(sentence)
                
                # Check if adding this sentence would cause a split
                temp_buffer_len = len(" ".join(current_text_buffer))
                if temp_buffer_len + len(sentence) > user_chunk_size and current_text_buffer:
                    # Finalize current chunk from buffer
                    raw_semantic_chunks.append(Document(
                        page_content=section_prefix + " ".join(current_text_buffer),
                        metadata=current_metadata_accumulator.copy()
                    ))
                    current_text_buffer = [sentence]
                else:
                    current_text_buffer.append(sentence)
        else:
            should_finalize_current_chunk = False
            current_buffer_content_length = len(" ".join(current_text_buffer))

            if current_buffer_content_length + len(element_text) > user_chunk_size:
                should_finalize_current_chunk = True
            elif element.metadata.get("category") in STRUCTURAL_CUES_AND_CRITICAL_CONCISE_CONTENT and \
                 current_buffer_content_length > MIN_CONTENT_FOR_BOUNDARY_SPLIT:
                should_finalize_current_chunk = True
            
            if should_finalize_current_chunk and current_text_buffer:
                raw_semantic_chunks.append(Document(
                    page_content=section_prefix + " ".join(current_text_buffer),
                    metadata=current_metadata_accumulator.copy()
                ))
                current_text_buffer = [element_text]
            else:
                current_text_buffer.append(element_text)

    # Finalize any remaining content in the buffer after the loop
    if current_text_buffer:
        raw_semantic_chunks.append(Document(
            page_content=section_prefix + " ".join(current_text_buffer),
            metadata=current_metadata_accumulator.copy()
        ))
    
    logger.info(f"Phase 1: Aggregated {len(content_elements_for_section)} meaningful elements into {len(raw_semantic_chunks)} raw semantic chunks for section '{heading_text}'.")

    # Phase 2: Explicit Overlap Generation
    chunks_with_overlap = []
    previous_chunk_overlap_text = ""
    previous_chunk_metadata_for_overlap = {} # Store minimal metadata for overlap context

    for i, chunk in enumerate(raw_semantic_chunks):
        current_chunk_content = chunk.page_content
        current_chunk_metadata = chunk.metadata.copy() # Start with the chunk's base metadata

        if previous_chunk_overlap_text:
            # Check if prepending overlap would make the chunk excessively large
            if len(current_chunk_content) + len(previous_chunk_overlap_text) > user_chunk_size * 1.5:
                # Trim overlap if it makes the chunk too large
                trimmed_overlap = previous_chunk_overlap_text[-(user_chunk_overlap // 2):] # Take last half of target overlap
                logger.warning(f"Trimmed overlap for chunk {i} ('{chunk.metadata.get('chunk_id', 'N/A')}') from {len(previous_chunk_overlap_text)} to {len(trimmed_overlap)} chars to prevent excessive chunk size.")
                current_chunk_content = trimmed_overlap + "\n\n" + current_chunk_content
            else:
                current_chunk_content = previous_chunk_overlap_text + "\n\n" + current_chunk_content
            
            current_chunk_metadata["has_leading_overlap"] = True

        sentences = nltk.sent_tokenize(chunk.page_content) # Use original content for overlap extraction
        overlap_sentences = []
        current_overlap_len = 0
        for sent in reversed(sentences): # Iterate backwards to get sentences from the end
            if current_overlap_len + len(sent) + 1 <= user_chunk_overlap: # +1 for space
                overlap_sentences.insert(0, sent) # Insert at beginning to maintain order
                current_overlap_len += len(sent) + 1
            else:
                break

        if overlap_sentences:
            previous_chunk_overlap_text = " ".join(overlap_sentences)
            previous_chunk_metadata_for_overlap = chunk.metadata.copy() # Store metadata for next iteration
        else:
            # Fallback to character-based if no semantic units fit
            previous_chunk_overlap_text = chunk.page_content[-user_chunk_overlap:]
            previous_chunk_metadata_for_overlap = chunk.metadata.copy()
            logger.debug(f"Fallback to character-based overlap for chunk {i}.")

        # Create the Document for this chunk with its content and metadata
        chunks_with_overlap.append(Document(page_content=current_chunk_content, metadata=current_chunk_metadata))

    logger.info(f"Phase 2: Applied overlap, resulting in {len(chunks_with_overlap)} chunks for section '{heading_text}'.")

    # Phase 3: Minimum Length Enforcement
    post_processed_chunks = []
    i = 0
    while i < len(chunks_with_overlap):
        current_chunk = chunks_with_overlap[i]
        
        if len(current_chunk.page_content) < min_chunk_length_ui:
            logger.debug(f"Chunk {i} ('{current_chunk.metadata.get('chunk_id', 'N/A')}') is short ({len(current_chunk.page_content)} chars). Attempting to merge.")
            
            merged = False
            # Try to merge with the next chunk (preferred)
            if i + 1 < len(chunks_with_overlap):
                next_chunk = chunks_with_overlap[i+1]
                merged_content = current_chunk.page_content + "\n\n" + next_chunk.page_content
                
                # Check if merging would make it excessively large
                if len(merged_content) < user_chunk_size * 1.5: # Allow some flexibility
                    merged_metadata = _merge_chunk_metadata(current_chunk.metadata, next_chunk.metadata, logger)
                    post_processed_chunks.append(Document(page_content=merged_content, metadata=merged_metadata))
                    logger.info(f"Merged short chunk {i} with next chunk {i+1}.")
                    i += 1 # Skip the next chunk as it's been merged
                    merged = True
                else:
                    logger.warning(f"Skipping merge of short chunk {i} with next chunk {i+1} as it would exceed 1.5x user_chunk_size ({len(merged_content)} chars).")
            
            if not merged:
                post_processed_chunks.append(current_chunk)
                logger.warning(f"Chunk {i} ('{current_chunk.metadata.get('chunk_id', 'N/A')}') is short ({len(current_chunk.page_content)} chars) and could not be merged. Adding as is.")
        else:
            post_processed_chunks.append(current_chunk)
        i += 1
    
    # If after all merging, we end up with fewer chunks, update the list
    if not post_processed_chunks and chunks_with_overlap: # Edge case: all chunks were short and merged, but nothing was added
        post_processed_chunks = chunks_with_overlap # Fallback to original if nothing was added
    elif not post_processed_chunks and not chunks_with_overlap:
        pass

    logger.info(f"Phase 3: Enforced minimum length, resulting in {len(post_processed_chunks)} chunks for section '{heading_text}'.")
      
    # ... (after Phase 3 logic) ...

    # Phase 4: Final Metadata Enrichment and Pinecone Limit Check (Existing logic, adapted)
    final_section_chunks = []
    for chunk_idx, chunk in enumerate(post_processed_chunks): # Iterate through post_processed_chunks
        final_content = chunk.page_content 
        initial_chunk_metadata = chunk.metadata.copy()

        sanitized_meta = _canonicalize_and_filter_metadata(
            initial_chunk_metadata,
            parsed_global_custom_metadata,
            document_specific_metadata_map.get(file_name, {}),
            logger
        )

        # Ensure core document/section metadata is present (it should be from _merge_chunk_metadata)
        # These lines are mostly for robustness, ensuring these critical fields are there.
        sanitized_meta.setdefault("document_id", document_id)
        sanitized_meta.setdefault("file_name", file_name)
        sanitized_meta.setdefault("section_id", section_id)
        sanitized_meta.setdefault("section_heading", heading_text)
        sanitized_meta.setdefault("heading_level", heading_level)
        sanitized_meta.setdefault("chunk_confidence", float(confidence))
        sanitized_meta.setdefault("chunk_index_in_section", chunk_idx) # Use current loop index
        sanitized_meta.setdefault("total_chunks_in_section", len(post_processed_chunks))

        # Generate a base chunk ID for this logical chunk (before potential re-splitting)
        base_chunk_id = deterministic_chunk_id(
            document_id,
            final_content,
            sanitized_meta.get("page_number", sanitized_meta.get("page", "")),
            sanitized_meta.get("start_index", "")
        )
        sanitized_meta["chunk_id"] = base_chunk_id

        temp_meta_with_full_text = sanitized_meta.copy()
        temp_meta_with_full_text["text"] = final_content

        total_chunk_size_bytes = 0
        try:
            total_chunk_size_bytes = len(json.dumps(temp_meta_with_full_text).encode("utf-8"))
        except TypeError:
            logger.error(f"Failed to JSON serialize metadata for chunk '{base_chunk_id}' during size estimation. Attempting string conversion.")
            safe_temp_meta = {}
            for k, v in temp_meta_with_full_text.items():
                try:
                    json.dumps({k: v})
                    safe_temp_meta[k] = v
                except Exception:
                    safe_temp_meta[k] = str(v)
            total_chunk_size_bytes = len(json.dumps(safe_temp_meta).encode("utf-8"))

        if total_chunk_size_bytes > pinecone_metadata_max_bytes:
            logger.warning(f"Chunk '{base_chunk_id}' (size: {total_chunk_size_bytes} bytes) exceeds Pinecone metadata limit ({pinecone_metadata_max_bytes} bytes). Re-splitting content.")
            metadata_for_resplitter = sanitized_meta.copy() 
            metadata_for_resplitter.pop("text", None) 
            resplit_docs = _resplit_chunk_for_pinecone_limit(final_content, metadata_for_resplitter, logger)

            for sub_idx, sc_doc in enumerate(resplit_docs):
                sub_chunk_id = f"{base_chunk_id}-{sub_idx}"
                sc_doc.metadata["chunk_id"] = sub_chunk_id                  
                sc_doc.metadata["text"] = sc_doc.page_content
                pruned_sub_meta = _shrink_metadata_to_limit(sc_doc.metadata, logger, pinecone_metadata_max_bytes)
                
                if pruned_sub_meta.get("text_truncated", False):
                    logger.critical(f"CRITICAL: Sub-chunk '{sub_chunk_id}' text was truncated even after re-splitting. This indicates a serious issue with metadata or very small chunk size configuration.")
                    sc_doc.page_content = pruned_sub_meta["text"]
                final_section_chunks.append(Document(page_content=sc_doc.page_content, metadata=pruned_sub_meta))
        else:
            sanitized_meta["text"] = final_content
            pruned_meta = _shrink_metadata_to_limit(sanitized_meta, logger, pinecone_metadata_max_bytes)
        
            if pruned_meta.get("text_truncated", False):
               logger.warning(f"Chunk '{base_chunk_id}' text was truncated to fit Pinecone metadata limit.")
               final_content = pruned_meta["text"]

            final_section_chunks.append(Document(page_content=final_content, metadata=pruned_meta))
        
    final_semantic_chunks.extend(final_section_chunks)

    logger.info(f"Generated {len(final_semantic_chunks)} final semantic chunks for document '{file_name}'.")
    return final_semantic_chunks

# -------- API Connection Test Function --------
def test_api_connections(pinecone_api_key: str, embedding_api_key: str, logger: logging.Logger):
    """
    Tests the provided Pinecone and OpenAI API keys for valid connections.
    Displays success or error messages in the Streamlit UI.
    """
    st.info("Attempting to test API connections...")

    # Test Pinecone API Key
    if pinecone_api_key:
        try:
            pc_test = Pinecone(api_key=pinecone_api_key)
            pc_test.list_indexes() # A simple call to verify connection
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
            embed_model_test = OpenAIEmbeddings(openai_api_key=embedding_api_key, model="text-embedding-3-small")
            embed_model_test.embed_query("test query") # Generate a small embedding to verify
            st.success("✅ OpenAI API Key is valid and can generate embeddings.")
            logger.info("OpenAI API connection test successful.")
        except Exception as e:
            st.error(f"❌ OpenAI API Key test failed: {e}. Please check your key or model permissions.")
            logger.error(f"OpenAI API connection test failed: {e}")
    else:
        st.warning("⚠️ OpenAI API Key not provided for testing.")

# -------- Document Management UI --------
def manage_documents_ui(pinecone_api_key: str, pinecone_index_name: str, namespace: str, embedding_dimension: int):
    """
    Provides a Streamlit UI for managing existing documents in the Pinecone index.
    Allows viewing index statistics, loading document names, viewing metadata,
    and deleting individual or all documents within a namespace.
    """
    logger = st.session_state.app_logger
    logger.debug("Entering manage_documents_ui function.")
    logger.info(f"Current namespace setting in UI: '{namespace}' (empty string means default).")

    # Initialize session state variables for UI control
    if 'delete_pending' not in st.session_state: st.session_state.delete_pending = False
    if 'file_to_delete_staged' not in st.session_state: st.session_state.file_to_delete_staged = ""
    if 'docid_to_delete_staged' not in st.session_state: st.session_state.docid_to_delete_staged = ""
    if 'bulk_delete_pending' not in st.session_state: st.session_state.bulk_delete_pending = False
    if 'all_document_names' not in st.session_state: st.session_state.all_document_names = []
    if 'metadata_display_doc_name' not in st.session_state: st.session_state.metadata_display_doc_name = None

    st.markdown("---")
    st.header("3. Manage Existing Documents")
    st.write("Load, search, and manage individual documents in the selected Pinecone namespace.")

    if not pinecone_api_key or not pinecone_index_name:
        st.info("Please provide Pinecone API Key and Index Name in the configuration above to manage documents.")
        logger.info("Skipping document management UI due to missing Pinecone API Key or Index Name.")
        return

    try:
        pc = Pinecone(api_key=pinecone_api_key)
        logger.info("Pinecone client initialized successfully for document management.")
    except Exception as e:
        logger.exception("Failed to initialize Pinecone client for document management.")
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
            st.metric(label="Total Vectors in Index", value=f"{stats.get('total_vector_count', 'N/A'):,}")

            namespaces_info = stats.get('namespaces', {})
            namespace_to_check = namespace or '__default__'
            current_ui_ns_count = namespaces_info.get(namespace_to_check, {}).get('vector_count',
                                                       namespaces_info.get('', {}).get('vector_count', 0))

            if namespace:
                st.metric(label=f"Vectors in current namespace `'{namespace}'`", value=f"{current_ui_ns_count:,}")
            else:
                st.metric(label=f"Vectors in default namespace", value=f"{current_ui_ns_count:,}")

            if namespaces_info:
                with st.expander("View All Namespaces & Counts"):
                    if namespaces_info:
                        for ns_name, ns_data in namespaces_info.items():
                            vector_count = ns_data.get('vector_count', 0)
                            display_ns_name = f"'{ns_name}'" if ns_name else "Default (`''` or `__default__`)"
                            st.write(f"- Namespace {display_ns_name}: `{vector_count:,}` vectors")
                            logger.info(f"Namespace '{ns_name}' has {vector_count} vectors.")
                    else:
                        st.info("No namespaces found in this index.")
                        logger.info("No namespaces found in index stats.")

            logger.info(f"Displayed index stats for '{pinecone_index_name}'. Total vectors: {stats.get('total_vector_count')}, Current UI target namespace '{namespace_to_check}' vectors: {current_ui_ns_count}")
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
        help="Specify the maximum number of document names to attempt to load from Pinecone. Higher values may be slower and might not retrieve all names if the index is very large (Pinecone's query limit is 10,000)."
    )

    if st.button("Load Document Names", key="load_all_docs_button"):
        logger.info(f"Loading document names requested with limit: {load_limit}.")
        if not index:
            st.warning("Index not available. Ensure Pinecone index exists and is reachable.")
            logger.warning("Loading document names aborted: Index not available.")
        else:
            with st.spinner(f"Loading up to {load_limit} document names from namespace `'{namespace or '__default__'}'`. This may take a moment for large indexes..."):
                try:
                    dummy_vec = [0.0] * embedding_dimension
                    q = index.query(vector=dummy_vec, top_k=load_limit, include_metadata=True, namespace=(namespace or None))
                    matches = getattr(q, "matches", None) or q.get("matches", [])

                    unique_doc_names = set()
                    for m in matches:
                        md = m.get("metadata") or {}
                        fn = md.get("file_name")
                        if not fn:
                            fn = md.get("document_id")
                        if fn:
                            unique_doc_names.add(fn)

                    st.session_state.all_document_names = sorted(list(unique_doc_names))
                    logger.info(f"Loaded {len(st.session_state.all_document_names)} unique document names (up to {load_limit} requested) from namespace '{namespace or '__default__'}'.")
                    st.success(f"Successfully loaded {len(st.session_state.all_document_names)} unique document names.")

                    if current_ui_ns_count > load_limit and current_ui_ns_count != 'N/A':
                        st.warning(f"Note: Your namespace contains approximately {current_ui_ns_count:,} vectors. Only a sample of {load_limit:,} was queried. This list might not be exhaustive and may not include all documents.")

                except Exception as e:
                    logger.exception("Failed to load document names from index.")
                    st.error(f"Error loading document names: {e}")
            st.session_state.metadata_display_doc_name = None

    all_document_names = st.session_state.get("all_document_names", [])

    search_query = st.text_input("Filter loaded documents by name:", key="doc_search_input", help="Type to filter the list of loaded documents.", value="")

    filtered_document_names = [name for name in all_document_names if search_query.lower() in name.lower()]

    if filtered_document_names:
        st.write(f"Displaying {len(filtered_document_names)} of {len(all_document_names)} loaded documents:")
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
            st.subheader(f"Metadata for: `{st.session_state.metadata_display_doc_name}`")
            st.info("Displaying metadata from a single representative chunk. Full document metadata may vary across chunks.")
            if index:
                try:
                    query_filter = {"file_name": st.session_state.metadata_display_doc_name}
                    # Heuristic to check if it's likely a document_id (SHA256 hash)
                    if "_" not in st.session_state.metadata_display_doc_name and len(st.session_state.metadata_display_doc_name) == 64 and all(c in '0123456789abcdef' for c in st.session_state.metadata_display_doc_name.lower()):
                         query_filter = {"document_id": st.session_state.metadata_display_doc_name}

                    q = index.query(
                        vector=[0.0] * embedding_dimension,
                        top_k=1,
                        filter=query_filter,
                        include_metadata=True,
                        namespace=(namespace or None)
                    )
                    if q.matches and q.matches[0].metadata:
                        st.json(q.matches[0].metadata)
                        logger.info(f"Displayed metadata for a chunk of '{st.session_state.metadata_display_doc_name}'.")
                    else:
                        st.warning("No metadata found for this document, or no matching chunks.")
                        logger.warning(f"No metadata found for '{st.session_state.metadata_display_doc_name}' during display.")
                except Exception as e:
                    logger.exception(f"Error fetching metadata for '{st.session_state.metadata_display_doc_name}'.")
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
        help="Enter the exact 'file_name' or 'document_id' of a document to delete its associated vectors. This is useful if the document is not listed above or if you want to delete by its unique ID."
    )
    if st.button("Initiate Deletion for Specific Document", key="initiate_direct_delete_button"):
        if doc_name_or_id_to_delete:
            if len(doc_name_or_id_to_delete) == 64 and all(c in '0123456789abcdef' for c in doc_name_or_id_to_delete.lower()):
                st.session_state.docid_to_delete_staged = doc_name_or_id_to_delete
                st.session_state.file_to_delete_staged = ""
                logger.info(f"Deletion staged for specific document ID: {doc_name_or_id_to_delete}")
            else:
                st.session_state.file_to_delete_staged = doc_name_or_id_to_delete
                st.session_state.docid_to_delete_staged = ""
                logger.info(f"Deletion staged for specific file name: {doc_name_or_id_to_delete}")
            st.session_state.delete_pending = True
            st.rerun()
        else:
            st.warning("Please enter a Document Name or Document ID to delete.")

    st.markdown("---")
    st.subheader("Bulk Actions")
    st.write("Perform actions on all documents within the current namespace.")

    if st.button("Delete ALL documents in current namespace", key="bulk_delete_namespace_button"):
        st.session_state.bulk_delete_pending = True
        logger.info(f"Bulk deletion for namespace '{namespace or '__default__'}' prepared.")
        st.rerun()

    if st.session_state.bulk_delete_pending:
        st.markdown("---")
        st.subheader("Confirm Bulk Deletion")
        st.error(f"WARNING: You are about to permanently delete ALL documents from Pinecone index `{pinecone_index_name}` in namespace `'{namespace or '__default__'}'`.")
        st.write("This action cannot be undone.")

        bulk_confirm_checkbox = st.checkbox("I understand this will delete ALL vectors in the current namespace permanently.", value=False, key="bulk_delete_confirm_checkbox")

        col_bulk_exec, col_bulk_cancel = st.columns(2)
        with col_bulk_exec:
            if st.button("Execute BULK Deletion", key="execute_bulk_delete_button", disabled=not bulk_confirm_checkbox):
                logger.info(f"Execute BULK Deletion button clicked for namespace '{namespace or '__default__'}'.")
                if not index:
                    st.error("Index not available. Ensure Pinecone index exists and is reachable.")
                    logger.error("Bulk deletion aborted: Index not available.")
                    st.session_state.bulk_delete_pending = False
                    st.rerun()
                    return
                try:
                    index.delete(delete_all=True, namespace=(namespace or None))
                    st.success(f"Successfully initiated bulk deletion for namespace `'{namespace or '__default__'}'`.")
                    logger.info(f"Bulk deletion request sent for namespace '{namespace or '__default__'}'.")
                    time.sleep(3)
                    st.session_state.bulk_delete_pending = False
                    st.session_state.all_document_names = []
                    st.session_state.metadata_display_doc_name = None
                    st.rerun()
                except Exception as e:
                    logger.exception(f"Bulk deletion failed for namespace '{namespace or '__default__'}'.")
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
        st.write(f"From Pinecone index `{pinecone_index_name}` in namespace `'{namespace or '__default__'}'`.")

        confirm_checkbox = st.checkbox("I understand this action is permanent and cannot be undone.", value=False, key="delete_confirm_checkbox")

        col_exec, col_cancel = st.columns(2)
        with col_exec:
            if st.button("Execute Deletion", key="execute_delete_button", disabled=not confirm_checkbox):
                logger.info("Execute Deletion button clicked.")
                file_to_delete = st.session_state.file_to_delete_staged
                docid_to_delete = st.session_state.docid_to_delete_staged

                logger.debug(f"Executing delete with: file_to_delete='{file_to_delete}', docid_to_delete='{docid_to_delete}', target_namespace='{namespace or '__default__'}'.")

                if not index:
                    st.error("Index not available. Ensure Pinecone index exists and is reachable.")
                    logger.error("Deletion aborted: Index not available.")
                    st.session_state.delete_pending = False
                    st.rerun()
                    return

                try:
                    deleted_summaries = []

                    if file_to_delete:
                        logger.info(f"Attempting to delete records with file_name == '{file_to_delete}' in namespace '{namespace or '__default__'}'.")
                        index.delete(filter={"file_name": file_to_delete}, namespace=(namespace or None))
                        deleted_summaries.append(f"Deleted records with file_name == '{file_to_delete}'")
                        logger.info(f"Delete request sent for file_name '{file_to_delete}'.")

                    if docid_to_delete:
                        logger.info(f"Attempting to delete record with document_id == '{docid_to_delete}' in namespace '{namespace or '__default__'}'.")
                        index.delete(ids=[docid_to_delete], namespace=(namespace or None))
                        deleted_summaries.append(f"Deleted record with document_id == '{docid_to_delete}'")
                        logger.info(f"Delete request sent for document_id '{docid_to_delete}'.")

                    st.success("Deletion successfully initiated:")
                    for s in deleted_summaries:
                        st.write(f"- {s}")
                    logger.info(f"Deletion summaries: {deleted_summaries}")

                    time.sleep(2)
                    logger.info("Paused for 2 seconds for Pinecone eventual consistency.")

                    st.session_state.all_document_names = []
                    st.session_state.metadata_display_doc_name = None

                    try:
                        stats_after = index.describe_index_stats()
                        st.info(f"Index stats (after delete):")
                        st.json(stats_after.to_dict())
                        logger.info(f"Retrieved index stats after deletion: {stats_after.to_dict()}")
                    except Exception as e:
                        logger.exception("Failed to retrieve index stats after deletion.")
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
        initial_sidebar_state="auto"
    )
    st.title("📚 Pinecone Ingestor")
    st.markdown(
        """
        This tool helps you build and manage a Retrieval-Augmented Generation (RAG) knowledge base
        by ingesting your documents into a Pinecone vector database.
        """
    )

    load_dotenv() # Load environment variables from .env file
    current_logging_level_name = os.environ.get("LOGGING_LEVEL", DEFAULT_SETTINGS["LOGGING_LEVEL"])
    logger = setup_logging_once(level=logging.getLevelName(current_logging_level_name))
    logger.info("Environment variables loaded from .env (if present).")

    # Load SpaCy model for NER filtering, caching it for performance
    nlp = load_spacy_model()
    if nlp is None:
        st.session_state.enable_ner_filtering = False # Disable NER if model failed to load
        logger.warning("NER filtering disabled due to SpaCy model loading failure.")
    else:
        if 'enable_ner_filtering' not in st.session_state:
            st.session_state.enable_ner_filtering = (os.environ.get("ENABLE_NER_FILTERING", "True").lower() == "true")

    # Initialize session state variables for UI control and configuration
    if "show_reset_dialog" not in st.session_state: st.session_state.show_reset_dialog = False
    if "config_form_key" not in st.session_state: st.session_state.config_form_key = 0

    if 'dynamic_metadata_fields' not in st.session_state:
        env_metadata = os.environ.get("CUSTOM_METADATA", DEFAULT_SETTINGS["CUSTOM_METADATA"])
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
    st.write("Set up your API keys, Pinecone index details, and document processing parameters.")

    # Retrieve current API key values for pre-filling inputs and external button access
    pinecone_api_key_val = os.environ.get("PINECONE_API_KEY", DEFAULT_SETTINGS["PINECONE_API_KEY"])
    embedding_api_key_val = os.environ.get("EMBEDDING_API_KEY", DEFAULT_SETTINGS["EMBEDDING_API_KEY"])

    with st.form(key=f"config_form_{st.session_state.config_form_key}"):
        # API Keys section within an expander for sensitive information
        with st.expander("🔑 API Keys (Sensitive)", expanded=True):
            pinecone_api_key = st.text_input(
                "Pinecone API Key", type="password",
                value=pinecone_api_key_val,
                help="Your Pinecone API key used to authenticate API requests. Keep this secret and do not share it."
            )
            embedding_api_key = st.text_input(
                "OpenAI API Key (for Embeddings)", type="password",
                value=embedding_api_key_val,
                help="Your OpenAI API key. Required to generate vector embeddings using OpenAI's models."
            )

        st.subheader("Index & Model Settings")
        pinecone_index_name = st.text_input(
            "Pinecone Index Name",
            value=os.environ.get("PINECONE_INDEX_NAME", DEFAULT_SETTINGS["PINECONE_INDEX_NAME"]),
            help="The name of the Pinecone index where your vectors will be stored. Choose a unique name."
        )

        region_options = list(SUPPORTED_PINECONE_REGIONS.keys())
        current_region = os.environ.get("PINECONE_CLOUD_REGION", DEFAULT_SETTINGS["PINECONE_CLOUD_REGION"])
        default_region_index = region_options.index(current_region) if current_region in region_options else 0
        pinecone_cloud_region = st.selectbox(
            "Pinecone Cloud Region",
            options=region_options,
            index=default_region_index,
            help=(
                "Select the cloud provider and region where your Pinecone index will be hosted. "
                "Free tier supports only 'aws-us-east-1'. Choose accordingly."
            )
        )

        embedding_model_name = st.text_input(
            "Embedding Model Name",
            value=os.environ.get("EMBEDDING_MODEL_NAME", DEFAULT_SETTINGS["EMBEDDING_MODEL_NAME"]),
            help="The embedding model identifier (e.g., 'text-embedding-3-small'). Choose based on your provider's available models."
        )
        embedding_dimension = st.number_input(
            "Embedding Dimension",
            min_value=1,
            value=int(os.environ.get("EMBEDDING_DIMENSION", DEFAULT_SETTINGS["EMBEDDING_DIMENSION"])),
            help="Dimensionality of the embedding vectors. Typically 1536 for 'text-embedding-3-small'."
        )
        metric_type = st.selectbox(
            "Pinecone Metric Type",
            options=["cosine", "euclidean", "dotproduct"],
            index=["cosine", "euclidean", "dotproduct"].index(os.environ.get("METRIC_TYPE", DEFAULT_SETTINGS["METRIC_TYPE"])),
            help=(
                "Similarity metric used by Pinecone for vector search. "
                "'cosine' is common for semantic similarity. Choose based on your embedding model's training."
            )
        )
        namespace = st.text_input(
            "Namespace (Optional)",
            value=os.environ.get("NAMESPACE", DEFAULT_SETTINGS["NAMESPACE"]),
            help=(
                "Optional partition within the Pinecone index to isolate data (e.g., per customer or project). "
                "Leave blank to use the default namespace."
            )
        )

        st.subheader("Document Processing Settings")

        # Unstructured Loader Strategy selection
        unstructured_strategy_options = ["hi_res", "fast", "auto"]
        current_unstructured_strategy = os.environ.get("UNSTRUCTURED_STRATEGY", DEFAULT_SETTINGS["UNSTRUCTURED_STRATEGY"])
        default_strategy_index = unstructured_strategy_options.index(current_unstructured_strategy) if current_unstructured_strategy in unstructured_strategy_options else 0
        unstructured_strategy = st.selectbox(
            "Unstructured Loader Strategy",
            options=unstructured_strategy_options,
            index=default_strategy_index,
            help=(
                "Select the strategy for Unstructured.io document parsing: "
                "'hi_res' (high-resolution, slower, more accurate), "
                "'fast' (quicker, less accurate), or "
                "'auto' (Unstructured decides based on document type)."
            )
        )

        chunk_size = st.number_input(
            "Chunk Size (characters)",
            min_value=256,
            value=int(os.environ.get("CHUNK_SIZE", DEFAULT_SETTINGS["CHUNK_SIZE"])),
            help=(
                "Maximum size of each text chunk to embed, measured in characters. "
                "This will be dynamically adjusted if needed to fit Pinecone's metadata limit."
            )
        )
        chunk_overlap = st.number_input(
            "Chunk Overlap (characters)",
            min_value=0,
            value=int(os.environ.get("CHUNK_OVERLAP", DEFAULT_SETTINGS["CHUNK_OVERLAP"])),
            help=(
                "Number of characters overlapping between consecutive chunks to maintain context. "
                "Typical values are 100-200."
            )
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
            key="config_enable_filtering_checkbox"
        )
        whitelisted_keywords_input = st.text_input(
            "Whitelisted Keywords (comma-separated)",
            value=os.environ.get("WHITELISTED_KEYWORDS", DEFAULT_SETTINGS["WHITELISTED_KEYWORDS"]),
            help=(
                "Enter words or short phrases that should always be kept, even if short or generic. "
                "E.g., 'Error Code', 'API Key', 'Product ID'. Case-insensitive matching."
            ),
            key="config_whitelisted_keywords_input"
        )
        min_generic_content_length_ui = st.number_input(
            "Min Length for Generic Content (characters)",
            min_value=0,
            value=int(os.environ.get("MIN_GENERIC_CONTENT_LENGTH", DEFAULT_SETTINGS["MIN_GENERIC_CONTENT_LENGTH"])),
            help=(
                "Minimum character length for generic text segments to be included. "
                "Text shorter than this, and not categorized as important or whitelisted, will be filtered out. "
                "A lower value may capture more short facts but could introduce noise."
            ),
            key="config_min_generic_content_length_input"
        )
        enable_ner_filtering = st.checkbox(
            "Enable NER Filtering for Short Generic Content",
            value=st.session_state.get("enable_ner_filtering", DEFAULT_SETTINGS["ENABLE_NER_FILTERING"].lower() == "true"),
            help=(
                "If enabled, short generic text (below min length) will be kept if it contains recognized Named Entities (e.g., dates, organizations, persons). "
                "Requires SpaCy 'en_core_web_sm' model. NER filtering is automatically disabled if SpaCy model fails to load."
            ),
            disabled=(nlp is None),
            key="config_enable_ner_filtering_checkbox"
        )
        if enable_ner_filtering and nlp is None:
            st.warning("SpaCy 'en_core_web_sm' model is required for NER filtering but failed to load. Please install it using `python -m spacy download en_core_web_sm` and restart the app.")

        document_metadata_file = st.file_uploader(
            "Upload Document-Specific Metadata (CSV/JSON, optional)",
            type=["csv", "json"],
            accept_multiple_files=False,
            help=(
                "Upload a CSV or JSON file containing metadata for individual documents. "
                "For CSV, include a 'file_name' column. For JSON, use an array of objects with a 'file_name' key. "
                "This metadata will override global settings for matching documents. "
                f"Note: Reserved keys ({', '.join(RESERVED_METADATA_KEYS)}) will be ignored."
            )
        )

        overwrite_existing_docs = st.checkbox(
            "Overwrite existing documents with the same file name?",
            value=(os.environ.get("OVERWRITE_EXISTING_DOCS", "False").lower() == "true"),
            help=(
                "If checked, any existing vectors in Pinecone associated with uploaded files (matched by file name) "
                "will be deleted before new chunks are uploaded. Use with caution."
            ),
            key="config_overwrite_checkbox"
        )

        with st.expander("⚙️ Advanced Chunking Settings", expanded=False):
            heading_detection_confidence_threshold_ui = st.slider(
                "Heading Detection Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=float(os.environ.get("HEADING_DETECTION_CONFIDENCE_THRESHOLD", DEFAULT_SETTINGS["HEADING_DETECTION_CONFIDENCE_THRESHOLD"])),
                step=0.01,
                format="%.2f",
                help=(
                    "Minimum confidence score for a text segment to be classified as a heading by Layer 2. "
                    "Higher values are more strict, lower values are more permissive. Default: 0.65."
                ),
                key="config_heading_detection_confidence_threshold"
            )

            min_chunks_per_section_for_merge_ui = st.number_input(
                "Min Content Elements Per Section (for merging)",
                min_value=1,
                value=int(os.environ.get("MIN_CHUNKS_PER_SECTION_FOR_MERGE", DEFAULT_SETTINGS["MIN_CHUNKS_PER_SECTION_FOR_MERGE"])),
                step=1,
                help=(
                    "Minimum number of meaningful content elements (paragraphs, list items, etc.) a section must contain "
                    "after its heading. Sections with fewer elements will be merged with neighbors during Layer 2 post-processing. Default: 2."
                ),
                key="config_min_chunks_per_section_for_merge"
            )

            sentence_split_threshold_chars_ui = st.number_input(
                "Sentence Split Threshold (characters)",
                min_value=100,
                value=int(os.environ.get("SENTENCE_SPLIT_THRESHOLD_CHARS", DEFAULT_SETTINGS["SENTENCE_SPLIT_THRESHOLD_CHARS"])),
                step=50,
                help=(
                    "For sections with medium/low confidence structure, if an individual content element (e.g., a long paragraph) "
                    "exceeds this character length, it may be further split into sentences to manage chunk size. Default: 300."
                ),
                key="config_sentence_split_threshold_chars"
            )
            
            min_chunk_length_ui = st.number_input(
                "Minimum Chunk Length (characters)",
                min_value=1,
                value=int(os.environ.get("MIN_CHUNK_LENGTH", DEFAULT_SETTINGS["MIN_CHUNK_LENGTH"])),
                step=10,
                help=(
                    "Minimum character length for a final chunk. Chunks shorter than this will be merged with adjacent chunks "
                    "if possible, to ensure meaningful content. Default: 100."
                ),
                key="config_min_chunk_length"
            )
        # Advanced Logging Settings
        with st.expander("Advanced Logging Settings", expanded=False):
            logging_level_options = ["DEBUG", "INFO", "WARNING", "ERROR"]
            current_logging_level_name_from_env = os.environ.get("LOGGING_LEVEL", DEFAULT_SETTINGS["LOGGING_LEVEL"])
            default_logging_index = logging_level_options.index(current_logging_level_name_from_env) if current_logging_level_name_from_env in logging_level_options else 1
            logging_level_selected = st.selectbox(
                "Logging Level",
                options=logging_level_options,
                index=default_logging_index,
                help="Set the verbosity of the application logs. DEBUG is most verbose, ERROR is least verbose."
            )

            if logging.getLevelName(logger.level) != logging_level_selected:
                logger.setLevel(logging_level_selected)
                logger.info(f"Logging level dynamically set to {logging_level_selected}.")

        # Form submission buttons
        col1, col2 = st.columns(2)
        with col1:
            save_conf = st.form_submit_button("💾 Save Configuration (local .env)")
        with col2:
            reset_conf = st.form_submit_button("🔄 Reset to Defaults (local only)")

    # Test API Connections button (outside the form, uses current state of API key inputs)
    if st.button("Test API Connections", key="test_api_connections_button_outside_form"):
        test_api_connections(pinecone_api_key, embedding_api_key, logger)

    # Logic for saving configuration to .env file
    if save_conf:
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
            "HEADING_DETECTION_CONFIDENCE_THRESHOLD": str(heading_detection_confidence_threshold_ui),
            "MIN_CHUNKS_PER_SECTION_FOR_MERGE": str(min_chunks_per_section_for_merge_ui),
            "SENTENCE_SPLIT_THRESHOLD_CHARS": str(sentence_split_threshold_chars_ui),
            "MIN_CHUNK_LENGTH": str(min_chunk_length_ui),
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
            defaults["CUSTOM_METADATA"] = json.dumps(st.session_state.dynamic_metadata_fields)
            defaults["HEADING_DETECTION_CONFIDENCE_THRESHOLD"] = DEFAULT_SETTINGS["HEADING_DETECTION_CONFIDENCE_THRESHOLD"]
            defaults["MIN_CHUNKS_PER_SECTION_FOR_MERGE"] = DEFAULT_SETTINGS["MIN_CHUNKS_PER_SECTION_FOR_MERGE"]
            defaults["SENTENCE_SPLIT_THRESHOLD_CHARS"] = DEFAULT_SETTINGS["SENTENCE_SPLIT_THRESHOLD_CHARS"]
            defaults["MIN_CHUNK_LENGTH"] = DEFAULT_SETTINGS["MIN_CHUNK_LENGTH"]

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
    st.write("Define custom key-value pairs here. These will be added to all uploaded document chunks.")
    st.info(f"Note: Reserved keys ({', '.join(RESERVED_METADATA_KEYS)}) will be ignored or overwritten by internal values.")
    st.markdown("Example: `{\"project_name\": \"MyRAGProject\", \"department\": \"Engineering\"}`")

    # Dynamic UI for adding/removing custom metadata fields
    for i, field in enumerate(st.session_state.dynamic_metadata_fields):
        cols = st.columns([0.45, 0.45, 0.1])
        with cols[0]:
            field['key'] = st.text_input(f"Key {i+1}", value=field['key'], key=f"meta_key_{i}", label_visibility="collapsed")
        with cols[1]:
            field['value'] = st.text_input(f"Value {i+1}", value=field['value'], key=f"meta_value_{i}", label_visibility="collapsed")
        with cols[2]:
            if len(st.session_state.dynamic_metadata_fields) > 1 or (len(st.session_state.dynamic_metadata_fields) == 1 and i == 0 and (field['key'] != "" or field['value'] != "")):
                if st.button("ðŸ—‘ï¸", key=f"remove_meta_{i}", help="Remove this custom field"):
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
        if st.session_state.dynamic_metadata_fields and st.button("➖ Remove Last Field", key="remove_last_meta_field"):
            st.session_state.dynamic_metadata_fields.pop()
            st.rerun()
    st.markdown("---")

    st.header("2. Upload Documents")
    st.write("Upload your documents here to embed them and upsert into your Pinecone knowledge base.")

    uploaded_files = st.file_uploader(
        "Select documents to upload",
        type=[
            "pdf", "txt", "md", "docx", "xlsx", "pptx", "csv", "html", "xml",
            "eml", "epub", "rtf", "odt", "org", "rst", "tsv"
        ],
        accept_multiple_files=True,
        help="Supported file types include common document, spreadsheet, and presentation formats."
    )

    # Display uploaded file names for better user experience
    if uploaded_files:
        st.subheader("Uploaded Files:")
        for i, file in enumerate(uploaded_files):
            st.markdown(f"- {file.name}")
    else:
        st.info("No files selected yet.")

    # Input validation for the main processing button
    can_process = bool(pinecone_api_key and embedding_api_key and pinecone_index_name and uploaded_files)
    if not can_process:
        st.warning("Please ensure Pinecone API Key, OpenAI API Key, Pinecone Index Name are set and at least one file is uploaded to enable processing.")

    if st.button("🚀 Process, Embed & Upsert to Pinecone", disabled=not can_process):
        logger.info("Process, Embed & Upsert button clicked.")
        logger.debug(f"Pinecone Index: {pinecone_index_name}, Namespace: {namespace or 'default'}, Region: {pinecone_cloud_region}")
        logger.debug(f"Embedding Model: {embedding_model_name}, Dimension: {embedding_dimension}, Metric: {metric_type}")
        logger.debug(f"Chunk Size: {chunk_size}, Chunk Overlap: {chunk_overlap}, Unstructured Strategy: {unstructured_strategy}")
        logger.debug(f"Overwrite existing documents setting: {overwrite_existing_docs}")
        logger.debug(f"Filtering Enabled: {enable_filtering}, Whitelisted Keywords: '{whitelisted_keywords_input}', Min Generic Content Length: {min_generic_content_length_ui}, NER Filtering Enabled: {enable_ner_filtering}")

        # Parse whitelisted keywords for efficient lookup
        whitelisted_keywords_set = {k.strip().lower() for k in whitelisted_keywords_input.split(',') if k.strip()}
        logger.info(f"Parsed whitelisted keywords: {whitelisted_keywords_set}")

        # Process global custom metadata from UI inputs
        parsed_global_custom_metadata = {}
        for i, field in enumerate(st.session_state.dynamic_metadata_fields):
            key = field['key'].strip()
            value = field['value']
            if key:
                if key in RESERVED_METADATA_KEYS:
                    logger.warning(f"Global custom metadata: Key '{key}' is reserved and will be ignored or overwritten.")
                    st.warning(f"Warning: Custom metadata key '{key}' is reserved and will be ignored or overwritten.")
                try:
                    parsed_value = json.loads(value)
                    parsed_global_custom_metadata[key] = parsed_value
                except (json.JSONDecodeError, TypeError):
                    parsed_global_custom_metadata[key] = value
        logger.info(f"Global custom metadata generated from dynamic fields: {parsed_global_custom_metadata}")

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
                    if 'file_name' not in df.columns:
                        raise ValueError("CSV metadata file must contain a 'file_name' column.")
                    for _, row in df.iterrows():
                        file_name_val = row['file_name']
                        doc_md = {}
                        for k, v in row.drop('file_name').items():
                            if k not in RESERVED_METADATA_KEYS:
                                try:
                                    parsed_v = json.loads(str(v))
                                    doc_md[k] = parsed_v
                                except (json.JSONDecodeError, TypeError):
                                    doc_md[k] = str(v)
                            else:
                                logger.warning(f"Document-specific CSV metadata for '{file_name_val}': Key '{k}' is reserved and will be ignored.")
                        document_specific_metadata_map[file_name_val] = doc_md
                    logger.info(f"Loaded {len(document_specific_metadata_map)} document-specific metadata entries from CSV.")
                elif document_metadata_file.type == "application/json":
                    json_data = json.loads(file_content)
                    if not isinstance(json_data, list):
                        raise ValueError("JSON metadata file must be an array of objects.")
                    for entry in json_data:
                        if 'file_name' not in entry:
                            raise ValueError("Each object in JSON metadata array must contain a 'file_name' key.")
                        file_name_val = entry['file_name']
                        cleaned_entry = {}
                        for k, v in entry.items():
                            if k == 'file_name': continue
                            if k not in RESERVED_METADATA_KEYS:
                                cleaned_entry[k] = v
                            else:
                                logger.warning(f"Document-specific JSON metadata for '{file_name_val}': Key '{k}' is reserved and will be ignored.")
                        document_specific_metadata_map[file_name_val] = cleaned_entry
                    logger.info(f"Loaded {len(document_specific_metadata_map)} document-specific metadata entries from JSON.")
                st.success(f"Successfully loaded document-specific metadata from '{document_metadata_file.name}'.")
            except Exception as e:
                logger.exception(f"Error processing document-specific metadata file '{document_metadata_file.name}'.")
                st.error(f"Error processing document-specific metadata file: {e}. Please check its format.")
                return

        # Initialize Pinecone client
        try:
            pc = Pinecone(api_key=pinecone_api_key) if pinecone_api_key else None
            if pc:
                logger.info("Pinecone client initialized for upsert process.")
            else:
                logger.warning("Pinecone API key not provided, Pinecone client not initialized.")
        except Exception as e:
            logger.exception("Pinecone client initialization failed for upsert process.")
            st.error(f"Pinecone initialization error: {e}. Please check your API key.")
            pc = None

        # --- PHASE 1: Generate and Display Initial Processing Plan ---
        files_to_process_plan = []
        plan_summary_messages = []

        st.subheader("Processing Plan Summary")
        plan_summary_placeholder = st.empty()

        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            file_bytes = uploaded_file.getvalue()
            document_id = deterministic_document_id(file_name, file_bytes)

            status_message = ""
            should_process = True

            if pc and pc.has_index(pinecone_index_name):
                idx = pc.Index(pinecone_index_name)
                if overwrite_existing_docs:
                    status_message = f"🔄 '{file_name}': Overwrite enabled. Existing records will be removed."
                    logger.info(f"Plan: Overwrite enabled for '{file_name}'.")
                else:
                    try:
                        dummy = [0.0] * int(embedding_dimension)
                        q = idx.query(vector=dummy, top_k=1, filter={"file_name": file_name}, include_values=False, namespace=(namespace or None))
                        matches = getattr(q, "matches", None) or q.get("matches", [])
                        if matches:
                            should_process = False
                            status_message = f"⏩ '{file_name}': SKIPPED (already present, overwrite disabled)."
                            logger.info(f"Plan: Skipped '{file_name}': already present and overwrite disabled.")
                        else:
                            status_message = f"📄 '{file_name}': Will be processed (not found, or overwrite enabled)."
                            logger.info(f"Plan: '{file_name}' will be processed.")
                    except Exception as e:
                        logger.exception(f"Plan: Presence check for '{file_name}' failed.")
                        status_message = f"⚠️ '{file_name}': Presence check failed ({e}). Will attempt to process."
                        should_process = True
            else:
                status_message = f"📄 '{file_name}': Will be processed (Pinecone index not yet available or provided)."
                logger.info(f"Plan: '{file_name}' will be processed. Pinecone client not ready.")

            plan_summary_messages.append(status_message)

            if should_process:
                files_to_process_plan.append((uploaded_file, document_id))

            with plan_summary_placeholder.container():
                for msg in plan_summary_messages:
                    st.markdown(f"- {msg}")
            time.sleep(0.05)

        logger.info(f"Initial file processing plan generated: {plan_summary_messages}")
        if not files_to_process_plan:
            st.error("No documents selected for processing after initial plan. Please check your files and overwrite settings.")
            logger.error("No documents to process after initial plan. Aborting.")
            return

        # --- PHASE 2: Process Documents with Real-time Updates ---
        total_files_to_process = len(files_to_process_plan)
        st.subheader("Document Processing Progress")

        # Create Pinecone index if it doesn't exist
        if pc and not pc.has_index(pinecone_index_name):
            conf = SUPPORTED_PINECONE_REGIONS.get(pinecone_cloud_region, SUPPORTED_PINECONE_REGIONS["aws-us-east-1"])
            logger.info(f"Pinecone index '{pinecone_index_name}' does not exist. Creating new index.")
            st.info(f"Creating Pinecone index '{pinecone_index_name}'...")
            with st.spinner("Waiting for index to become ready..."):
                pc.create_index(name=pinecone_index_name, dimension=int(embedding_dimension), metric=metric_type, spec=ServerlessSpec(cloud=conf["cloud"], region=conf["region"]))
                while not pc.describe_index(pinecone_index_name).status["ready"]:
                    time.sleep(1)
            st.success(f"Pinecone index '{pinecone_index_name}' is ready.")
            logger.info(f"Pinecone index '{pinecone_index_name}' created and is ready.")

        index = pc.Index(pinecone_index_name) if pc and pc.has_index(pinecone_index_name) else None
        if not index:
            st.error("Pinecone index could not be created or connected. Aborting document processing.")
            logger.error("Pinecone index not available for processing loop. Aborting.")
            return

        for file_idx, (uploaded_file, document_id) in enumerate(files_to_process_plan):
            file_name = uploaded_file.name
            with st.status(f"Processing document: **{file_name}** ({file_idx + 1}/{total_files_to_process})", expanded=True) as status_container:
                temp_dir = None
                try:
                    status_container.update(label=f"Processing document: **{file_name}** - Saving to temporary storage...", state="running")
                    file_path, temp_dir, file_bytes = save_uploaded_file_to_temp(uploaded_file)
                    if not file_path:
                        raise Exception("Failed to save uploaded file.")

                    # Delete existing records if overwrite is enabled
                    if overwrite_existing_docs:
                        status_container.update(label=f"Processing document: **{file_name}** - Deleting existing records...", state="running")
                        try:
                            index.delete(filter={"file_name": file_name}, namespace=(namespace or None))
                            status_container.write(f"✅ Existing records for '{file_name}' removed.")
                            logger.info(f"Existing records for '{file_name}' deleted from Pinecone during processing.")
                        except Exception as e:
                            status_container.write(f"⚠️ Failed to delete existing records for '{file_name}': {e}")
                            logger.exception(f"Failed to delete existing records for '{file_name}' during processing.")

                    status_container.update(label=f"Processing document: **{file_name}** - Loading content...", state="running")
                    loader = UnstructuredLoader(file_path, strategy=unstructured_strategy)
                    docs = list(loader.lazy_load())
                    status_container.write(f"Loaded {len(docs)} raw parts using '{unstructured_strategy}' strategy.")
                    logger.debug(f"UnstructuredLoader loaded {len(docs)} raw parts for '{file_name}' with strategy '{unstructured_strategy}'.")

                    meaningful_chunks = []
                    status_container.update(label=f"Processing document: **{file_name}** - Filtering content...", state="running")
                    for d in docs:
                        text = (d.page_content or "").strip()
                        category = d.metadata.get("category")
                        keep = False

                        if not enable_filtering:
                            if text: keep = True
                            else: continue
                        else:
                            if text:
                                if any(keyword in text.lower() for keyword in whitelisted_keywords_set):
                                    keep = True
                                elif category in STRUCTURAL_CUES_AND_CRITICAL_CONCISE_CONTENT:
                                    keep = True
                                elif len(text) >= min_generic_content_length_ui:
                                    keep = True
                                elif enable_ner_filtering and nlp is not None:
                                    doc_spacy = nlp(text)
                                    if doc_spacy.ents:
                                        keep = True
                                        logger.debug(f"Kept short generic chunk due to NER: '{text[:50]}...' Entities: {[ent.text for ent in doc_spacy.ents]}")

                        if keep:
                            d.metadata["document_id"] = document_id
                            d.metadata["file_name"] = file_name
                            d.metadata["original_file_path"] = file_path
                            meaningful_chunks.append(d)
                        else:
                            logger.debug(f"Filtered out chunk from '{file_name}' (category: {category}, length: {len(text)}) due to filtering rules.")

                    status_container.write(f"Filtered to {len(meaningful_chunks)} meaningful parts.")
                    logger.info(f"'{file_name}': {len(docs)} raw parts, {len(meaningful_chunks)} meaningful parts kept after filtering.")

                    if not meaningful_chunks:
                        status_container.write(f"⚠️ No meaningful content found for '{file_name}' after filtering. This document will not be embedded.")
                        logger.warning(f"No meaningful content for '{file_name}' after filtering.")
                        status_container.update(label=f"Document: **{file_name}** - No meaningful content!", state="warning", expanded=False)
                        continue

                    # Layer 1: Adaptive Document Structure Analysis
                    status_container.update(label=f"Processing document: **{file_name}** - Analyzing document structure (Layer 1)...", state="running")
                    doc_context = _analyze_document_structure(meaningful_chunks, logger)
                    status_container.write(f"✅ Layer 1: Document structure analyzed. Inferred type: '{doc_context.get('inferred_document_type', 'N/A')}'.")
                    logger.info(f"'{file_name}': Document structure analysis complete. Context: {doc_context}")

                    # Layer 2: Universal Heading Detection & Hierarchy Extraction
                    status_container.update(label=f"Processing document: **{file_name}** - Extracting document outline (Layer 2)...", state="running")
                    document_outline = _extract_document_outline(
                        meaningful_chunks,
                        doc_context,
                        heading_detection_confidence_threshold_ui, # Pass new UI param
                        min_chunks_per_section_for_merge_ui,      # Pass new UI param
                        logger
                    )
                    status_container.write(f"✅ Layer 2: Document outline extracted. Detected {len(document_outline)} sections.")
                    logger.info(f"'{file_name}': Document outline extracted. Outline: {document_outline}")

                    # Layer 3: Adaptive Hierarchical Semantic Chunking
                    status_container.update(label=f"Processing document: **{file_name}** - Generating semantic chunks (Layer 3)...", state="running")
                    file_specific_final_chunks = _generate_semantic_chunks(
                        meaningful_chunks=meaningful_chunks,
                        document_outline=document_outline,
                        user_chunk_size=chunk_size, # User-defined chunk size
                        user_chunk_overlap=chunk_overlap, # User-defined chunk overlap
                        pinecone_metadata_max_bytes=PINECONE_METADATA_MAX_BYTES,
                        parsed_global_custom_metadata=parsed_global_custom_metadata,
                        document_specific_metadata_map=document_specific_metadata_map,
                        document_id=document_id,
                        file_name=file_name,
                        sentence_split_threshold_chars=sentence_split_threshold_chars_ui, # Pass new UI param
                        logger=logger
                    )
                    status_container.write(f"✅ Layer 3: Final semantic chunks generated: {len(file_specific_final_chunks)} chunks.")
                    logger.info(f"'{file_name}': Generated {len(file_specific_final_chunks)} final semantic chunks.")

                    # Embedding and Upserting for the current file's chunks
                    status_container.update(label=f"Processing document: **{file_name}** - Generating embeddings...", state="running")
                    embed_model = OpenAIEmbeddings(openai_api_key=embedding_api_key, model=embedding_model_name, dimensions=int(embedding_dimension))

                    file_texts_to_embed = [c.page_content for c in file_specific_final_chunks]
                    file_vectors = []
                    current_batch_texts = []
                    current_batch_tokens = 0

                    for i, text in enumerate(file_texts_to_embed):
                        text_tokens = count_tokens(text, embedding_model_name, logger)

                        if (current_batch_tokens + text_tokens > OPENAI_MAX_TOKENS_PER_EMBEDDING_REQUEST) or \
                           (len(current_batch_texts) >= OPENAI_MAX_TEXTS_PER_EMBEDDING_REQUEST):

                            if current_batch_texts:
                                logger.debug(f"Embedding batch of {len(current_batch_texts)} texts ({current_batch_tokens} tokens) for '{file_name}'.")
                                batch_vectors = embed_model.embed_documents(current_batch_texts)
                                file_vectors.extend(batch_vectors)

                            current_batch_texts = [text]
                            current_batch_tokens = text_tokens
                        else:
                            current_batch_texts.append(text)
                            current_batch_tokens += text_tokens

                        status_container.write(f"Generating embeddings for '{file_name}': {i+1}/{len(file_texts_to_embed)} chunks.")

                    if current_batch_texts:
                        logger.debug(f"Embedding final batch of {len(current_batch_texts)} texts ({current_batch_tokens} tokens) for '{file_name}'.")
                        batch_vectors = embed_model.embed_documents(current_batch_texts)
                        file_vectors.extend(batch_vectors)

                    status_container.write(f"Generated {len(file_vectors)} embeddings for '{file_name}'.")
                    logger.info(f"Generated {len(file_vectors)} embeddings for '{file_name}'.")

                    if len(file_vectors) != len(file_specific_final_chunks):
                        raise ValueError(f"Mismatch: Expected {len(file_specific_final_chunks)} embeddings but got {len(file_vectors)} for '{file_name}'.")

                    file_records = []
                    for c, vec in zip(file_specific_final_chunks, file_vectors):
                        # The metadata 'c.metadata' should already be clean and pruned from Layer 3.
                        # This loop acts as a final safeguard.
                        
                        # Ensure chunk_id is present (it should be from Layer 3)
                        chunk_id = c.metadata.get("chunk_id")
                        if not chunk_id:
                            # Fallback if somehow chunk_id is missing (should not happen with Layer 3 changes)
                            chunk_id = deterministic_chunk_id(document_id, c.page_content or "", c.metadata.get("page_number", ""), c.metadata.get("start_index", ""))
                            logger.warning(f"Chunk ID missing from Layer 3 output. Generated fallback ID: {chunk_id}")
                            c.metadata["chunk_id"] = chunk_id # Add to metadata for consistency

                        # Ensure 'text' field is present (it should be from Layer 3)
                        if "text" not in c.metadata:
                             c.metadata["text"] = c.page_content
                             logger.warning(f"Metadata 'text' field missing for chunk '{chunk_id}'. Added from page_content.")

                        # Final shrink check: This should ideally not trigger if Layer 3 worked correctly,
                        # but it's a critical last line of defense against oversized metadata.
                        final_metadata_for_upsert = _shrink_metadata_to_limit(c.metadata, logger, PINECONE_METADATA_MAX_BYTES)
                        
                        # If the text was truncated here, it means something went wrong earlier,
                        # or the original chunk was already too large.
                        if final_metadata_for_upsert.get("text_truncated", False):
                            logger.critical(f"CRITICAL: Chunk '{chunk_id}' text was truncated AGAIN at upsert stage. This indicates a serious issue in prior metadata handling or chunk size.")
                            status_container.write(f"❌ Critical: Chunk '{chunk_id}' text truncated at final upsert. Review logs.")

                        file_records.append((chunk_id, vec, final_metadata_for_upsert))
                    logger.info(f"Prepared {len(file_records)} records for upsert for '{file_name}'.")

                    status_container.update(label=f"Processing document: **{file_name}** - Upserting to Pinecone...", state="running")
                    total_file_records = len(file_records)
                    for i in range(0, total_file_records, UPSERT_BATCH_SIZE):
                        batch = file_records[i : i + UPSERT_BATCH_SIZE]
                        index.upsert(vectors=batch, namespace=(namespace or None))
                        status_container.write(f"Upserted batch {i//UPSERT_BATCH_SIZE + 1}/{(total_file_records + UPSERT_BATCH_SIZE - 1) // UPSERT_BATCH_SIZE} ({len(batch)} records) for '{file_name}'.")
                        logger.debug(f"Upserted batch for '{file_name}': {i//UPSERT_BATCH_SIZE + 1}/{(total_file_records + UPSERT_BATCH_SIZE - 1) // UPSERT_BATCH_SIZE}.")

                    status_container.update(label=f"Document: **{file_name}** processed successfully!", state="complete", expanded=False)
                    logger.info(f"Successfully processed and upserted all records for '{file_name}'.")

                except Exception as e:
                    logger.exception(f"Error processing document '{file_name}'.")
                    status_container.error(f"Error processing '{file_name}': {e}")
                    status_container.update(label=f"Document: **{file_name}** failed!", state="error", expanded=True)
                finally:
                    if temp_dir and os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                        logger.debug(f"Cleaned up temp directory for '{file_name}': {temp_dir}")

        st.success("All selected documents have been processed (or skipped as per plan).")
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
    manage_documents_ui(pinecone_api_key, pinecone_index_name, namespace, int(embedding_dimension))

    # Application Logs section
    st.markdown("---")
    st.header("Application Logs")
    with st.expander("View Logs", expanded=False):
        log_display_area = st.empty()
        log_display_area.code("\n".join(reversed(st.session_state.streamlit_handler.get_records())))
        if st.button("Clear Logs", key="clear_logs_button"):
            st.session_state.streamlit_handler.log_records.clear()
            log_display_area.code("")
            logger.info("Application logs cleared by user.")

if __name__ == "__main__":
    main()
