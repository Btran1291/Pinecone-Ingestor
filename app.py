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
    "CUSTOM_METADATA": "[]",
    "OVERWRITE_EXISTING_DOCS": "False",
    "LOGGING_LEVEL": "INFO",
    "ENABLE_FILTERING": "True",
    "WHITELISTED_KEYWORDS": "",
    "MIN_GENERIC_CONTENT_LENGTH": "50",
    "ENABLE_NER_FILTERING": "True",
    "UNSTRUCTURED_STRATEGY": "hi_res",
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
IMPORTANT_UNSTRUCTURED_CATEGORIES = {
    "Title", "NarrativeText", "ListItem", "Table", "FigureCaption", "Formula", "Text"
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
    except Exception as e:
        logger.exception(f"Error saving uploaded file {uploaded_file.name}")
        st.error(f"Error saving uploaded file: {e}")
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
                if st.button("🗑️", key=f"remove_meta_{i}", help="Remove this custom field"):
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
                                elif category in IMPORTANT_UNSTRUCTURED_CATEGORIES:
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

                    status_container.update(label=f"Processing document: **{file_name}** - Preparing for chunking...", state="running")
                    # Dynamic chunking logic to ensure Pinecone metadata byte limit is respected
                    file_names_for_overhead_estimation = [f[0].name for f in files_to_process_plan]
                    base_metadata_overhead_bytes = get_estimated_base_metadata_overhead(
                        parsed_global_custom_metadata,
                        document_specific_metadata_map,
                        file_names_for_overhead_estimation,
                        logger
                    )
                    logger.info(f"Estimated base metadata overhead: {base_metadata_overhead_bytes} bytes.")

                    MAX_TEXT_BYTES_PER_CHUNK_PINE = PINECONE_METADATA_MAX_BYTES - base_metadata_overhead_bytes - SAFETY_BUFFER_BYTES
                    MAX_SAFE_CHARS_FOR_SPLITTER_PINE = max(MIN_RE_SPLIT_CHUNK_SIZE, MAX_TEXT_BYTES_PER_CHUNK_PINE // MAX_UTF8_BYTES_PER_CHAR)
                    logger.info(f"Calculated MAX_TEXT_BYTES_PER_CHUNK (Pinecone): {MAX_TEXT_BYTES_PER_CHUNK_PINE} bytes.")
                    logger.info(f"Calculated MAX_SAFE_CHARS_FOR_SPLITTER (Pinecone): {MAX_SAFE_CHARS_FOR_SPLITTER_PINE} characters.")

                    initial_splitter_chunk_size = min(chunk_size, MAX_SAFE_CHARS_FOR_SPLITTER_PINE)

                    if chunk_size > initial_splitter_chunk_size:
                        status_container.write(
                            f"⚠️ Requested chunk size of {chunk_size} chars was too large. "
                            f"Adjusted to {initial_splitter_chunk_size} chars for initial split due to Pinecone's metadata limit."
                        )
                        logger.warning(f"User chunk_size {chunk_size} adjusted to {initial_splitter_chunk_size} for initial split due to Pinecone's metadata byte limit.")
                    else:
                        status_container.write(f"Using chunk size {chunk_size} for initial split.")
                        logger.info(f"Using user-defined chunk size {chunk_size} for initial split.")

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=initial_splitter_chunk_size,
                        chunk_overlap=chunk_overlap,
                        length_function=len,
                        add_start_index=True
                    )
                    initial_chunks = splitter.split_documents(meaningful_chunks)
                    status_container.write(f"Initially split into {len(initial_chunks)} chunks.")
                    logger.info(f"Document '{file_name}' initially split into {len(initial_chunks)} chunks.")

                    # Post-split validation and dynamic re-chunking loop for Pinecone byte limit
                    file_specific_final_chunks = []
                    re_chunked_byte_count = 0

                    status_container.update(label=f"Processing document: **{file_name}** - Validating chunks...", state="running")
                    for i, chunk in enumerate(initial_chunks):
                        provisional_metadata = {}

                        for key, value in parsed_global_custom_metadata.items():
                            if key not in RESERVED_METADATA_KEYS:
                                provisional_metadata[key] = value

                        current_file_name_in_chunk = chunk.metadata.get("file_name")
                        if current_file_name_in_chunk and current_file_name_in_chunk in document_specific_metadata_map:
                            doc_specific_md = document_specific_metadata_map[current_file_name_in_chunk]
                            for k, v in doc_specific_md.items():
                                if k not in RESERVED_METADATA_KEYS:
                                    provisional_metadata[k] = v

                        page_num = chunk.metadata.get("page_number", chunk.metadata.get("page", ""))
                        start_idx = chunk.metadata.get("start_index", "")

                        provisional_metadata["document_id"] = document_id
                        provisional_metadata["file_name"] = current_file_name_in_chunk if current_file_name_in_chunk else ""
                        provisional_metadata["page_number"] = page_num if page_num else ""
                        provisional_metadata["start_index"] = start_idx if start_idx else ""
                        if "category" in chunk.metadata:
                            provisional_metadata["category"] = chunk.metadata["category"]
                        if "original_file_path" in chunk.metadata:
                            provisional_metadata["original_file_path"] = chunk.metadata["original_file_path"]

                        provisional_metadata["text"] = chunk.page_content

                        cleaned_provisional_metadata = {}
                        for k, v in provisional_metadata.items():
                            if isinstance(v, (str, int, float, bool, list, dict)):
                                cleaned_provisional_metadata[k] = v
                            else:
                                cleaned_provisional_metadata[k] = str(v)

                        provisional_metadata_bytes = len(json.dumps(cleaned_provisional_metadata).encode("utf-8"))

                        if provisional_metadata_bytes > PINECONE_METADATA_MAX_BYTES:
                            re_chunked_byte_count += 1
                            logger.warning(
                                f"Chunk {i+1} from '{file_name}' (page {page_num}) "
                                f"is too large for Pinecone metadata ({provisional_metadata_bytes} bytes). "
                                f"Dynamically re-chunking this piece."
                            )
                            status_container.write(
                                f"⚠️ Re-chunking part of '{file_name}' (page {page_num}) to fit Pinecone's metadata limit."
                            )

                            non_text_metadata_bytes = provisional_metadata_bytes - len(chunk.page_content.encode("utf-8"))
                            remaining_space_for_text = PINECONE_METADATA_MAX_BYTES - non_text_metadata_bytes - SAFETY_BUFFER_BYTES
                            remaining_space_for_text = max(0, remaining_space_for_text)
                            new_re_split_chunk_size_chars = max(MIN_RE_SPLIT_CHUNK_SIZE, remaining_space_for_text // MAX_UTF8_BYTES_PER_CHAR)

                            logger.info(f"Re-splitting by bytes with new chunk_size: {new_re_split_chunk_size_chars} characters.")

                            re_splitter_bytes = RecursiveCharacterTextSplitter(
                                chunk_size=new_re_split_chunk_size_chars,
                                chunk_overlap=0,
                                length_function=len,
                                add_start_index=True
                            )
                            temp_doc_for_resplit = Document(page_content=chunk.page_content, metadata=chunk.metadata)
                            sub_chunks_byte_split = re_splitter_bytes.split_documents([temp_doc_for_resplit])

                            for sub_chunk in sub_chunks_byte_split:
                                sub_chunk.metadata.update(chunk.metadata)
                                sub_chunk.metadata["original_file_path"] = chunk.metadata.get("original_file_path", "")
                                file_specific_final_chunks.append(sub_chunk)
                        else:
                            file_specific_final_chunks.append(chunk)

                    if re_chunked_byte_count > 0:
                        status_container.write(f"Completed byte-based re-chunking. {re_chunked_byte_count} chunks were re-split.")
                        logger.info(f"Completed byte-based re-chunking for '{file_name}'. {re_chunked_byte_count} chunks were re-split.")
                    else:
                        status_container.write("No chunks required byte-based re-chunking.")
                        logger.info(f"No chunks required byte-based re-chunking for '{file_name}'.")

                    status_container.write(f"Final chunks for '{file_name}': {len(file_specific_final_chunks)}.")
                    logger.info(f"Final chunks for '{file_name}': {len(file_specific_final_chunks)}.")

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
                        final_metadata = {}

                        for key, value in parsed_global_custom_metadata.items():
                            if key not in RESERVED_METADATA_KEYS:
                                final_metadata[key] = value
                            else:
                                logger.warning(f"Chunk {c.metadata.get('chunk_id', 'N/A')}: Ignored global custom metadata key '{key}' as it is reserved.")

                        current_file_name_in_chunk = c.metadata.get("file_name")
                        if current_file_name_in_chunk and current_file_name_in_chunk in document_specific_metadata_map:
                            doc_specific_md = document_specific_metadata_map[current_file_name_in_chunk]
                            for k, v in doc_specific_md.items():
                                if k not in RESERVED_METADATA_KEYS:
                                    final_metadata[k] = v
                                else:
                                    logger.warning(f"Chunk {c.metadata.get('chunk_id', 'N/A')}: Ignored document-specific metadata key '{k}' as it is reserved.")

                        page_num = c.metadata.get("page_number", c.metadata.get("page", ""))
                        start_idx = c.metadata.get("start_index", "")

                        chunk_id = deterministic_chunk_id(document_id, c.page_content or "", page_num, start_idx)

                        final_metadata["document_id"] = document_id
                        final_metadata["file_name"] = current_file_name_in_chunk if current_file_name_in_chunk else ""
                        final_metadata["text"] = c.page_content
                        if page_num: final_metadata["page_number"] = page_num
                        if start_idx: final_metadata["start_index"] = start_idx
                        if "category" in c.metadata: final_metadata["category"] = c.metadata["category"]
                        if "original_file_path" in c.metadata: final_metadata["original_file_path"] = c.metadata["original_file_path"]

                        cleaned_metadata = {}
                        for k, v in final_metadata.items():
                            if isinstance(v, (str, int, float, bool, list, dict)):
                                cleaned_metadata[k] = v
                            else:
                                cleaned_metadata[k] = str(v)

                        metadata_json_size = len(json.dumps(cleaned_metadata).encode("utf-8"))
                        if metadata_json_size > PINECONE_METADATA_MAX_BYTES:
                            logger.critical(f"CRITICAL ERROR: Metadata for chunk {chunk_id} STILL exceeds {PINECONE_METADATA_MAX_BYTES} bytes ({metadata_json_size} bytes) AFTER re-chunking. This should not happen with current logic. Data might be lost or upsert will fail.")
                            status_container.write(f"❌ Critical Error: Chunk {chunk_id} metadata is still too large. Upsert might fail. Please review logs.")

                        file_records.append((chunk_id, vec, cleaned_metadata))
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
