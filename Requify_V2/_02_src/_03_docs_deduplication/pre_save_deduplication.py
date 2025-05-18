"""
pre_save_deduplication.py

This script provides document-level deduplication for the document processing pipeline.
It performs the following operations:
1. Checks newly parsed documents against existing documents in the LanceDB database
2. Uses vector similarity with high thresholds (0.99+) to identify duplicate pages
3. Detects updated versions of existing documents through partial content matching
4. Compares document metadata and embeddings to identify similarities
5. Classifies pages as new, duplicates, or updates to existing content
6. Provides detailed information about duplicate pages and their source documents

The script works as a pre-processing step before saving content to the database,
ensuring only unique or updated content is saved while maintaining references to
duplicate or previous versions for traceability.
"""
# -------------------------------------------------------------------------------------
# Imports & Setup
# -------------------------------------------------------------------------------------
import os
import sys
import logging
import time
import hashlib
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import lancedb
from dotenv import load_dotenv
from datetime import timedelta

# Add the parent directory to the system path to allow importing modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import _00_utils
_00_utils.setup_project_directory()

# Setup logging with script prefix
class ScriptLogger(logging.LoggerAdapter):
    def __init__(self, logger, prefix):
        super().__init__(logger, {})
        self.prefix = prefix

    def process(self, msg, kwargs):
        return f"{self.prefix}{msg}", kwargs

logger = ScriptLogger(_00_utils.setup_logging(), "[Docs_Deduplication] ")

# Load environment variables
load_dotenv()

# -------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------
OUTPUT_DIR_BASE = "_03_output"  # Define base output directory
LANCEDB_SUBDIR_NAME = "lancedb"  # Subdirectory for LanceDB within _03_output
LANCEDB_TABLE_NAME = "documents"
CHUNKS_TABLE_NAME = "document_chunks"
EMBEDDING_DIMENSION = 1024  # Dimension for e5-large models (must match stable_save_to_lancedb.py)
DUPLICATE_THRESHOLD = 0.995  # Cosine similarity threshold for duplicate pages
# LanceDB returns distance for cosine as 1 - similarity. So, distance_threshold = 1 - SIMILARITY_THRESHOLD
DISTANCE_THRESHOLD = 1.0 - DUPLICATE_THRESHOLD
SIMILAR_THRESHOLD = 0.82  # Reduced from 0.9 to 0.82 to better detect reordered content
MIN_PAGES_TO_SAMPLE = 3  # Minimum number of pages to sample for comparison
MAX_PAGES_TO_SAMPLE = 5  # Maximum number of pages to sample for comparison
VERSION_SIMILARITY_THRESHOLD = 0.82  # Reduced from 0.9 to 0.82 to match SIMILAR_THRESHOLD
INDEX_INITIALIZED_DOCS = False
INDEX_INITIALIZED_CHUNKS = False
CHUNK_DUPLICATION_THRESHOLD = 0.995  # Threshold for considering chunks as duplicates
CHUNK_SIMILARITY_THRESHOLD = 0.82  # Reduced from 0.9 to 0.82 to better detect reordered content

# Set to True to enable more detailed console output for deduplication processes
VERBOSE_DEDUPLICATION_OUTPUT = True

# -------------------------------------------------------------------------------------
# Deduplication Logging Functions
# -------------------------------------------------------------------------------------
def log_document_comparison(doc_id: str, comparison_doc_id: str, similarity: float):
    """Log information about document comparison."""
    if similarity >= DUPLICATE_THRESHOLD:
        logger.info(
            f"📄 Document comparison: {doc_id} is a duplicate of {comparison_doc_id} "
            f"(similarity: {similarity:.4f})",
            extra={"icon": "♻️"}
        )
    elif similarity >= SIMILAR_THRESHOLD:
        logger.info(
            f"📄 Document comparison: {doc_id} is similar to {comparison_doc_id} "
            f"(similarity: {similarity:.4f})",
            extra={"icon": "🔄"}
        )
    else:
        logger.info(
            f"📄 Document comparison: {doc_id} is different from {comparison_doc_id} "
            f"(similarity: {similarity:.4f})",
            extra={"icon": "🆕"}
        )

def log_page_comparison(
    doc_id: str, 
    page_num: int, 
    comparison_doc_id: str, 
    comparison_page_num: int, 
    similarity: float
):
    """Log information about page comparison."""
    page_id = f"{doc_id} p{page_num}"
    comparison_page_id = f"{comparison_doc_id} p{comparison_page_num}"
    
    if similarity >= DUPLICATE_THRESHOLD:
        logger.info(
            f"📃 Page comparison: {page_id} is a duplicate of {comparison_page_id} "
            f"(similarity: {similarity:.4f})",
            extra={"icon": "♻️"}
        )
    elif similarity >= SIMILAR_THRESHOLD:
        logger.info(
            f"📃 Page comparison: {page_id} is similar to {comparison_page_id} "
            f"(similarity: {similarity:.4f})",
            extra={"icon": "🔄"}
        )
    else:
        logger.info(
            f"📃 Page comparison: {page_id} appears to be new "
            f"(similarity: {similarity:.4f} with {comparison_page_id})",
            extra={"icon": "🆕"}
        )

def log_chunk_comparison(
    chunk_id: str, 
    match_id: str = "", 
    similarity: float = 0.0, 
    is_duplicate: bool = False,
    match_type: str = "embedding",
    doc_id: str = "",
    comparison_doc_id: str = ""
):
    """Log information about chunk comparison."""
    doc_context = f"({doc_id})" if doc_id else ""
    comparison_doc_context = f"({comparison_doc_id})" if comparison_doc_id else ""
    
    if match_type == "hash":
        logger.info(
            f"🧩 Chunk comparison: {chunk_id} {doc_context} is an exact hash match with "
            f"{match_id} {comparison_doc_context}",
            extra={"icon": "♻️"}
        )
    elif is_duplicate or similarity >= DUPLICATE_THRESHOLD:
        logger.info(
            f"🧩 Chunk comparison: {chunk_id} {doc_context} is a duplicate of "
            f"{match_id} {comparison_doc_context} "
            f"(similarity: {similarity:.4f})",
            extra={"icon": "♻️"}
        )
    elif similarity >= SIMILAR_THRESHOLD:
        logger.info(
            f"🧩 Chunk comparison: {chunk_id} {doc_context} is similar to "
            f"{match_id} {comparison_doc_context} "
            f"(similarity: {similarity:.4f})",
            extra={"icon": "🔄"}
        )
    else:
        logger.info(
            f"🧩 Chunk comparison: {chunk_id} {doc_context} appears to be new "
            f"(top similarity: {similarity:.4f} with {match_id})",
            extra={"icon": "🆕"}
        )

def log_document_deduplication_summary(
    doc_id: str,
    total_pages: int,
    duplicate_pages: int,
    similar_pages: int,
    new_pages: int,
    is_new_version: bool,
    old_version_id: Optional[str] = None,
    version_similarity: float = 0.0
):
    """Log a summary of document deduplication results."""
    if is_new_version and old_version_id:
        logger.info(
            f"📊 Document deduplication summary: {doc_id} is a new version of {old_version_id} "
            f"(similarity: {version_similarity:.4f})",
            extra={"icon": "🔄"}
        )
    elif duplicate_pages == total_pages:
        logger.info(
            f"📊 Document deduplication summary: {doc_id} is a complete duplicate "
            f"(all {total_pages} pages are duplicates)",
            extra={"icon": "♻️"}
        )
    elif duplicate_pages > 0 or similar_pages > 0:
        logger.info(
            f"📊 Document deduplication summary: {doc_id} has {new_pages} new pages, "
            f"{duplicate_pages} duplicate pages, and {similar_pages} similar pages "
            f"(total: {total_pages} pages)",
            extra={"icon": "🔄"}
        )
    else:
        logger.info(
            f"📊 Document deduplication summary: {doc_id} is completely new "
            f"(all {total_pages} pages are new)",
            extra={"icon": "🆕"}
        )

def log_chunk_deduplication_summary(
    total_chunks: int,
    duplicate_chunks: int,
    similar_chunks: int,
    new_chunks: int,
    processing_time: float = 0.0,
    doc_id: str = "",
    is_document_update: bool = False,
    updated_doc_id: Optional[str] = None
):
    """Log a summary of chunk deduplication results."""
    doc_prefix = f"{doc_id}: " if doc_id else ""
    
    if is_document_update and updated_doc_id:
        logger.info(
            f"📊 Chunk deduplication summary: {doc_prefix}updating chunks from {updated_doc_id} "
            f"({duplicate_chunks} duplicates, {similar_chunks} updates, {new_chunks} new chunks) "
            f"in {processing_time:.2f}s",
            extra={"icon": "🔄"}
        )
    elif duplicate_chunks == total_chunks:
        logger.info(
            f"📊 Chunk deduplication summary: {doc_prefix}only duplicate chunks "
            f"(all {total_chunks} chunks are duplicates) "
            f"in {processing_time:.2f}s",
            extra={"icon": "♻️"}
        )
    elif duplicate_chunks > 0 or similar_chunks > 0:
        logger.info(
            f"📊 Chunk deduplication summary: {doc_prefix}{new_chunks} new chunks, "
            f"{duplicate_chunks} duplicate chunks, and {similar_chunks} similar chunks "
            f"(total: {total_chunks} chunks) "
            f"in {processing_time:.2f}s",
            extra={"icon": "🔄"}
        )
    else:
        logger.info(
            f"📊 Chunk deduplication summary: {doc_prefix}all new chunks "
            f"({new_chunks} chunks) "
            f"in {processing_time:.2f}s",
            extra={"icon": "🆕"}
        )

def log_embedding_similarity(similarity: float, description: str = "Embedding comparison"):
    """Log information about embedding similarity."""
    if similarity >= DUPLICATE_THRESHOLD:
        logger.info(
            f"📏 {description} - Embedding similarity: {similarity:.4f} (DUPLICATE)",
            extra={"icon": "♻️"}
        )
    elif similarity >= SIMILAR_THRESHOLD:
        logger.info(
            f"📏 {description} - Embedding similarity: {similarity:.4f} (SIMILAR)",
            extra={"icon": "🔄"}
        )
    else:
        logger.info(
            f"📏 {description} - Embedding similarity: {similarity:.4f} (DIFFERENT)",
            extra={"icon": "🆕"}
        )

# -------------------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------------------
def connect_to_lancedb(lancedb_path: str):
    """Connect to LanceDB and return the connection."""
    logger.info(f"Connecting to LanceDB at: {lancedb_path}", extra={"icon": "🔄"})
    if not os.path.exists(lancedb_path):
        logger.warning(
            f"LanceDB directory does not exist at {lancedb_path}. It will be created when saving.",
            extra={"icon": "⚠️"}
        )
        return None

    try:
        db = lancedb.connect(lancedb_path)
        logger.info(f"Successfully connected to LanceDB", extra={"icon": "✅"})
        return db
    except Exception as e:
        logger.error(f"Failed to connect to LanceDB: {e}", extra={"icon": "❌"})
        return None


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """Normalize an embedding vector to unit length."""
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def calculate_cosine_similarity(embed1: np.ndarray, embed2: np.ndarray) -> float:
    """Calculate cosine similarity between two embedding vectors."""
    if len(embed1) != len(embed2):
        raise ValueError(
            f"Embedding dimensions don't match: {len(embed1)} vs {len(embed2)}"
        )
    norm_embed1 = normalize_embedding(embed1)
    norm_embed2 = normalize_embedding(embed2)
    similarity = float(np.dot(norm_embed1, norm_embed2))
    
    # Log the embedding similarity if verbose mode is enabled
    if VERBOSE_DEDUPLICATION_OUTPUT:
        log_embedding_similarity(similarity)
        
    return similarity


def create_chunk_hash(chunk_text: str) -> str:
    """
    Create a unique hash for a chunk of text.
    """
    return hashlib.md5(chunk_text.encode('utf-8')).hexdigest()


def ensure_index(db, table_name=None):
    """Ensure ANN index exists on the embedding column."""
    global INDEX_INITIALIZED_DOCS, INDEX_INITIALIZED_CHUNKS
    
    if not db:
        return
        
    if not table_name:
        table_name = LANCEDB_TABLE_NAME
        
    if table_name not in db.table_names():
        return
        
    # Check if this table's index is already initialized
    if (table_name == LANCEDB_TABLE_NAME and INDEX_INITIALIZED_DOCS) or \
       (table_name == CHUNKS_TABLE_NAME and INDEX_INITIALIZED_CHUNKS):
        return
    
    table = db.open_table(table_name)
    
    # Check if there are enough rows to build the index (minimum 256 required)
    row_count = len(table.to_pandas())
    
    if row_count < 256:
        logger.info(f"Not creating index: {table_name} has only {row_count} rows, minimum 256 required", extra={"icon": "⚠️"})
        return
        
    table.create_index(
        metric="cosine",
        vector_column_name="embedding"
    )
    
    if table_name == LANCEDB_TABLE_NAME:
        INDEX_INITIALIZED_DOCS = True
    elif table_name == CHUNKS_TABLE_NAME:
        INDEX_INITIALIZED_CHUNKS = True
        
    logger.info(f"Built ANN index on embedding column for {table_name}", extra={"icon": "✅"})

# -------------------------------------------------------------------------------------
# Chunk Deduplication Logic
# -------------------------------------------------------------------------------------
def check_chunk_duplicates(
    chunks_data: List[Dict], db_connection=None
) -> Tuple[Dict[int, Dict[str, object]], List[int], Dict[int, Dict[str, object]]]:
    """
    Check if newly created chunks have duplicates in the existing database.
    
    Returns:
        duplicate_chunks: dict mapping chunk idx -> {'similar_id', 'similarity', 'hash_match'}
        new_chunks: list of new chunk indices
        update_chunks: dict mapping chunk idx -> {'record_id', 'is_newer'}
    """
    start_time = time.time()
    if not chunks_data:
        logger.warning("Empty chunks data provided", extra={"icon": "⚠️"})
        return {}, [], {}
    
    doc_id = chunks_data[0].get('document_id', 'unknown')
    
    if VERBOSE_DEDUPLICATION_OUTPUT:
        logger.info("\n" + "=" * 80, extra={"icon": "🧩"})
        logger.info(f"CHUNK DEDUPLICATION PROCESS FOR DOCUMENT: {doc_id}", extra={"icon": "🧩"})
        logger.info("=" * 80, extra={"icon": "🧩"})
    
    logger.info(f"Checking for duplicate chunks in document: {doc_id}", extra={"icon": "🔄"})
    
    # Connect to LanceDB
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    lancedb_path = os.path.join(project_root, OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)
    db = db_connection or connect_to_lancedb(lancedb_path)
    
    if not db or CHUNKS_TABLE_NAME not in db.table_names():
        logger.info(
            f"No existing chunks table found. All {len(chunks_data)} chunks are new.",
            extra={"icon": "✅"}
        )
        if VERBOSE_DEDUPLICATION_OUTPUT:
            logger.info(f"✅ No existing chunks table found. All {len(chunks_data)} chunks are new.", extra={"icon": "✅"})
        return {}, list(range(len(chunks_data))), {}
    
    # Ensure index exists if we have enough chunks
    ensure_index(db, CHUNKS_TABLE_NAME)
    chunks_table = db.open_table(CHUNKS_TABLE_NAME)
    
    # Get a dataframe of all existing chunks for hash comparison
    existing_chunks_df = chunks_table.to_pandas()
    
    if VERBOSE_DEDUPLICATION_OUTPUT and not existing_chunks_df.empty:
        logger.info(f"📊 Found {len(existing_chunks_df)} existing chunks to compare against", extra={"icon": "🔍"})
    
    duplicate_chunks = {}
    new_chunks = []
    update_chunks = {}
    
    for idx, chunk_data in enumerate(chunks_data):
        chunk_id = chunk_data.get('chunk_id', f"chunk_{idx}")
        chunk_text = chunk_data.get('chunk_text', '')
        chunk_hash = create_chunk_hash(chunk_text)
        
        # Add hash to chunk data for future use
        chunks_data[idx]['chunk_hash'] = chunk_hash
        
        # First check for hash-based duplicates (exact matches)
        # Safely check for chunk_hash in dataframe columns
        if 'chunk_hash' in existing_chunks_df.columns:
            hash_matches = existing_chunks_df[existing_chunks_df['chunk_hash'] == chunk_hash]
            
            if not hash_matches.empty:
                # We have an exact duplicate based on hash
                match_id = hash_matches.iloc[0].get('chunk_id', '')
                duplicate_chunks[idx] = {
                    'similar_id': match_id,
                    'similarity': 1.0,  # Perfect match
                    'hash_match': True
                }
                
                # Log the duplicate with enhanced visibility
                log_chunk_comparison(
                    chunk_id=chunk_id,
                    match_id=match_id,
                    similarity=1.0,
                    is_duplicate=True,
                    match_type="hash"
                )
                
                continue
        else:
            logger.warning(f"No 'chunk_hash' column in existing chunks table. Skipping hash-based deduplication.", extra={"icon": "⚠️"})
        
        # If no hash match, check for vector similarity
        embedding = chunk_data.get('embedding')
        if embedding is None:
            logger.warning(f"Chunk {chunk_id} has no embedding. Marking as new.", extra={"icon": "⚠️"})
            new_chunks.append(idx)
            continue
            
        if isinstance(embedding, list):
            embedding = np.array(embedding)
            
        # Attempt vector similarity search with robust error handling
        try:
            # First check if the schema has embedding column
            has_embedding_column = False
            for field in chunks_table.schema:
                if field.name == 'embedding':
                    has_embedding_column = True
                    break
                    
            if not has_embedding_column:
                logger.warning(f"No embedding column found in chunks table. Using direct comparison.", extra={"icon": "⚠️"})
                
                # Fall back to direct comparison without vector search
                best_match = None
                best_similarity = 0.0
                
                for _, row in existing_chunks_df.iterrows():
                    existing_emb = row.get('embedding')
                    if existing_emb is not None:
                        if isinstance(existing_emb, list):
                            existing_emb = np.array(existing_emb)
                            
                        try:
                            similarity = calculate_cosine_similarity(embedding, existing_emb)
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_match = row
                        except Exception:
                            continue
                            
                if best_match is not None and best_similarity >= SIMILAR_THRESHOLD:
                    match_id = best_match.get('chunk_id', '')
                    
                    log_chunk_comparison(
                        chunk_id=chunk_id,
                        match_id=match_id,
                        similarity=best_similarity,
                        is_duplicate=best_similarity >= CHUNK_DUPLICATION_THRESHOLD
                    )
                    
                    if best_similarity >= CHUNK_DUPLICATION_THRESHOLD:
                        duplicate_chunks[idx] = {
                            'similar_id': match_id,
                            'similarity': best_similarity,
                            'hash_match': False
                        }
                    elif best_similarity >= SIMILAR_THRESHOLD:
                        update_chunks[idx] = {
                            'similar_id': match_id,
                            'similarity': best_similarity
                        }
                    else:
                        new_chunks.append(idx)
                else:
                    new_chunks.append(idx)
                    if VERBOSE_DEDUPLICATION_OUTPUT:
                        logger.info(f"🆕 Chunk {chunk_id} has no similar chunks. Marking as new.", extra={"icon": "🆕"})
                continue
            
            # Use vector search with explicit column specification
            query_result = chunks_table.search(
                embedding, 
                vector_column_name="embedding"
            ).limit(5).to_pandas()
            
            if query_result.empty:
                if VERBOSE_DEDUPLICATION_OUTPUT:
                    logger.info(f"🆕 Chunk {chunk_id} has no similar chunks. Marking as new.", extra={"icon": "🆕"})
                new_chunks.append(idx)
                continue
                
            # Calculate similarity (1.0 - distance for cosine similarity)
            best_match = query_result.iloc[0]
            similarity = 1.0 - best_match['_distance']
            match_id = best_match.get('chunk_id', '')
            
            # Log detailed chunk comparison
            log_chunk_comparison(
                chunk_id=chunk_id,
                match_id=match_id,
                similarity=similarity,
                is_duplicate=similarity >= CHUNK_DUPLICATION_THRESHOLD
            )
            
            if similarity >= CHUNK_DUPLICATION_THRESHOLD:
                # This is a duplicate chunk
                duplicate_chunks[idx] = {
                    'similar_id': match_id,
                    'similarity': similarity,
                    'hash_match': False
                }
            elif similarity >= CHUNK_SIMILARITY_THRESHOLD:
                # This might be an updated version of an existing chunk
                update_chunks[idx] = {
                    'similar_id': match_id,
                    'similarity': similarity
                }
            else:
                # This is a new chunk
                new_chunks.append(idx)
                
        except Exception as e:
            logger.error(f"Error during vector search for chunk {chunk_id}: {str(e)}", extra={"icon": "❌"})
            if VERBOSE_DEDUPLICATION_OUTPUT:
                logger.error(f"Error during vector search: {str(e)}", extra={"icon": "❌"})
                logger.info(f"Marking chunk {chunk_id} as new due to search error.", extra={"icon": "🆕"})
            new_chunks.append(idx)
    
    # Log summary of chunk deduplication
    processing_time = time.time() - start_time
    log_chunk_deduplication_summary(
        total_chunks=len(chunks_data),
        duplicate_chunks=len(duplicate_chunks),
        similar_chunks=len(update_chunks),
        new_chunks=len(new_chunks),
        processing_time=processing_time
    )
    
    # Print more detailed summary in verbose mode
    if VERBOSE_DEDUPLICATION_OUTPUT:
        logger.info("\n" + "-" * 50, extra={"icon": "📊"})
        logger.info(f"CHUNK DEDUPLICATION SUMMARY:", extra={"icon": "📊"})
        logger.info(f"  • Total chunks processed: {len(chunks_data)}", extra={"icon": "📊"})
        logger.info(f"  • Exact duplicates found: {len(duplicate_chunks)}", extra={"icon": "📊"})
        logger.info(f"  • Similar chunks found: {len(update_chunks)}", extra={"icon": "📊"})
        logger.info(f"  • New unique chunks: {len(new_chunks)}", extra={"icon": "📊"})
        logger.info(f"  • Processing time: {processing_time:.2f} seconds", extra={"icon": "📊"})
        logger.info("-" * 50 + "\n", extra={"icon": "📊"})
    
    return duplicate_chunks, new_chunks, update_chunks

# -------------------------------------------------------------------------------------
# Main Deduplication Logic
# -------------------------------------------------------------------------------------
def check_document_duplicates(
    new_doc_data: List[Dict], db_connection=None
) -> Tuple[Dict[int, Dict[str, object]], List[int], Dict[int, Dict[str, object]]]:
    """
    Check if a newly parsed document has duplicate pages in the existing database.

    Returns:
        duplicate_pages: dict mapping page idx -> {{'similar_id', 'similarity'}}
        new_pages: list of new page indices
        update_pages: dict mapping page idx -> {{'record_id', 'is_newer'}}
    """
    start_time = time.time()
    if not new_doc_data:
        logger.warning("Empty document data provided", extra={"icon": "⚠️"})
        return {}, [], {}

    doc_id = new_doc_data[0].get('pdf_identifier', 'unknown')
    logger.info(f"Checking for duplicates of document: {doc_id}", extra={"icon": "🔄"})

    # Connect and index
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    lancedb_path = os.path.join(project_root, OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)
    db = db_connection or connect_to_lancedb(lancedb_path)
    if not db or LANCEDB_TABLE_NAME not in db.table_names():
        logger.info(
            f"No existing database or table found. All {len(new_doc_data)} pages are new.",
            extra={"icon": "✅"}
        )
        return {}, list(range(len(new_doc_data))), {}

    ensure_index(db)
    table = db.open_table(LANCEDB_TABLE_NAME)

    duplicate_pages: Dict[int, Dict[str, object]] = {}
    new_pages: List[int] = []
    update_pages: Dict[int, Dict[str, object]] = {}

    for idx, page_data in enumerate(new_doc_data):
        page_num = page_data.get('page_number', idx + 1)
        embedding = page_data.get('embedding')
        if embedding is None:
            logger.warning(f"Page {page_num} has no embedding. Marking as new.", extra={"icon": "⚠️"})
            new_pages.append(idx)
            continue
        if isinstance(embedding, list):
            embedding = np.array(embedding)

        try:
            df = (
                table.search(embedding)
                     .metric("cosine")
                     .limit(MAX_PAGES_TO_SAMPLE)
                     .nprobes(32)
                     .refine_factor(5)
                     .to_df()
            )
            if df.empty:
                logger.info(f"No similar pages found for page {page_num}. It's new.", extra={"icon": "✅"})
                new_pages.append(idx)
                continue

            found = False
            for _, row in df.iterrows():
                sim = 1.0 - row['_distance']
                existing_id = row['pdf_identifier']
                existing_page = row.get('page_number', 'unknown')

                # Same-doc update check
                if existing_id == doc_id and existing_page == page_num:
                    exist_ts = pd.to_datetime(row.get('timestamp', None))
                    new_ts = pd.to_datetime(page_data.get('timestamp', None))
                    if new_ts and exist_ts and new_ts > exist_ts:
                        update_pages[idx] = {'record_id': row.name, 'is_newer': True}
                        logger.info(
                            f"Page {page_num} is a newer version. Marked for update.",
                            extra={"icon": "🔄"}
                        )
                    else:
                        duplicate_pages[idx] = {'similar_id': f"{existing_id}_{existing_page}", 'similarity': sim}
                        logger.info(
                            f"Page {page_num} is an older version. Skipping.",
                            extra={"icon": "⏩"}
                        )
                    found = True
                    break

                # Cross-doc duplicate
                if sim >= DUPLICATE_THRESHOLD:
                    duplicate_pages[idx] = {'similar_id': f"{existing_id}_{existing_page}", 'similarity': sim}
                    logger.info(
                        f"Page {page_num} duplicates {existing_id} page {existing_page} (sim={sim:.4f}). Skipping.",
                        extra={"icon": "⏩"}
                    )
                    found = True
                    break

            if not found:
                logger.info(f"Page {page_num} has no close matches. It's new.", extra={"icon": "✅"})
                new_pages.append(idx)

        except Exception as e:
            logger.error(f"Error searching for page {page_num}: {e}", extra={"icon": "❌"})
            new_pages.append(idx)

    elapsed = time.time() - start_time
    logger.info(
        f"Duplicate check completed in {elapsed:.2f}s: {len(new_pages)} new, {len(duplicate_pages)} dup, {len(update_pages)} updates",
        extra={"icon": "📊"}
    )
    return duplicate_pages, new_pages, update_pages


def get_document_pages_by_id(doc_id: str, db_connection=None) -> pd.DataFrame:
    """
    Get all pages for a specific document ID from the database.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    lancedb_path = os.path.join(project_root, OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)
    db = db_connection or connect_to_lancedb(lancedb_path)
    if not db or LANCEDB_TABLE_NAME not in db.table_names():
        logger.warning(
            f"No existing database or table found for document {doc_id}",
            extra={"icon": "⚠️"}
        )
        return pd.DataFrame()
    ensure_index(db)
    table = db.open_table(LANCEDB_TABLE_NAME)
    try:
        df = table.to_pandas()
        res = df[df['pdf_identifier'] == doc_id]
        logger.info(f"Found {len(res)} pages for {doc_id}", extra={"icon": "✅"})
        return res
    except Exception as e:
        logger.error(f"Error retrieving {doc_id}: {e}", extra={"icon": "❌"})
        return pd.DataFrame()


def check_for_document_version_update(
    new_doc_data: List[Dict], db_connection=None
) -> Tuple[bool, float, Optional[str]]:
    """
    Check if a document is a new version of an existing document via ANN.
    """
    if not new_doc_data:
        return False, 0.0, None
    doc_id = new_doc_data[0].get('pdf_identifier', 'unknown')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    lancedb_path = os.path.join(project_root, OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)
    db = db_connection or connect_to_lancedb(lancedb_path)
    if not db or LANCEDB_TABLE_NAME not in db.table_names():
        return False, 0.0, None
    ensure_index(db)
    table = db.open_table(LANCEDB_TABLE_NAME)

    sim_map: Dict[str, List[float]] = {}
    samples = new_doc_data[:min(MAX_PAGES_TO_SAMPLE, len(new_doc_data))]
    embeddings = [np.array(p['embedding']) for p in samples if p.get('embedding')]
    for emb in embeddings:
        try:
            df = (
                table.search(emb)
                     .metric("cosine")
                     .limit(MAX_PAGES_TO_SAMPLE)
                     .nprobes(32)
                     .refine_factor(5)
                     .to_df()
            )
            for _, row in df.iterrows():
                pid = row['pdf_identifier']
                if pid == doc_id:
                    continue
                sim = 1.0 - row['_distance']
                sim_map.setdefault(pid, []).append(sim)
        except Exception as e:
            logger.error(f"Error during version search: {e}", extra={"icon": "❌"})

    if not sim_map:
        return False, 0.0, None
    avg_sims = {pid: sum(v)/len(v) for pid, v in sim_map.items()}
    best_id, best_sim = max(avg_sims.items(), key=lambda x: x[1])
    is_new = VERSION_SIMILARITY_THRESHOLD <= best_sim < DUPLICATE_THRESHOLD
    if is_new:
        logger.info(
            f"Document {doc_id} appears to be new version of {best_id} (sim={best_sim:.4f})",
            extra={"icon": "🔄"}
        )
    return is_new, best_sim, best_id

# -------------------------------------------------------------------------------------
# Main Function
# -------------------------------------------------------------------------------------
def check_new_document(doc_data: List[Dict]) -> Dict[str, object]:
    """
    Check if a document is already in the database and return detailed information.
    
    Args:
        doc_data: List of page dictionaries with document information
        
    Returns:
        Dictionary with detailed duplicate/similarity information
    """
    if not doc_data:
        logger.warning("Empty document data provided", extra={"icon": "⚠️"})
        return {
            "is_duplicate": False,
            "duplicate_pages": {},
            "new_pages": [],
            "update_pages": {},
            "is_new_version": False,
            "old_version_id": None,
            "version_similarity": 0.0
        }
    
    # Connect to LanceDB
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    lancedb_path = os.path.join(project_root, OUTPUT_DIR_BASE, LANCEDB_SUBDIR_NAME)
    db = connect_to_lancedb(lancedb_path)
    
    # Get document ID and log the check
    doc_id = doc_data[0].get('pdf_identifier', 'unknown')
    logger.info(f"Checking for duplicate/similar documents: {doc_id}", extra={"icon": "🔍"})
    
    # Check duplicate pages first with existing document database
    duplicate_pages, new_pages, update_pages = check_document_duplicates(doc_data, db)
    
    # Calculate summary counts
    total_pages = len(doc_data)
    duplicate_count = len(duplicate_pages)
    update_count = len(update_pages)
    new_count = len(new_pages)
    
    # Check if this is a complete duplicate (all pages are duplicates)
    is_complete_duplicate = duplicate_count == total_pages
    
    # Check if this might be a new version of an existing document
    is_new_version, version_similarity, old_version_id = check_for_document_version_update(doc_data, db)
    
    # Log detailed summary using enhanced logging
    log_document_deduplication_summary(
        doc_id=doc_id,
        total_pages=total_pages,
        duplicate_pages=duplicate_count,
        similar_pages=update_count,
        new_pages=new_count,
        is_new_version=is_new_version,
        old_version_id=old_version_id,
        version_similarity=version_similarity
    )
    
    # Return comprehensive results
    return {
        "is_duplicate": is_complete_duplicate,
        "duplicate_pages": duplicate_pages,
        "new_pages": new_pages,
        "update_pages": update_pages,
        "is_new_version": is_new_version,
        "old_version_id": old_version_id,
        "version_similarity": version_similarity
    }

if __name__ == "__main__":
    logger.info("This script is designed to be imported and used by the document processing pipeline.", extra={"icon": "ℹ️"})
    logger.info("It checks for duplicates before adding new documents to the database.", extra={"icon": "ℹ️"})
